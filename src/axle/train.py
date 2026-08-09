"""Hydra entry point: train + evaluate one (data, model, loss, protocol) combination.

Example
-------
    python -m axle.train data=germany model=lstm loss=axle protocol=loyo crop=wheat

Runs the protocol's folds, trains each, aggregates best-epoch metrics
(mean +/- std across folds), writes per-fold predictions and a metrics summary,
and optionally logs to Weights & Biases.
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .data.dataset import YieldSATPixels
from .data.patches import YieldSATPatches, load_directions
from .data.splits import SELECTION_KEY, inner_split, make_splits
from .losses import build_loss
from .models import build_model
from .trainer import train_fold
from .utils.seed import set_seed


def run(cfg: DictConfig) -> dict:
    set_seed(cfg.seed)
    import torch
    dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda":
        print(f"[device] cuda | {torch.cuda.device_count()} visible GPU(s): "
              f"{torch.cuda.get_device_name(0)}")
    else:
        print(f"[device] {dev} (no CUDA GPU visible)")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")

    full = YieldSATPixels(cfg.data.cache_dir, nan_fill=cfg.data.nan_fill)
    print(f"[data] {cfg.data.name}: {len(full.meta):,} px | {full.num_features} time-varying "
          f"bands x {full.seq_len} steps" + (f" + {full.num_static} static" if full.num_static else ""))
    meta = full.meta
    keep = np.ones(len(meta), bool)
    if cfg.crop:
        keep &= (meta["crop"] == cfg.crop).to_numpy()
    sub_pos = np.nonzero(keep)[0]
    sub_meta = meta.iloc[sub_pos].reset_index(drop=True)
    splits = make_splits(sub_meta, cfg.protocol.name,
                         **{k: v for k, v in cfg.protocol.items() if k != "name"})

    run_wandb = cfg.wandb.enabled
    if run_wandb:
        import wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                   config=OmegaConf.to_container(cfg, resolve=True), dir=str(out))

    loss_kw = {k: v for k, v in cfg.loss.items() if k != "name"}
    model_kw = {k: v for k, v in cfg.model.items() if k != "name"}
    members = int(cfg.get("ensemble", {}).get("members", 1))

    # AXLE-M2 trains on field patches (correlated NLL); everything else on pixel bags.
    # Validation is per-pixel either way -- the objective changes, not the predictor.
    use_patches = getattr(build_loss(cfg.loss.name, **loss_kw), "requires_patches", False)
    directions, train_bs = None, cfg.train.batch_size
    if use_patches:
        train_bs = cfg.train.patch_batch_size
        directions = load_directions(cfg.data.cache_dir, cfg.patch.directions)
        if directions is None:
            print(f"[patches] no {cfg.data.cache_dir}/directions.parquet -- M2 runs isotropic; "
                  "run scripts/estimate_directions.py to anchor the swath geometry")
        else:
            share = float(directions["has_direction"].mean())
            print(f"[patches] tile={cfg.patch.tile} min_pixels={cfg.patch.min_pixels} | "
                  f"d_f found for {share:.1%} of {len(directions)} fields")

    fold_metrics, all_preds = [], []
    for fold, (tr_local, va_local) in enumerate(splits):
        # map local (sub_meta) positions back to global dataset rows
        # selection set mirrors this protocol's shift (held-out year / farm / fields)
        fit_local, sel_local = inner_split(sub_meta, tr_local,
                                           key=SELECTION_KEY[cfg.protocol.name],
                                           frac=cfg.train.inner_val_frac, seed=cfg.seed)
        tr = sub_pos[fit_local]
        va = sub_pos[va_local]
        select_ds = (YieldSATPixels(cfg.data.cache_dir, indices=sub_pos[sel_local],
                                    nan_fill=cfg.data.nan_fill) if len(sel_local) else None)
        train_ds = (
            YieldSATPatches(cfg.data.cache_dir, indices=tr, nan_fill=cfg.data.nan_fill,
                            tile=cfg.patch.tile, min_pixels=cfg.patch.min_pixels,
                            directions=directions)
            if use_patches else
            YieldSATPixels(cfg.data.cache_dir, indices=tr, nan_fill=cfg.data.nan_fill)
        )
        val_ds = YieldSATPixels(cfg.data.cache_dir, indices=va, nan_fill=cfg.data.nan_fill)

        def build():  # fresh (model, loss) per ensemble member
            loss_fn = build_loss(cfg.loss.name, **loss_kw)
            model = build_model(cfg.model.name, in_dim=full.num_features,
                                predict_variance=loss_fn.predicts_variance,
                                static_dim=full.num_static, **model_kw)
            return model, loss_fn

        tag = f"{cfg.model.name}+{cfg.loss.name}" + (f" x{members}" if members > 1 else "")
        unit = f"{len(train_ds):,} patches" if use_patches else f"{len(tr):,}"
        sel = (f" select={len(sel_local):,}(held-out {SELECTION_KEY[cfg.protocol.name]})"
               if len(sel_local) else " select=none(legacy)")
        print(f"[fold {fold}] train={unit}{sel} val={len(va):,} ({tag}, {cfg.protocol.name})")
        df, m = train_fold(build, train_ds, val_ds, select_ds=select_ds,
                           members=members, seed=cfg.seed,
                           epochs=cfg.train.epochs, batch_size=train_bs,
                           lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
                           grad_clip=cfg.train.grad_clip, num_workers=cfg.num_workers,
                           device=cfg.device, log_every=cfg.train.log_every)
        m["fold"] = fold
        fold_metrics.append(m)
        df["fold"] = fold
        all_preds.append(df)
        print(f"[fold {fold}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))
        if run_wandb:
            wandb.log({f"fold{fold}/{k}": v for k, v in m.items() if isinstance(v, float)})

    pd.concat(all_preds, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    summary = _summarise(fold_metrics)
    summary["run"] = {  # self-describing so results collect without re-parsing the config
        "data": cfg.data.name, "model": cfg.model.name, "loss": cfg.loss.name,
        "protocol": cfg.protocol.name, "crop": cfg.crop or "all", "seed": cfg.seed,
        "members": members,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY (mean +/- std across folds) ===")
    for k, v in summary.items():
        if k.endswith("_mean"):
            base = k[:-5]
            print(f"  {base:24s} {v:.4f} +/- {summary.get(base+'_std', 0):.4f}")
    if run_wandb:
        wandb.log({f"summary/{k}": v for k, v in summary.items()})
        wandb.finish()
    return summary


def _summarise(fold_metrics: list[dict]) -> dict:
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], float)]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[f"{k}_mean"] = float(vals.mean())
            out[f"{k}_std"] = float(vals.std())
    out["n_folds"] = len(fold_metrics)
    return out


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
