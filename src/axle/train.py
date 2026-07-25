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
from .data.splits import make_splits
from .losses import build_loss
from .models import build_model
from .trainer import train_fold
from .utils.seed import set_seed


def run(cfg: DictConfig) -> dict:
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")

    full = YieldSATPixels(cfg.data.cache_dir, nan_fill=cfg.data.nan_fill)
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

    fold_metrics, all_preds = [], []
    for fold, (tr_local, va_local) in enumerate(splits):
        # map local (sub_meta) positions back to global dataset rows
        tr = sub_pos[tr_local]
        va = sub_pos[va_local]
        train_ds = YieldSATPixels(cfg.data.cache_dir, indices=tr, nan_fill=cfg.data.nan_fill)
        val_ds = YieldSATPixels(cfg.data.cache_dir, indices=va, nan_fill=cfg.data.nan_fill)

        loss_fn = build_loss(cfg.loss.name, **loss_kw)
        model = build_model(cfg.model.name, in_dim=full.num_features,
                            predict_variance=loss_fn.predicts_variance, **model_kw)
        print(f"[fold {fold}] train={len(tr):,} val={len(va):,} "
              f"({cfg.model.name}+{cfg.loss.name}, {cfg.protocol.name})")
        df, m = train_fold(model, loss_fn, train_ds, val_ds,
                           epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
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
