"""A small, transparent training loop for one CV fold.

Deliberately raw PyTorch (no framework) so every step is visible. Handles both
point and variance-predicting heads, collects loss-consistent predictive
variances at validation (for NLL/PICP), and supports **deep ensembles**: train
``members`` models with decorrelated seeds and combine them.

Ensemble uncertainty follows the standard decomposition
``Var = mean_m(aleatoric_m) + Var_m(mu_m)`` -- the mean of members' predicted
variances (aleatoric) plus the spread of their means (epistemic). For point
losses (no predicted variance) only the epistemic term remains, which is exactly
the classic Deep Ensemble uncertainty. Combining ``loss=axle`` with members > 1
gives a *reliability-aware* deep ensemble.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .eval.metrics import all_metrics
from .utils.seed import set_seed

BuildFn = Callable[[], tuple[torch.nn.Module, torch.nn.Module]]


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _forward(model, batch: dict):
    """Run the backbone on either a pixel batch (B, T, C) or a patch batch (B, K, T, C).

    Patches are flattened to pixels for the (per-pixel) backbone and reshaped back, so
    the spatial loss sees (B, K) fields it can build a covariance over. The backbone
    itself stays untouched -- M2 changes the objective, not the predictor.
    """
    x, mask, static = batch["sample"], batch["mask"], batch.get("static")
    if getattr(model, "consumes_patches", False):
        # a spatial backbone wants the tile intact and emits (B, K, D) itself
        return model(x, mask, static, batch.get("pix_mask"))
    if x.dim() != 4:
        return model(x, mask, static)
    b, k = x.shape[:2]
    out = model(x.reshape(b * k, *x.shape[2:]), mask.reshape(b * k, -1),
                None if static is None else static.reshape(b * k, -1))
    if isinstance(out, dict):
        return {key: v.reshape(b, k) for key, v in out.items()}
    return out.reshape(b, k)


def train_fold(
    build_fn: BuildFn,
    train_ds,
    val_ds,
    *,
    select_ds=None,
    members: int = 1,
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    num_workers: int = 4,
    device: str | None = None,
    grad_clip: float = 5.0,
    log_every: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Train one fold (optionally an ensemble of ``members``); return (val df, metrics).

    ``build_fn()`` returns a fresh ``(model, loss_fn)`` each call so members are
    independently initialised.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dfs, diags = [], []
    for m in range(members):
        if members > 1:
            set_seed(seed + m)  # decorrelate members (init + shuffling)
            if log_every:
                print(f"  [member {m + 1}/{members}]")
        model, loss_fn = build_fn()
        df, _ = _train_single(model, loss_fn, train_ds, val_ds, select_ds=select_ds, epochs=epochs,
                              batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                              num_workers=num_workers, device=device, grad_clip=grad_clip,
                              log_every=log_every)
        dfs.append(df)
        diags.append(getattr(loss_fn, "diagnostics", dict)())
    val_df = dfs[0] if members == 1 else _aggregate_members(dfs)
    metrics = all_metrics(val_df)
    metrics.update(_mean_diagnostics(diags))   # learned loss parameters, averaged over members
    return val_df, metrics


def _mean_diagnostics(diags: list[dict]) -> dict:
    """Average the losses' learned parameters (AXLE grade scales, M2 swath geometry).

    These are end-of-training values, not best-epoch ones -- they describe the fitted
    noise model, so they are reported alongside the metrics rather than selected on.
    """
    keys = set().union(*diags) if diags else set()
    return {k: float(np.mean([d[k] for d in diags if k in d])) for k in sorted(keys)}


def _train_single(model, loss_fn, train_ds, val_ds, *, select_ds=None, epochs, batch_size, lr,
                  weight_decay, num_workers, device, grad_clip, log_every):
    """Train one model.

    With ``select_ds`` (a slice held out of the *training* fields) the best epoch is
    chosen on that set and the outer fold is scored once, at the end, with the restored
    weights -- so the reported number never selected on itself. Without it, the legacy
    behaviour applies: the best epoch is picked on the outer fold, which is optimistic
    and kept only for reproducing earlier runs.
    """
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    params = list(model.parameters()) + list(loss_fn.parameters())  # AXLE grade-scale lives in the loss
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    # drop_last only when it still leaves a batch (BatchNorm needs >1 sample; small
    # folds must not end up with zero batches).
    drop_last = len(train_ds) >= 2 * batch_size
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,
                    drop_last=drop_last, collate_fn=getattr(train_ds, "collate_fn", None))
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    honest = select_ds is not None and len(select_ds) > 0
    sl = (DataLoader(select_ds, batch_size=batch_size, shuffle=False, pin_memory=True,
                     num_workers=num_workers) if honest else None)

    best_score, best_state, best_epoch = -np.inf, None, -1
    best_df, best_metrics = None, {}
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in tl:
            batch = _to_device(batch, device)
            opt.zero_grad()
            loss = loss_fn(_forward(model, batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            running += loss.item()

        if honest:  # score the inner selection set; the outer fold stays untouched
            # pixel R2, not field R2: the inner split may hold only a couple of fields,
            # where a field-level R2 is noisy or undefined. The two rank epochs alike.
            score = all_metrics(_validate(model, loss_fn, sl, select_ds, device))["pixel_r2"]
            if np.isfinite(score) and score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
                print(f"    epoch {epoch:3d} | train_loss {running/max(len(tl),1):.4f} | "
                      f"select_pixel_r2 {score:.4f}")
        else:      # legacy: select on the outer fold (optimistic)
            val_df = _validate(model, loss_fn, vl, val_ds, device)
            m = all_metrics(val_df)
            if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
                print(f"    epoch {epoch:3d} | train_loss {running/max(len(tl),1):.4f} | "
                      f"pixel_r2 {m['pixel_r2']:.4f} | field_r2 {m['field_r2']:.4f}")
            if m["field_r2"] > best_score:
                best_score, best_df, best_metrics = m["field_r2"], val_df, m

    if honest:
        if best_state is not None:      # no finite score at any epoch -> keep the last weights
            model.load_state_dict(best_state)
        best_df = _validate(model, loss_fn, vl, val_ds, device)
        best_metrics = all_metrics(best_df)
        best_metrics["selected_epoch"] = float(best_epoch if best_state is not None else epochs - 1)
        if log_every:
            print(f"    -> epoch {best_metrics['selected_epoch']:.0f} selected on the inner split | "
                  f"held-out field_r2 {best_metrics['field_r2']:.4f}")
    return best_df, best_metrics


def _aggregate_members(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine member predictions into an ensemble prediction + decomposed variance."""
    base = dfs[0][["index", "field_shared_name", "n_i", "target"]].reset_index(drop=True)
    order = base["index"].to_numpy()
    preds, vars_ = [], []
    for d in dfs:
        aligned = d.set_index("index").reindex(order)
        preds.append(aligned["prediction"].to_numpy())
        vars_.append(aligned["variance"].to_numpy())
    preds = np.stack(preds)                       # (M, N)
    vars_ = np.stack(vars_)                        # (M, N), NaN for point losses
    epistemic = preds.var(axis=0)                 # spread of member means
    aleatoric = np.nanmean(vars_, axis=0)         # mean of members' aleatoric variance
    base["prediction"] = preds.mean(axis=0)
    base["variance"] = np.where(np.isfinite(aleatoric), aleatoric + epistemic, epistemic)
    return base


@torch.no_grad()
def _validate(model, loss_fn, loader, val_ds, device) -> pd.DataFrame:
    """Score a loader and join predictions back to ``meta`` by each pixel's row index.

    Joining on ``row_idx`` rather than on batch order is what lets a tile loader (whose
    items contain empty cells, dropped here via ``pix_mask``) and a pixel loader produce
    the same table.
    """
    model.eval()
    rows, tgt, pred, var = [], [], [], []
    for batch in loader:
        b = _to_device(batch, device)
        out = _forward(model, b)
        mu = out["mu"] if isinstance(out, dict) else out
        v = loss_fn.predictive_variance(out, b) if hasattr(loss_fn, "predictive_variance") else None
        keep = b.get("pix_mask")
        keep = torch.ones_like(mu, dtype=torch.bool) if keep is None else keep.reshape(mu.shape) > 0.5
        rows.append(b["row_idx"].reshape(-1)[keep.reshape(-1)].cpu().numpy())
        tgt.append(b["target"].reshape(-1)[keep.reshape(-1)].cpu().numpy())
        pred.append(mu.reshape(-1)[keep.reshape(-1)].cpu().numpy())
        var.append((v.reshape(-1)[keep.reshape(-1)].cpu().numpy() if v is not None
                    else np.full(int(keep.sum()), np.nan, np.float32)))

    r = np.concatenate(rows)
    meta = val_ds.meta if not hasattr(val_ds, "pixels") else val_ds.pixels.meta
    meta = meta.iloc[r]
    return pd.DataFrame({
        "index": meta["index"].to_numpy(),
        "field_shared_name": meta["field_shared_name"].to_numpy(),
        "n_i": meta["n_i"].to_numpy() if "n_i" in meta else np.nan,
        "target": np.concatenate(tgt).astype(np.float32),
        "prediction": np.concatenate(pred).astype(np.float32),
        "variance": np.concatenate(var).astype(np.float32),
    })
