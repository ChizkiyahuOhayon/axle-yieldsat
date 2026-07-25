"""A small, transparent training loop for one CV fold.

Deliberately raw PyTorch (no framework) so every step is visible. Handles both
point and variance-predicting heads, and collects loss-consistent predictive
variances at validation so calibration metrics (NLL, PICP) are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .eval.metrics import all_metrics


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_fold(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_ds,
    val_ds,
    *,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    num_workers: int = 4,
    device: str | None = None,
    grad_clip: float = 5.0,
    log_every: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Train one fold; return (validation predictions df, best-epoch metrics)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    params = list(model.parameters()) + list(loss_fn.parameters())  # AXLE grade-scale lives in the loss
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    best_r2, best_df, best_metrics = -np.inf, None, {}
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in tl:
            batch = _to_device(batch, device)
            opt.zero_grad()
            loss = loss_fn(model(batch["sample"], batch["mask"]), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            running += loss.item()
        val_df = _validate(model, loss_fn, vl, val_ds, device)
        m = all_metrics(val_df)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d} | train_loss {running/len(tl):.4f} | "
                  f"pixel_r2 {m['pixel_r2']:.4f} | field_r2 {m['field_r2']:.4f}")
        if m["field_r2"] > best_r2:
            best_r2, best_df, best_metrics = m["field_r2"], val_df, m
    return best_df, best_metrics


@torch.no_grad()
def _validate(model, loss_fn, loader, val_ds, device) -> pd.DataFrame:
    model.eval()
    idx, tgt, pred, var = [], [], [], []
    for batch in loader:
        b = _to_device(batch, device)
        out = model(b["sample"], b["mask"])
        mu = out["mu"] if isinstance(out, dict) else out
        v = loss_fn.predictive_variance(out, b) if hasattr(loss_fn, "predictive_variance") else None
        tgt.extend(b["target"].cpu().numpy())
        pred.extend(mu.cpu().numpy())
        var.extend(v.cpu().numpy() if v is not None else [np.nan] * len(mu))
    # positions in val_ds map back to meta rows for field/reliability columns
    meta = val_ds.meta.iloc[val_ds.rows].reset_index(drop=True)
    df = pd.DataFrame({
        "index": meta["index"].to_numpy(),
        "field_shared_name": meta["field_shared_name"].to_numpy(),
        "n_i": meta["n_i"].to_numpy() if "n_i" in meta else np.nan,
        "target": np.asarray(tgt, np.float32),
        "prediction": np.asarray(pred, np.float32),
        "variance": np.asarray(var, np.float32),
    })
    return df
