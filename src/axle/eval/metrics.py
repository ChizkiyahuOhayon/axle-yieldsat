"""Evaluation metrics: accuracy, calibration, and reliability-stratified reporting.

Beyond RMSE/R2 (pixel and field level), AXLE reports:

* calibration -- Gaussian NLL and PICP@90 (prediction-interval coverage), and
* the reliability-stratified gap -- R2 pooled minus R2 restricted to trustworthy
  pixels (high support count / Good quality). A method that gains by *not* fitting
  unreliable labels shows a positive gap; this makes "SOTA under shift" a
  confound-isolated claim rather than a metric that partly rewards fitting noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_Z90 = 1.6448536269514722  # standard-normal 0.95 quantile (two-sided 90% interval)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def gaussian_nll(y, p, var):
    y, p, var = np.asarray(y, float), np.asarray(p, float), np.asarray(var, float) + 1e-6
    return float(np.mean(0.5 * ((y - p) ** 2 / var + np.log(2 * np.pi * var))))


def picp(y, p, var, z=_Z90):
    """Prediction-interval coverage probability at the z-level (target ~0.90)."""
    y, p, sd = np.asarray(y, float), np.asarray(p, float), np.sqrt(np.asarray(var, float) + 1e-6)
    return float(np.mean(np.abs(y - p) <= z * sd))


def pixel_metrics(df: pd.DataFrame) -> dict:
    """df columns: target, prediction, [variance]. Averages duplicate pixels first."""
    g = df.groupby("index", as_index=False).agg(
        target=("target", "first"), prediction=("prediction", "mean"),
        **({"variance": ("variance", "mean")} if "variance" in df else {})
    )
    out = {"pixel_rmse": rmse(g.target, g.prediction), "pixel_r2": r2(g.target, g.prediction)}
    if "variance" in g:
        out["pixel_nll"] = gaussian_nll(g.target, g.prediction, g.variance)
        out["pixel_picp90"] = picp(g.target, g.prediction, g.variance)
    return out


def field_metrics(df: pd.DataFrame) -> dict:
    g = df.groupby("index", as_index=False).agg(
        target=("target", "first"), prediction=("prediction", "mean"),
        field=("field_shared_name", "first"))
    f = g.groupby("field", as_index=False).agg(target=("target", "mean"), prediction=("prediction", "mean"))
    return {"field_rmse": rmse(f.target, f.prediction), "field_r2": r2(f.target, f.prediction)}


def reliability_stratified(df: pd.DataFrame, n_col: str = "n_i", n_thresh: float = 5.0) -> dict:
    """R2 gap between the pooled set and the trustworthy subset (n_i >= threshold)."""
    if n_col not in df:
        return {}
    g = df.groupby("index", as_index=False).agg(
        target=("target", "first"), prediction=("prediction", "mean"), n_i=(n_col, "first"))
    pooled = r2(g.target, g.prediction)
    trust = g[g.n_i >= n_thresh]
    r2_trust = r2(trust.target, trust.prediction) if len(trust) > 20 else float("nan")
    return {"pixel_r2_pooled": pooled, "pixel_r2_trustworthy": r2_trust,
            "reliability_gap": pooled - r2_trust if np.isfinite(r2_trust) else float("nan")}


def all_metrics(df: pd.DataFrame) -> dict:
    out = {}
    out.update(pixel_metrics(df))
    out.update(field_metrics(df))
    out.update(reliability_stratified(df))
    return out
