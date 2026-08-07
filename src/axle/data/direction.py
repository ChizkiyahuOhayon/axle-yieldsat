r"""Harvester travel-direction estimation (AXLE-M2, :math:`d_f`).

A combine harvester drives a field in parallel passes, so its acquisition error is
coherent *along* the travel direction and breaks *across* it: the support-count
raster :math:`n_i` shows stripes. M2's covariance needs that direction, one angle
per field.

We estimate it with a discrete Radon projection. For a candidate stripe direction
:math:`u_\theta=(\cos\theta,\sin\theta)` we project every pixel onto the *across*-track
axis :math:`v_\theta=(-\sin\theta,\cos\theta)`, bin the projections at one-pixel
spacing, and average the signal in each bin. If the signal is constant along
:math:`u_\theta` and jumps between passes, that across-track profile carries a large
share of the field's variance. So

.. math::  d_f = \arg\max_\theta \ \frac{\mathrm{Var}\,[\,\text{profile}_\theta\,]}{\mathrm{Var}\,[\,\text{signal}\,]},

and the maximised ratio doubles as a *strength* score: the fraction of the signal's
variance explained by an across-track step pattern. Fields whose strength falls below
``min_strength`` get no direction (``(0, 0)``), and the loss falls back to an isotropic
kernel there -- an honest "no stripe detected" rather than a fabricated angle.

The estimate runs off the prepared cache's ``meta.parquet`` alone (it carries
``row``/``col``/``n_i``), so it needs neither the raw release nor a re-``prepare``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-8


def estimate_direction(
    row: np.ndarray,
    col: np.ndarray,
    signal: np.ndarray,
    *,
    n_angles: int = 60,
    min_bin_count: int = 3,
) -> dict:
    """Estimate one field's stripe direction from a scattered raster.

    Args:
        row, col: integer pixel coordinates (any origin).
        signal:   per-pixel value whose stripes reveal the travel direction (``n_i``).
        n_angles: angles scanned over [0, 180) -- 60 gives 3-degree resolution.
        min_bin_count: bins thinner than this many pixels are dropped (edge slivers
            of the field would otherwise contribute spurious profile variance).

    Returns:
        dict with ``dir_row``/``dir_col`` (unit vector), ``theta_deg``, ``strength``
        (variance fraction, in [0, 1]) and ``n_px``.
    """
    y = np.asarray(signal, np.float64)
    ok = np.isfinite(y)
    r = np.asarray(row, np.float64)[ok]
    c = np.asarray(col, np.float64)[ok]
    y = y[ok]
    n = len(y)
    total_var = y.var()
    if n < 3 * min_bin_count or total_var < _EPS:
        return _no_direction(n)

    r = r - r.mean()
    c = c - c.mean()
    y = y - y.mean()

    thetas = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    # across-track axis for each candidate stripe direction u = (cos, sin)
    v_row, v_col = -np.sin(thetas), np.cos(thetas)          # (A,)
    proj = np.outer(r, v_row) + np.outer(c, v_col)          # (N, A) across-track coordinate

    best_score, best_theta = -np.inf, 0.0
    for a in range(n_angles):
        t = proj[:, a]
        b = np.rint(t - t.min()).astype(np.int64)           # 1-pixel bins
        cnt = np.bincount(b)
        tot = np.bincount(b, weights=y)
        keep = cnt >= min_bin_count
        if keep.sum() < 2:
            continue
        profile = tot[keep] / cnt[keep]
        # variance of the profile, weighted by how many pixels back each bin
        w = cnt[keep] / cnt[keep].sum()
        mean = (w * profile).sum()
        score = float((w * (profile - mean) ** 2).sum() / (total_var + _EPS))
        if score > best_score:
            best_score, best_theta = score, float(thetas[a])

    if not np.isfinite(best_score):
        return _no_direction(n)
    return {
        "dir_row": float(np.cos(best_theta)),
        "dir_col": float(np.sin(best_theta)),
        "theta_deg": float(np.degrees(best_theta)),
        "strength": float(np.clip(best_score, 0.0, 1.0)),
        "n_px": int(n),
    }


def _no_direction(n: int) -> dict:
    return {"dir_row": 0.0, "dir_col": 0.0, "theta_deg": np.nan, "strength": 0.0, "n_px": int(n)}


def estimate_field_directions(
    meta: pd.DataFrame,
    *,
    signal: str = "n_i",
    min_strength: float = 0.10,
    n_angles: int = 60,
) -> pd.DataFrame:
    """Estimate :math:`d_f` for every field in ``meta``.

    Fields whose stripe pattern is weaker than ``min_strength`` keep ``strength`` but
    get a zeroed direction, which the M2 kernel reads as "isotropic".

    Returns one row per field: ``field_shared_name, dir_row, dir_col, theta_deg,
    strength, n_px, has_direction``.
    """
    if signal not in meta:
        raise KeyError(f"meta has no column {signal!r}; available: {list(meta.columns)}")
    out = []
    for field, g in meta.groupby("field_shared_name", sort=True):
        est = estimate_direction(g["row"].to_numpy(), g["col"].to_numpy(),
                                 g[signal].to_numpy(), n_angles=n_angles)
        est["field_shared_name"] = field
        est["has_direction"] = bool(est["strength"] >= min_strength)
        if not est["has_direction"]:
            est["dir_row"], est["dir_col"] = 0.0, 0.0
        out.append(est)
    cols = ["field_shared_name", "dir_row", "dir_col", "theta_deg", "strength", "n_px", "has_direction"]
    return pd.DataFrame(out)[cols]
