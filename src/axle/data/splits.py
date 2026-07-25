"""Distribution-shift split protocols for YieldSAT.

All splits group by field so pixels from one field never straddle train/val
(leakage-free), reproducing the benchmark's protocols:

* ``cv10``  -- field-grouped K-fold (in-distribution upper bound),
* ``loyo``  -- leave-one-year-out (temporal shift),
* ``loro``  -- leave-one-region-out, region = farm (spatial shift),
* ``xcountry`` -- train on 3 countries, test on the 4th (the hardest setting).

Each returns a list of ``(train_idx, val_idx)`` integer-position arrays into ``meta``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def _pos(meta: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    return np.nonzero(mask)[0]


def cv10(meta: pd.DataFrame, n_splits: int = 10, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = meta["field_shared_name"].to_numpy()
    idx = np.arange(len(meta))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    gkf = GroupKFold(n_splits=n_splits)
    return [(perm[tr], perm[va]) for tr, va in gkf.split(perm, groups=groups[perm])]


def leave_one_out(meta: pd.DataFrame, key: str) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for val in sorted(meta[key].unique()):
        va = _pos(meta, (meta[key] == val).to_numpy())
        tr = _pos(meta, (meta[key] != val).to_numpy())
        splits.append((tr, va))
    return splits


def loyo(meta: pd.DataFrame, **_) -> list[tuple[np.ndarray, np.ndarray]]:
    return leave_one_out(meta, "year")


def loro(meta: pd.DataFrame, **_) -> list[tuple[np.ndarray, np.ndarray]]:
    return leave_one_out(meta, "farm")


def make_splits(meta: pd.DataFrame, protocol: str, **kw) -> list[tuple[np.ndarray, np.ndarray]]:
    fns = {"cv10": cv10, "loyo": loyo, "loro": loro}
    if protocol not in fns:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {list(fns)}")
    fold_id = 0
    out = []
    for tr, va in fns[protocol](meta, **kw):
        # sanity: no field leakage
        assert set(meta["field_shared_name"].iloc[tr]).isdisjoint(
            meta["field_shared_name"].iloc[va]
        ), f"field leakage in {protocol} fold {fold_id}"
        out.append((tr, va))
        fold_id += 1
    return out
