"""Distribution-shift split protocols for YieldSAT.

All splits group by field so pixels from one field never straddle train/val
(leakage-free), reproducing the benchmark's protocols:

* ``cv10``  -- field-grouped K-fold (in-distribution upper bound),
* ``loyo``  -- leave-one-year-out (temporal shift),
* ``loro``  -- leave-one-region-out, region = farm (spatial shift),
* ``xcountry`` -- train on 3 countries, test on the 4th (the hardest setting).

``loyo``/``loro`` take an optional ``n_splits``: with more distinct years/farms than
that, folds hold out disjoint *groups* rather than single values, which bounds the cost
on countries with many farms without ever letting a farm straddle the split.

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


def leave_one_out(meta: pd.DataFrame, key: str,
                  n_splits: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out one value of ``key`` per fold, or ``n_splits`` disjoint groups of values.

    With few distinct values (Germany has 6 farms) one fold per value is the natural
    protocol. With many (Argentina has 57) it is also 57 trainings, ~10x the cost of the
    year-shift protocol for no extra evidence -- and each fold holds out so little data
    that fold estimates get noisy. Capping at ``n_splits`` groups keeps the shift honest
    (no farm ever straddles the split) while making fold counts comparable across
    countries. ``n_splits=None`` or a value >= the number of groups keeps the strict form.
    """
    vals = sorted(meta[key].unique())
    if n_splits is None or n_splits >= len(vals):
        return [(_pos(meta, (meta[key] != v).to_numpy()), _pos(meta, (meta[key] == v).to_numpy()))
                for v in vals]

    groups = meta[key].to_numpy()
    idx = np.arange(len(meta))
    return [(idx[tr], idx[va]) for tr, va in GroupKFold(n_splits=n_splits).split(idx, groups=groups)]


def loyo(meta: pd.DataFrame, n_splits: int | None = None, **_) -> list[tuple[np.ndarray, np.ndarray]]:
    return leave_one_out(meta, "year", n_splits)


def loro(meta: pd.DataFrame, n_splits: int | None = None, **_) -> list[tuple[np.ndarray, np.ndarray]]:
    return leave_one_out(meta, "farm", n_splits)


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
