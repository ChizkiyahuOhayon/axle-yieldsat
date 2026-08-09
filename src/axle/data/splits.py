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


# The inner selection split must mimic the *outer* shift, or epoch selection is made
# in-distribution while the score is reported out-of-distribution -- which reliably picks
# an over-trained model. So each protocol selects on a held-out slice of its own shift key.
SELECTION_KEY = {"cv10": "field_shared_name", "loyo": "year", "loro": "farm"}


def inner_split(meta: pd.DataFrame, idx: np.ndarray, key: str = "field_shared_name",
                frac: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Carve a *shift-matched* selection set out of a fold's training rows.

    Epoch selection has to happen somewhere. Doing it on the held-out fold means
    selecting on the very data the number is reported from -- optimistic, and unequally
    so across losses, since a loss that overfits harder gains more from an oracle stop.
    But holding out a random slice of *fields* is not enough either: that slice shares
    the fold's years and farms, so it is in-distribution, and the epoch that is best
    in-distribution is the over-trained one that does worst under shift.

    So ``key`` is the protocol's own shift variable -- a held-out year for LOYO, a
    held-out farm for LORO, held-out fields for CV10 -- and the selection set is a
    miniature of the evaluation it is standing in for.

    Returns ``(fit_idx, select_idx)``; ``frac <= 0`` returns everything as fit.
    """
    idx = np.asarray(idx)
    if frac <= 0:
        return idx, np.empty(0, dtype=idx.dtype)
    values = meta[key].to_numpy()[idx]
    groups = np.unique(values)
    if len(groups) < 3:      # need at least one to spare and two to fit
        return idx, np.empty(0, dtype=idx.dtype)
    n_sel = int(np.clip(round(frac * len(groups)), 1, len(groups) - 2))
    chosen = set(np.random.default_rng(seed).choice(groups, size=n_sel, replace=False))
    in_sel = np.fromiter((v in chosen for v in values), dtype=bool, count=len(idx))
    return idx[~in_sel], idx[in_sel]


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
