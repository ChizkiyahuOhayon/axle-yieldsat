"""Shift protocols: no leakage, and a bounded fold count on countries with many farms."""
import numpy as np
import pandas as pd
import pytest

from axle.data.splits import make_splits


def _meta(n_farms=57, n_years=8, fields_per_farm=3, px=4):
    """A meta table shaped like Argentina: many farms, several years, many fields."""
    rows = []
    for f in range(n_farms):
        for k in range(fields_per_farm):
            year = 2017 + (f + k) % n_years
            for p in range(px):
                rows.append({"field_shared_name": f"ARG_farm{f}_field{k}_{year}",
                             "farm": f"farm{f}", "year": year, "crop": "soybean"})
    return pd.DataFrame(rows)


def _assert_disjoint(meta, splits, key):
    for tr, va in splits:
        assert set(meta[key].iloc[tr]).isdisjoint(meta[key].iloc[va]), f"{key} leaked"
        assert len(va) > 0 and len(tr) > 0


def test_loro_defaults_to_one_fold_per_farm():
    meta = _meta(n_farms=6)
    splits = make_splits(meta, "loro")
    assert len(splits) == 6
    _assert_disjoint(meta, splits, "farm")


def test_loro_caps_folds_by_grouping_farms():
    """57 farms must not become 57 trainings; groups stay farm-disjoint."""
    meta = _meta(n_farms=57)
    splits = make_splits(meta, "loro", n_splits=6)
    assert len(splits) == 6
    _assert_disjoint(meta, splits, "farm")
    # every pixel is validated exactly once across the folds
    seen = np.concatenate([va for _, va in splits])
    assert len(seen) == len(meta) and len(set(seen.tolist())) == len(meta)


def test_capping_above_the_group_count_is_the_strict_protocol():
    meta = _meta(n_farms=6)
    assert len(make_splits(meta, "loro", n_splits=99)) == 6


def test_loyo_one_fold_per_year_and_no_field_leak():
    meta = _meta(n_farms=10)
    splits = make_splits(meta, "loyo")
    assert len(splits) == meta["year"].nunique()
    _assert_disjoint(meta, splits, "year")
    _assert_disjoint(meta, splits, "field_shared_name")


def test_unknown_protocol_is_rejected():
    with pytest.raises(ValueError, match="unknown protocol"):
        make_splits(_meta(n_farms=3), "leave-one-continent-out")
