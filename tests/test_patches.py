"""Field-patch dataset: patches must be within-field, disjoint, and padded losslessly."""
import numpy as np
import pandas as pd
import pytest
import torch

from axle.data.patches import YieldSATPatches, load_directions
from axle.data.synthetic import write_synthetic


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    out = tmp_path_factory.mktemp("cache") / "Swath"
    write_synthetic(str(out), fields=6, pixels_per_field=196, seed=0, swath=4.0)
    return str(out)


def test_patches_are_within_field_and_disjoint(cache):
    ds = YieldSATPatches(cache, tile=8, min_pixels=8)
    meta = ds.pixels.meta
    seen = []
    for patch, field in zip(ds.patches, ds.fields):
        rows = ds.pixels.rows[patch]
        assert meta["field_shared_name"].iloc[rows].nunique() == 1, "a patch must be one field"
        assert meta["field_shared_name"].iloc[rows].iloc[0] == field
        assert len(patch) <= 8 * 8
        seen.extend(rows.tolist())
    assert len(seen) == len(set(seen)), "patches must not share pixels"


def test_item_shapes_and_coords(cache):
    ds = YieldSATPatches(cache, tile=8, min_pixels=8)
    item = ds[0]
    k = item["target"].shape[0]
    assert item["sample"].shape == (k, ds.seq_len, ds.num_features)
    assert item["mask"].shape == (k, ds.seq_len)
    assert item["coords"].shape == (k, 2)
    assert item["direction"].shape == (2,)
    assert torch.isfinite(item["sample"]).all()


def test_collate_pads_and_marks(cache):
    ds = YieldSATPatches(cache, tile=8, min_pixels=8)
    items = [ds[i] for i in range(4)]
    sizes = [it["target"].shape[0] for it in items]
    b = YieldSATPatches.collate(items)
    k = max(sizes)
    assert b["sample"].shape == (4, k, ds.seq_len, ds.num_features)
    assert b["pix_mask"].shape == (4, k)
    assert b["pix_mask"].sum(1).tolist() == [float(s) for s in sizes]
    for i, s in enumerate(sizes):
        assert torch.equal(b["target"][i, :s], items[i]["target"])
        assert (b["target"][i, s:] == 0).all()
        assert (b["quality_idx"][i, s:] == 3).all()  # padded -> "missing" bucket


def test_directions_are_attached_per_field(cache, tmp_path):
    ds0 = YieldSATPatches(cache, tile=8, min_pixels=8)
    fields = sorted(set(ds0.fields))
    table = pd.DataFrame({
        "field_shared_name": fields,
        "dir_row": np.linspace(0.1, 1.0, len(fields)),
        "dir_col": 1.0,
        "theta_deg": 45.0, "strength": 0.9, "n_px": 100, "has_direction": True,
    })
    ds = YieldSATPatches(cache, tile=8, min_pixels=8, directions=table)
    lookup = table.set_index("field_shared_name")
    for i, field in enumerate(ds.fields):
        assert ds[i]["direction"].tolist() == pytest.approx(
            [lookup.loc[field, "dir_row"], lookup.loc[field, "dir_col"]], abs=1e-6)


def test_missing_directions_file_is_isotropic_not_an_error(cache):
    assert load_directions(cache) is None
    ds = YieldSATPatches(cache, tile=8, min_pixels=8, directions=None)
    assert (ds[0]["direction"] == 0).all()


def test_min_pixels_too_large_fails_loudly(cache):
    with pytest.raises(ValueError, match="min_pixels"):
        YieldSATPatches(cache, tile=4, min_pixels=10_000)
