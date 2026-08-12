"""The 3D spatial-temporal backbone and its dense-tile loader.

The benchmark's strongest models are 3D-CNN based (3D-LSTM 0.77, 3D-ConvLSTM 0.79-0.82
on ARG-S CV10 against Transformer 0.73), so matching that family is what a competitive
main table needs. These tests pin the contract that lets it drop in without changing any
loss, head or metric: dense tiles in, one embedding per pixel out, empty cells inert.
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from axle.data.patches import YieldSATTiles
from axle.data.synthetic import write_synthetic
from axle.models import build_model
from axle.models.backbones import Spatial3D
from axle.train import run

TILE = 8


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    out = tmp_path_factory.mktemp("cache") / "Tiles"
    return write_synthetic(str(out), fields=12, pixels_per_field=100, seed=0, swath=3.0)


def test_tiles_are_dense_and_fixed_size(cache):
    ds = YieldSATTiles(cache, tile=TILE, min_pixels=8)
    item = ds[0]
    k = TILE * TILE
    assert item["sample"].shape == (k, ds.seq_len, ds.num_features)
    assert item["pix_mask"].shape == (k,)
    assert 0 < item["pix_mask"].sum() <= k, "a tile holds some, not necessarily all, pixels"
    # empty cells are inert: zero inputs, zero target, "missing" grade
    empty = item["pix_mask"] == 0
    if empty.any():
        assert (item["sample"][empty] == 0).all()
        assert (item["target"][empty] == 0).all()
        assert (item["quality_idx"][empty] == 3).all()


def test_tile_cells_carry_the_right_pixel(cache):
    """Row-major placement must match the pixel's own (row, col), or the raster is scrambled."""
    ds = YieldSATTiles(cache, tile=TILE, min_pixels=8)
    item = ds[0]
    real = item["pix_mask"] > 0
    coords = item["coords"][real]
    r0, c0 = coords[:, 0].min(), coords[:, 1].min()
    cell = ((coords[:, 0] - r0) * TILE + (coords[:, 1] - c0)).long()
    assert torch.equal(cell, torch.nonzero(real).squeeze(1))
    # and the targets at those cells are the real pixels' targets
    rows = item["row_idx"][real].numpy()
    assert np.allclose(item["target"][real].numpy(),
                       ds.pixels.meta["target"].to_numpy()[rows], atol=1e-5)


def test_collate_is_a_plain_stack(cache):
    ds = YieldSATTiles(cache, tile=TILE, min_pixels=8)
    b = YieldSATTiles.collate_dense([ds[0], ds[1], ds[2]])
    assert b["sample"].shape == (3, TILE * TILE, ds.seq_len, ds.num_features)
    assert b["pix_mask"].shape == (3, TILE * TILE)


def test_backbone_emits_one_embedding_per_pixel():
    m = Spatial3D(in_dim=12, tile=TILE, hidden=32, layers=2)
    b, k = 4, TILE * TILE
    out = m(torch.randn(b, k, 24, 12), torch.ones(b, k, 24), torch.ones(b, k))
    assert out.shape == (b, k, 32)


def test_backbone_rejects_a_non_square_batch():
    m = Spatial3D(in_dim=12, tile=TILE)
    with pytest.raises(ValueError, match="tile\\^2"):
        m(torch.randn(2, 13, 24, 12), torch.ones(2, 13, 24))


def test_empty_cells_do_not_leak_into_their_neighbours():
    """Masked-out cells must be zeroed before convolution, else garbage bleeds spatially."""
    torch.manual_seed(0)
    m = Spatial3D(in_dim=4, tile=TILE, hidden=8, layers=1).eval()
    k = TILE * TILE
    x, mask = torch.randn(1, k, 24, 4), torch.ones(1, k, 24)
    pix = torch.ones(1, k)
    pix[0, k // 2:] = 0

    a = m(x, mask, pix)
    x2 = x.clone()
    x2[0, k // 2:] = 999.0                      # scribble over the *masked* cells only
    b = m(x2, mask, pix)
    assert torch.allclose(a, b, atol=1e-5), "masked cells changed the valid cells' features"


def test_spatial_neighbours_actually_matter():
    """If a pixel's embedding ignored its neighbours this would be a per-pixel model."""
    torch.manual_seed(0)
    m = Spatial3D(in_dim=4, tile=TILE, hidden=8, layers=1).eval()
    k = TILE * TILE
    x, mask, pix = torch.randn(1, k, 24, 4), torch.ones(1, k, 24), torch.ones(1, k)
    a = m(x, mask, pix)
    x2 = x.clone()
    x2[0, 0] += 5.0                              # perturb one pixel
    b = m(x2, mask, pix)
    assert not torch.allclose(a[0, 1], b[0, 1], atol=1e-6), "neighbour had no influence"


@pytest.mark.parametrize("loss", ["mse", "axle", "axle_spatial"])
def test_end_to_end_spatial3d(cache, tmp_path, loss):
    """Every loss must run on the 3D backbone, including M2's correlated NLL."""
    cfg = OmegaConf.create({
        "seed": 0, "crop": None, "device": "cpu", "num_workers": 0,
        "data": {"name": "tiles", "cache_dir": cache, "nan_fill": 0.0},
        "model": {"name": "spatial3d", "tile": TILE, "hidden": 16, "layers": 1,
                  "kernel": 3, "dropout": 0.0},
        "loss": {"name": loss, **({"learn_grade_scale": True} if loss.startswith("axle") else {})},
        "protocol": {"name": "cv10", "n_splits": 3},
        "patch": {"tile": TILE, "min_pixels": 8, "directions": None},
        "train": {"epochs": 2, "batch_size": 256, "patch_batch_size": 4, "lr": 2e-3,
                  "inner_val_frac": 0.15, "weight_decay": 0.0, "grad_clip": 5.0, "log_every": 0},
        "output_dir": str(tmp_path / f"run_{loss}"),
        "wandb": {"enabled": False, "project": "x", "entity": None},
    })
    summary = run(cfg)
    assert np.isfinite(summary["pixel_r2_mean"])
    assert np.isfinite(summary["field_r2_mean"])
