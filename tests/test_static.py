"""Static-band support: the full-band cache layout, and that it reaches the head.

The release repeats soil/DEM/coordinate channels across all 24 time steps, so
``prepare`` stores them once per pixel. These tests cover the reading side (both cache
layouts) and the model side (static context concatenated to the pooled embedding).
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from axle.data.dataset import YieldSATPixels
from axle.data.patches import YieldSATPatches
from axle.data.prepare import split_static_bands
from axle.data.synthetic import write_static, write_synthetic
from axle.models import build_model
from axle.train import run


@pytest.fixture(scope="module")
def static_cache(tmp_path_factory):
    out = tmp_path_factory.mktemp("cache") / "Static"
    write_synthetic(str(out), fields=12, pixels_per_field=64, seed=0, swath=3.0)
    return write_static(str(out), n_static=8, seed=0)


@pytest.fixture(scope="module")
def plain_cache(tmp_path_factory):
    out = tmp_path_factory.mktemp("cache") / "Plain"
    return write_synthetic(str(out), fields=12, pixels_per_field=64, seed=1)


def _fake_ds(x):
    """Minimal stand-in for the xarray Dataset that split_static_bands probes."""
    class FakeDS:
        sizes = {"index": x.shape[0]}

        def __getitem__(self, key):
            return type("A", (), {"isel": lambda s, **kw: type("B", (), {"values": x})()})()

    return FakeDS()


def test_split_static_bands_is_scale_free():
    """Only genuinely time-constant bands are static -- whatever a channel's units are."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 24, 4)).astype(np.float64)
    x[:, :, 1] *= 1e-9                             # tiny but varying -> dynamic
    x[:, :, 2] = rng.normal(size=(50, 1)) * 5000   # constant in time, large units -> static
    x[:, :, 3] = rng.normal(size=(50, 1)) * 1e-9   # constant in time, tiny units -> static

    dyn, static = split_static_bands(_fake_ds(x), ["s2", "small", "soil", "coord"], [0, 1, 2, 3])
    assert dyn == ["s2", "small"]
    assert static == ["soil", "coord"]


def test_a_band_constant_everywhere_counts_as_static():
    x = np.random.default_rng(0).normal(size=(20, 24, 2))
    x[:, :, 1] = 7.0                               # degenerate: no information at all
    dyn, static = split_static_bands(_fake_ds(x), ["s2", "fill"], [0, 1])
    assert static == ["fill"]


def test_plain_cache_has_no_static(plain_cache):
    ds = YieldSATPixels(plain_cache)
    assert ds.num_static == 0 and ds.static is None
    assert "static" not in ds[0]


def test_static_cache_exposes_normalised_context(static_cache):
    ds = YieldSATPixels(static_cache)
    assert ds.num_static == 8
    item = ds[0]
    assert item["static"].shape == (8,)
    assert torch.isfinite(item["static"]).all()
    # scalar and batched readers must agree
    assert torch.allclose(item["static"], ds.gather(np.array([0]))["static"][0])


def test_patches_carry_static(static_cache):
    ds = YieldSATPatches(static_cache, tile=8, min_pixels=8)
    item = ds[0]
    k = item["target"].shape[0]
    assert item["static"].shape == (k, 8)
    batch = YieldSATPatches.collate_padded([ds[0], ds[1]])
    assert batch["static"].shape[0] == 2 and batch["static"].shape[2] == 8


@pytest.mark.parametrize("predict_variance", [False, True])
def test_model_uses_static_and_changes_output(predict_variance):
    torch.manual_seed(0)
    model = build_model("lstm", in_dim=12, predict_variance=predict_variance, static_dim=8)
    x, mask = torch.randn(4, 24, 12), torch.ones(4, 24)
    a = model(x, mask, torch.zeros(4, 8))
    b = model(x, mask, torch.ones(4, 8))
    mu = lambda o: o["mu"] if isinstance(o, dict) else o
    assert not torch.allclose(mu(a), mu(b)), "static context must influence the prediction"


def test_model_without_static_dim_ignores_it():
    model = build_model("lstm", in_dim=12, predict_variance=False, static_dim=0)
    out = model(torch.randn(2, 24, 12), torch.ones(2, 24))
    assert out.shape == (2,)


def test_model_built_with_static_rejects_a_batch_without_it():
    model = build_model("lstm", in_dim=12, predict_variance=False, static_dim=8)
    with pytest.raises(ValueError, match="static"):
        model(torch.randn(2, 24, 12), torch.ones(2, 24))


@pytest.mark.parametrize("loss", ["mse", "axle", "axle_spatial"])
def test_end_to_end_with_static(static_cache, tmp_path, loss):
    cfg = OmegaConf.create({
        "seed": 0, "crop": None, "device": "cpu", "num_workers": 0,
        "data": {"name": "static", "cache_dir": static_cache, "nan_fill": 0.0},
        "model": {"name": "lstm", "hidden": 32, "layers": 1, "dropout": 0.0},
        "loss": {"name": loss, **({"learn_grade_scale": True} if loss.startswith("axle") else {})},
        "protocol": {"name": "cv10", "n_splits": 3},
        "patch": {"tile": 8, "min_pixels": 16, "directions": None},
        "train": {"epochs": 2, "batch_size": 256, "patch_batch_size": 4, "lr": 2e-3, "inner_val_frac": 0.15,
                  "weight_decay": 0.0, "grad_clip": 5.0, "log_every": 0},
        "output_dir": str(tmp_path / f"run_{loss}"),
        "wandb": {"enabled": False, "project": "x", "entity": None},
    })
    summary = run(cfg)
    assert np.isfinite(summary["pixel_r2_mean"])


def test_band_subset_serves_an_ablation_from_one_cache(static_cache):
    """A full-band cache must be able to stand in for a reduced-input run, no re-prepare."""
    import json, pathlib
    manifest = json.loads((pathlib.Path(static_cache) / "bands.json").read_text())
    dyn = manifest["dynamic"]

    full = YieldSATPixels(static_cache)
    sub = YieldSATPixels(static_cache, use_bands=dyn[:4])
    assert full.num_features == len(dyn) and sub.num_features == 4
    # the subset must be the same columns, not a reshuffle
    assert torch.allclose(sub[0]["sample"], full[0]["sample"][:, :4])

    nostatic = YieldSATPixels(static_cache, use_static=False)
    assert nostatic.num_static == 0 and "static" not in nostatic[0]


def test_band_subset_rejects_unknown_names(static_cache):
    with pytest.raises(ValueError, match="no dynamic bands"):
        YieldSATPixels(static_cache, use_bands=["B01", "not_a_band"])
