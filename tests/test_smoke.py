"""End-to-end smoke test: the clone-and-run guarantee.

Builds a tiny synthetic cache in a temp dir and runs one training fold for each
loss, asserting the pipeline completes and writes finite metrics + predictions.
This is what CI runs, so a green badge means ``clone -> install -> run`` works.
"""
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from axle.data.synthetic import write_synthetic
from axle.train import run


@pytest.fixture(scope="module")
def synth_cache(tmp_path_factory):
    out = tmp_path_factory.mktemp("cache") / "Synthetic"
    return write_synthetic(str(out), fields=24, pixels_per_field=64, seed=0, swath=3.0)


def _cfg(cache, loss, out_dir):
    return OmegaConf.create({
        "seed": 0, "crop": None, "device": "cpu", "num_workers": 0,
        "data": {"name": "synthetic", "cache_dir": cache, "nan_fill": 0.0},
        "model": {"name": "lstm", "hidden": 32, "layers": 1, "dropout": 0.0},
        "loss": {"name": loss, **({"learn_grade_scale": True} if loss.startswith("axle") else {})},
        "protocol": {"name": "cv10", "n_splits": 3},
        "patch": {"tile": 8, "min_pixels": 16, "directions": None},
        "train": {"epochs": 2, "batch_size": 512, "patch_batch_size": 4, "lr": 2e-3, "inner_val_frac": 0.15,
                  "weight_decay": 0.0, "grad_clip": 5.0, "log_every": 0},
        "output_dir": str(out_dir), "wandb": {"enabled": False, "project": "x", "entity": None},
    })


@pytest.mark.parametrize("loss", ["mse", "invvar", "hetero", "axle", "axle_spatial"])
def test_pipeline_runs_and_writes(synth_cache, tmp_path, loss):
    out = tmp_path / f"run_{loss}"
    summary = run(_cfg(synth_cache, loss, out))
    assert np.isfinite(summary["pixel_r2_mean"])
    assert np.isfinite(summary["field_r2_mean"])
    assert (out / "metrics.json").exists()
    preds = pd.read_parquet(out / "predictions.parquet")
    assert len(preds) > 0 and {"target", "prediction", "fold"} <= set(preds.columns)
    if loss in ("hetero", "axle", "axle_spatial"):  # variance losses report calibration
        assert np.isfinite(summary["pixel_picp90_mean"])


def test_m2_uses_directions_when_present(synth_cache, tmp_path):
    """With a directions table on disk the patch loader must pick it up automatically."""
    import pandas as pd
    from axle.data.direction import estimate_field_directions
    from axle.data.patches import DIRECTIONS_FILE, load_directions

    meta = pd.read_parquet(f"{synth_cache}/meta.parquet")
    estimate_field_directions(meta).to_parquet(f"{synth_cache}/{DIRECTIONS_FILE}", index=False)
    assert load_directions(synth_cache) is not None
    summary = run(_cfg(synth_cache, "axle_spatial", tmp_path / "run_m2_dirs"))
    assert np.isfinite(summary["pixel_r2_mean"])


def test_batch_larger_than_fold_does_not_crash(synth_cache, tmp_path):
    """Regression: a batch_size exceeding the fold size must not divide by zero."""
    cfg = _cfg(synth_cache, "axle", tmp_path / "big_batch")
    cfg.train.batch_size = 100000            # far larger than any fold
    summary = run(cfg)
    assert np.isfinite(summary["pixel_r2_mean"])
