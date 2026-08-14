"""Result collection: what counts as a repeat, and what supersedes.

The protocol changed under some configs mid-project (docs/EXPERIMENTS.md), so the same
(config, seed) exists on disk more than once. Averaging those would blend protocol
generations into the main table; averaging across *seeds* is exactly what we want. These
tests pin that asymmetry.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_results.py"


@pytest.fixture(scope="module")
def collect():
    spec = importlib.util.spec_from_file_location("collect_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(root: Path, name: str, *, seed: int, r2: float, mtime: float, **run):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "metrics.json"
    p.write_text(json.dumps({
        "field_r2_mean": r2, "field_r2_std": 0.1, "n_folds": 10,
        "run": {"data": "argentina_full", "model": "transformer", "loss": "axle",
                "protocol": "loro", "crop": "soybean", "members": 1, "seed": seed, **run},
    }))
    os.utime(p, (mtime, mtime))
    return p


def test_a_rerun_of_the_same_seed_supersedes_rather_than_averages(collect, tmp_path):
    write(tmp_path, "old", seed=0, r2=0.10, mtime=1_000)   # pre-fix protocol
    write(tmp_path, "new", seed=0, r2=0.55, mtime=2_000)   # post-fix, same config+seed
    df, _ = collect.load([str(tmp_path)])
    kept, superseded = collect.keep_newest(df)
    assert superseded == 1
    assert kept["field_r2_mean"].tolist() == [0.55], "the older generation leaked in"


def test_different_seeds_are_kept_and_averaged(collect, tmp_path):
    for seed, r2 in [(0, 0.50), (1, 0.60)]:
        write(tmp_path, f"s{seed}", seed=seed, r2=r2, mtime=2_000 + seed)
    df, _ = collect.load([str(tmp_path)])
    kept, superseded = collect.keep_newest(df)
    assert superseded == 0 and len(kept) == 2


def test_members_never_merge(collect, tmp_path):
    """A deep ensemble is a different row, not a repeat of the single model."""
    write(tmp_path, "single", seed=0, r2=0.50, mtime=2_000, members=1)
    write(tmp_path, "ens", seed=0, r2=0.60, mtime=2_001, members=5)
    df, _ = collect.load([str(tmp_path)])
    kept, superseded = collect.keep_newest(df)
    assert superseded == 0 and sorted(kept["members"]) == [1, 5]


def test_input_arms_of_one_cache_stay_separate(collect, tmp_path):
    """`argentina_full` serves three band ablation arms; merging them erases the ablation."""
    write(tmp_path, "full", seed=0, r2=0.743, mtime=2_000, in_dim=16, static_dim=104)
    write(tmp_path, "nostatic", seed=0, r2=0.760, mtime=2_001, in_dim=16, static_dim=0)
    df, _ = collect.load([str(tmp_path)])
    kept, superseded = collect.keep_newest(df)
    assert superseded == 0
    assert sorted(kept["input"]) == ["16d", "16d+104s"]


def test_input_arm_falls_back_to_the_saved_config(collect, tmp_path):
    """Runs predating the in_dim field are still placed, via the config.yaml beside them."""
    yaml = pytest.importorskip("yaml")
    p = write(tmp_path, "legacy", seed=0, r2=0.700, mtime=2_000)
    (p.parent / "config.yaml").write_text(yaml.safe_dump(
        {"data": {"use_bands": [f"B{i:02d}" for i in range(12)], "use_static": False}}))
    df, _ = collect.load([str(tmp_path)])
    assert df["input"].tolist() == ["12d"]


def test_input_arm_is_unknown_rather_than_wrong_without_a_config(collect, tmp_path):
    write(tmp_path, "orphan", seed=0, r2=0.5, mtime=2_000)
    df, _ = collect.load([str(tmp_path)])
    assert df["input"].tolist() == ["?"]


def test_runs_without_a_run_block_are_dropped(collect, tmp_path):
    (tmp_path / "stale").mkdir()
    (tmp_path / "stale" / "metrics.json").write_text(json.dumps({"field_r2_mean": 0.9}))
    write(tmp_path, "good", seed=0, r2=0.50, mtime=2_000)
    df, dropped = collect.load([str(tmp_path)])
    assert dropped == 1 and len(df) == 1
