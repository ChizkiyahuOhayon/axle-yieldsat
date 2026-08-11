"""Validate that every Hydra config group composes (catches config typos).

Uses the compose API rather than the CLI so it runs on any Python (the CLI's
argparse path is exercised separately in CI)."""
import itertools

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pathlib import Path

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
DATA = ["germany", "argentina", "brazil", "uruguay", "synthetic",
        "germany_full", "argentina_full", "brazil_full", "uruguay_full"]
MODELS = ["lstm", "tempcnn", "transformer"]
LOSSES = ["mse", "invvar", "hetero", "axle", "axle_spatial"]
PROTOCOLS = ["cv10", "loyo", "loro"]


def test_default_config_composes():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")
        assert cfg.data.name and cfg.model.name and cfg.loss.name and cfg.protocol.name


@pytest.mark.parametrize("data,model,loss,protocol", [
    # one representative full sweep row per group value (not the full cross-product)
    *[(d, "lstm", "axle", "loyo") for d in DATA],
    *[("synthetic", m, "axle", "cv10") for m in MODELS],
    *[("synthetic", "lstm", ls, "cv10") for ls in LOSSES],
    *[("synthetic", "lstm", "axle", p) for p in PROTOCOLS],
])
def test_group_overrides_compose(data, model, loss, protocol):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config",
                      overrides=[f"data={data}", f"model={model}", f"loss={loss}", f"protocol={protocol}"])
        assert cfg.data.name == data and cfg.protocol.name == protocol


@pytest.mark.parametrize("exp", ["axle_ablation", "germany_loyo"])
def test_experiment_configs_compose(exp):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"+experiment={exp}"], return_hydra_config=True)
        assert cfg.data.name and cfg.loss.name
