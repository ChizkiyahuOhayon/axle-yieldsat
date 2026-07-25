# AXLE — Acquisition-anchored Label noise for robust dense crop-yield regression

[![CI](https://github.com/ChizkiyahuOhayon/axle-yieldsat/actions/workflows/ci.yml/badge.svg)](https://github.com/ChizkiyahuOhayon/axle-yieldsat/actions/workflows/ci.yml)

> **Trust the Instrument: Acquisition-Anchored Label Noise for Robust Dense Yield Regression**
> Reference implementation on the [YieldSAT](https://yieldsat.github.io/) benchmark.

High-resolution crop-yield maps are produced by combine-harvester yield monitors — a
moving physical instrument. The resulting per-pixel label is a *noisy observation*
whose reliability varies by an order of magnitude (a pixel backed by 27 harvester
passes vs. one interpolated from a single point) and whose error is **striped along
the harvester's travel direction**. YieldSAT ships the metadata that records this
reliability (per-pixel support count, within-pixel std, field quality grade) but its
benchmark trains every model with an **equal-weight** loss and is then reported to
lose 22 pp R² under leave-one-year-out and to collapse cross-country.

**AXLE** makes that shipped reliability drive both the objective and the evaluation:

- **M1 — anchored heteroscedastic loss** *(implemented, this release)*: the training
  loss's aleatoric variance is *supplied* by the instrument metadata rather than
  inferred from the model, so it transfers under distribution shift by construction.
- **M2 — swath-correlated loss** *(module + tests; wiring on the roadmap)*: models the
  along-track correlation of harvester noise so the loss *de-stripes* the label field.
- **Reliability-stratified evaluation**: reports the R² gap between the pooled set and
  trustworthy pixels, making "SOTA under shift" a confound-isolated claim.

See [`docs/METHOD.md`](docs/METHOD.md) for the math and [`docs/DATA.md`](docs/DATA.md)
for the dataset layout and the (verified, exact) reliability join.

---

## Install

```bash
git clone https://github.com/ChizkiyahuOhayon/axle-yieldsat.git && cd axle-yieldsat
python -m venv .venv && source .venv/bin/activate     # Python 3.10–3.12
pip install -e .            # add ".[log]" for Weights & Biases, ".[dev]" for tests
```

> **Python 3.10–3.12** is required (a `hydra-core` argparse issue affects 3.13+).
> On a GPU server, install the matching CUDA build of PyTorch first (see
> [pytorch.org](https://pytorch.org)); a conda env is provided in `environment.yml`.

## Quick start — run without the dataset (≈1 min)

Verify the full pipeline (data → model → AXLE loss → eval) end-to-end on a tiny
**synthetic** cache — no 23 GB download needed:

```bash
make demo          # builds a synthetic cache, trains Transformer + AXLE
# or explicitly:
python scripts/make_synthetic_cache.py --out data/cache/Synthetic
python -m axle.train data=synthetic model=transformer loss=axle protocol=cv10 \
    protocol.n_splits=3 train.epochs=8
```

You should see pixel/field R² climb and a calibrated PICP@90 (~0.9). This is the
same code path the real data uses — it is the clone-and-run guarantee (also run in CI).

```bash
make test          # unit tests + the end-to-end smoke test
```

## Data

Point the pipeline at the YieldSAT release (`Both.zip` on the server at
`/home/smbu/dy/nas/yieldsat`). Extract the per-country inner zips you need
(see [`docs/DATA.md`](docs/DATA.md)), then build the training cache **once**:

```bash
# one country, with the AXLE reliability join (verifies the join is target-exact):
python scripts/prepare.py \
    --netcdf /path/Preprocessed/Germany/merge_s2-soil-dem-weather-coords.nc \
    --both   /path/Both.zip --country Germany --out data/cache/Germany

# all four countries at once (layout <root>/<Country>/merge_*.nc):
python scripts/prepare.py --root /path/Preprocessed --both /path/Both.zip --out data/cache
```

This writes `sample.npy` (memmap), `meta.parquet` (targets + reliability), `norm.json`,
`bands.json`. Default input = the 12 Sentinel-2 bands (`--bands` to change).

## Train

One command per (data, model, loss, protocol). Hydra composes the config:

```bash
# AXLE-M1 on Germany, LSTM backbone, leave-one-year-out, single crop:
python -m axle.train data=germany model=lstm loss=axle protocol=loyo crop=wheat

# the full ablation on one protocol:
python -m axle.train -m loss=mse,invvar,hetero,axle data=germany model=lstm protocol=loro
```

- **models**: `lstm`, `tempcnn`, `transformer` — AXLE is backbone-agnostic.
- **losses**: `mse` (equal-weight baseline) · `invvar` (naive inverse-variance) ·
  `hetero` (learned, unanchored) · `axle` (anchored — ours).
- **protocols**: `cv10` (in-distribution) · `loyo` · `loro` (the shift settings).

Each run writes `predictions.parquet` and `metrics.json` (pixel/field RMSE·R²,
calibration NLL·PICP@90, and the reliability-stratified gap), reported as
mean ± std across folds. Add `wandb.enabled=true` to log.

## Reproduce the benchmark tables

```bash
bash scripts/run_all.sh /path/Both.zip /path/Preprocessed   # builds caches + full grid
```

## Repository layout

```
configs/            Hydra configs (data / model / loss / protocol / experiment)
src/axle/
  data/             prepare (NetCDF→cache), dataset, splits, reliability join
  models/           backbones (lstm/tempcnn/transformer) + heads
  losses/           mse · invvar · hetero · axle (M1); spatial (M2, experimental)
  eval/             metrics (accuracy, calibration, reliability-stratified)
  trainer.py        one-fold training loop
  train.py          Hydra entry: run a protocol end-to-end
  data/synthetic.py synthetic cache generator (clone-and-run demo / CI)
scripts/            prepare.py, make_synthetic_cache.py, extract_reliability_table.py, run_all.sh
tests/              unit + config-composition + end-to-end smoke tests
docs/               METHOD.md, DATA.md, ROADMAP.md
.github/workflows/  CI (tests + a real CLI run on Python 3.10–3.12)
```

## Troubleshooting

- **`hydra` argparse error / Python 3.13+**: use Python 3.10–3.12 (`environment.yml` pins 3.11).
- **DataLoader worker crash on macOS**: pass `num_workers=0` (spawn quirk); Linux servers are fine with the default.
- **`pin_memory` warning on Apple MPS**: harmless.
- **First run is slow to build the cache**: `prepare.py` is a one-time step per country; training then reads the memmap directly.

## Tests

```bash
pytest -q          # objectives, AXLE grade-scale, and the M2 mechanism claim
```

## Citation

```bibtex
@misc{liu2026axle,
  title  = {Trust the Instrument: Acquisition-Anchored Label Noise for Robust Dense Yield Regression},
  author = {Liu, Zhao},
  year   = {2026}
}
```
Built on the YieldSAT benchmark (Miranda et al., 2026). MIT licensed.
