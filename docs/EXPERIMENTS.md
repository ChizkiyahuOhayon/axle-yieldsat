# Experiment log

A running, append-only record of every training campaign: date, environment,
exact command, full results, and reading. Newest first.

---

## Run 002 — AXLE-M2 wiring: is the swath geometry actually there?

- **Date**: 2026-08-07
- **Env**: laptop (macOS, CPU/MPS, Python 3.14, torch 2.13) — a correctness and
  mechanism check, not a benchmark run. Commit `50a061e`.
- **Data**: `data/cache/Germany` (609,645 px, 299 fields) + a synthetic cache with a
  *planted* swath (`write_synthetic(..., swath=4.0)`).

### Does the support-count raster carry a harvester direction?

```bash
python scripts/estimate_directions.py data/cache/Germany
```

Calibrated against a null that permutes `n_i` *within* each field — same geometry,
same marginal, no stripes:

| | median strength | p90 | fields ≥ 0.10 |
|---|---|---|---|
| real `n_i` | **0.131** | 0.197 | 76.3% |
| shuffled `n_i` (null) | 0.062 | 0.101 | 10.7% |

- Real beats its own null in **96.0% of 299 fields**; median ratio **1.97×**.
- Estimated angles spread fairly evenly over [0°, 180°) (26–44 fields per 30° bin),
  i.e. this is per-field geometry, not one global artifact.
- On the planted synthetic swath: **100%** of fields resolved, median strength 0.92,
  angle error ≤ 5°.

**Reading.** The stripe orientation is a real signal but a *modest* one — ~13% of the
`n_i` variance. That is enough to anchor a covariance and not enough to justify
claiming every field has a resolvable direction, so the default threshold is set at
the null's p90 (≈0.10, ~10% false positives) and the remaining 24% of fields stay
isotropic. Worth reporting honestly in the paper: M2 is anchored where the geometry is
visible and degrades to M1-with-isotropic-correlation where it is not.

### Wiring verification (Germany wheat, LOYO, TempCNN, 2 epochs)

Not a result — a plumbing check that the patch path runs on real data: 7 folds,
~1,900 patches/fold, correlated NLL descending (6.46 → 1.48 in one epoch),
PICP@90 ≈ 0.81, metrics and predictions written. 1,382 s on laptop CPU.

### Next

Mechanism test on the planted-swath synthetic (M2 vs M1 vs MSE, 3 seeds) — running;
numbers to be appended here.

---

## Run 001 — Germany wheat, loss ablation + Deep Ensemble

- **Date**: 2026-07-26
- **Env**: server `ubuntu-server`, conda env `axle`, PyTorch CUDA, **NVIDIA A40** (GPU 0 & 1), `torch.cuda.is_available()=True`.
- **Data**: `data/cache/Germany` (609,645 px, 24×12 S2), reliability join 100.00% target-exact, n_i coverage 99.1%.
- **Commit**: `3d45cc5` (deep-ensemble release).
- **Commands**:
  ```bash
  # GPU 0 — single-model ablation (members=1), 36 configs:
  bash scripts/run_gpu.sh 0 germany_ablation -m \
    loss=mse,invvar,hetero,axle model=lstm,tempcnn,transformer \
    data=germany crop=wheat protocol=cv10,loyo,loro train.epochs=30 train.batch_size=4096
  # GPU 1 — Deep Ensemble (members=5, LSTM), 4 configs:
  bash scripts/run_gpu.sh 1 germany_de -m +experiment=deep_ensemble   # loss=mse,axle x protocol=loyo,loro
  ```
- **Config**: 30 epochs, batch 4096, Adam lr 1e-3, seed 0, 12 S2 bands, field-grouped folds
  (cv10=10, loyo=7, loro=6). Metric = field-level R² (mean ± std across folds).

### Results — single-model ablation (members=1), field R²

| backbone | protocol | mse | invvar | hetero | **axle** |
|---|---|---|---|---|---|
| lstm | cv10 | 0.424 | 0.306 | 0.186 | 0.177 |
| lstm | loyo | 0.062 | −0.023 | −0.142 | (see note) |
| lstm | loro | −0.329 | −0.408 | −0.052 | (see note) |
| tempcnn | cv10 | 0.592 | 0.515 | 0.582 | 0.565 |
| tempcnn | loyo | −0.041 | 0.005 | −0.086 | **0.002** |
| tempcnn | loro | −0.289 | −0.914 | −0.563 | −0.522 |
| **transformer** | cv10 | 0.591 | 0.509 | 0.614 | 0.588 |
| **transformer** | **loyo** | −0.040 | −0.005 | 0.138 | **0.176** |
| **transformer** | loro | −0.947 | −1.816 | −1.053 | −1.135 |

Calibration (PICP@90, variance-predicting losses): axle/hetero ≈ 0.74–0.94; point
losses (mse/invvar) single-model have no interval (0.00) by construction.

### Reading

1. **In-distribution (cv10): AXLE is neutral, as predicted.** On the strong backbones
   axle ≈ mse ≈ hetero (0.57–0.61); there is no shift to be robust to, so anchoring
   should not help here — and it doesn't hurt. ✓
2. **Headline positive — LOYO on the transformer: axle 0.176 > hetero 0.138 > invvar
   −0.005 ≈ mse −0.040.** Under year-shift on the strongest backbone, AXLE turns an
   equal-weight *collapse* (−0.04) into a clearly positive R² (+0.18), a **+0.22**
   field-R² gain over MSE. This is exactly the mechanism's claim: not fitting the
   swath/quality-driven label noise buys OOD robustness.
3. **Backbone-dependent.** On TempCNN, LOYO axle (0.002) only ties the best baseline;
   on LSTM (weakest readout) the signal is muddier. The gain concentrates where the
   backbone has the capacity to otherwise overfit the label noise — consistent with
   the coherence-gate finding (`docs/METHOD.md`).
4. **LORO is inconclusive here.** All losses collapse to strongly negative R² with huge
   fold variance (std up to ~3.5 over 6 farms) — Germany has only 6 farms, so a single
   bad held-out farm dominates. Needs per-fold inspection; do not read a loss ordering
   from it yet.
5. **Deep Ensemble adds calibration to point losses** (mse members=5 PICP@90 ≈ 0.17–0.29
   vs 0.00 single-model), confirming the ensemble uncertainty path works.

### Caveats / to redo

- The first `collect_results.py` merged members=1 and members=5 (both LSTM) into the same
  row, so **LSTM axle/mse under loyo/loro are ambiguous** above, and the Deep-Ensemble
  comparison (axle-DE vs mse-DE — the "stand on DE's shoulders" test) could not be read.
  Fixed in commit adding a `members` key. **Re-run** to get the clean split:
  ```bash
  git pull && python scripts/collect_results.py multirun/ --metric field_r2
  ```
  (scan `multirun/` only — `outputs/` held stale `make demo` synthetic runs, now auto-skipped).

### Next

- Re-collect with the members fix; record the DE-vs-AXLE-DE numbers here.
- Add Germany **rapeseed**; then **Argentina soybean** LOYO — the plan's go/no-go.
- LORO: inspect per-fold to see which farm collapses and whether AXLE reduces the worst-fold damage.
