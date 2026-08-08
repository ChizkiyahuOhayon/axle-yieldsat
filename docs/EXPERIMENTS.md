# Experiment log

A running, append-only record of every training campaign: date, environment,
exact command, full results, and reading. Newest first.

---

## Run 003 — Germany wheat: M2 on real data + the Deep-Ensemble test that was missing

- **Date**: 2026-08-07. Commit `d9bf3c5`. Server `ubuntu-server`, env `axle`,
  A40 GPUs **1/2/3** (GPU 0 belongs to another user).
- **Data**: `data/cache/Germany`, wheat (306,843 px / 188 fields), `d_f` resolved for
  76.3% of 299 fields. Tiling keeps 97.9% of pixels.
- **Config**: transformer unless stated, 30 epochs, batch 4096 px / 64 patches,
  **seed 0 only**. 21 configs, all completed, no errors.

### A. Deep Ensemble (members=5, transformer) — the decisive comparison

| loss ×5 | LOYO | LORO |
|---|---|---|
| `mse` (the paper's strongest baseline) | −0.016 | −0.988 |
| `axle` (M1) | 0.201 | −0.637 |
| **`axle_spatial` (M2)** | **0.212** | **−0.471** |

**AXLE beats the Deep Ensemble on both shift protocols: LOYO +0.23, LORO +0.52.**
Run 001 ran this on the LSTM and AXLE lost; on the backbone where AXLE works, it wins.
Calibration is not close: `mse`×5 gives pixel NLL **20.5 / 24.3** and PICP@90 **0.31**,
versus NLL **2.27–2.40** and PICP **0.84–0.90** for AXLE. The Deep Ensemble's
uncertainty is essentially broken under shift.

### B. Single model (members=1, transformer), field R²

| loss | cv10 | LOYO | LORO (std) |
|---|---|---|---|
| `mse` | **0.591** | −0.040 | −0.947 (1.97) |
| `axle` (M1) | 0.588 | **0.176** | −1.135 (2.48) |
| `axle_spatial` (M2) | 0.456 | 0.127 | **−0.113 (0.27)** |

### C. Backbone gate for M2 (`axle_spatial`, members=1)

| backbone | cv10 | LOYO | LORO | vs Run 001 baselines |
|---|---|---|---|---|
| lstm | −0.057 | −0.136 | −0.252 | worse than `mse` (0.062 LOYO) |
| tempcnn | 0.551 | **0.112** | −0.272 | best on LOYO (`mse` −0.041, `axle` 0.002) |
| transformer | 0.456 | 0.127 | −0.113 | see B |

### Reading

1. **The go/no-go-style comparison is positive** (A). This is the first time
   AXLE-vs-DE exists on a capable backbone, and AXLE wins on accuracy *and*
   calibration under both shifts.
2. **The synthetic result did not transfer.** On planted swaths M2 beat M1 by +0.25
   pixel R²; on real Germany LOYO M2 is **0.05 *below*** M1. The caution logged in
   Run 002 was right: do not move the paper's centre of gravity to M2. Note the gap is
   far inside the fold std (0.33/0.36) at **one seed** — it is not evidence *against*
   M2 either, it is simply not resolved.
3. **M2's real effect here is variance, not mean.** LORO fold std drops from 2.48 (M1)
   / 1.97 (MSE) to **0.27**, a ~10× reduction, turning a −1.1 collapse into −0.11.
   Run 001 called LORO unreadable because one bad held-out farm dominated; M2 removes
   exactly that failure mode. Mechanistically coherent (a held-out farm harvests in a
   different direction, and modelling the stripe stops the model chasing it) — and it
   is a *new* claim, not the one we set out to test, so it needs its own confirmation
   on a country with more farms.
4. **M2 costs in-distribution accuracy** (cv10 0.456 vs 0.588). M1 was neutral there.
   Report it; a correlated likelihood with no shift to correct is a weaker estimator.
5. **The backbone gate holds for M2** (C): harmful on the LSTM, best-in-class on
   TempCNN under LOYO. Same conditional claim as M1, now with 3 backbones × 5 losses.
6. **`rho` 0.70 → 0.77 (single) / 0.83 (×5).** The correlation term earns something on
   real data, but less than on synthetic (0.85) — real harvester noise carries less
   kernel-capturable coherence than we planted.
7. **The learned grade scale contradicts the harvester's own labels**:
   `g(Good)` **1.86** > `g(Bad)` **1.00** for M1 (and 0.97 > 0.84 for M2). The model
   wants *more* acquisition variance on fields the metadata calls Good. Read: once
   `n_i` and `s_i` are in hand, `yieldmap_quality` adds no usable information and the
   fitted correction runs the other way. Worth a paragraph — it is a falsification of
   a component we assumed helpful.

### Caveats

- **One seed.** Every number above is seed 0. The M1-vs-M2 LOYO gap and the whole
  ordering within AXLE variants are inside fold noise. Seeds 1–2 must run before any
  of B/C is quotable.
- Germany, wheat, 6 farms. LORO std is still 1.2–2.1 in the ensemble runs.
- cv10 has 10 folds, LOYO 7, LORO 6 — fold counts differ, so std is not comparable
  across protocols.

### Next

1. **Seeds 1–2** on the headline configs (cheap, removes the biggest caveat).
2. **Argentina soybean** — the plan's real go/no-go, and the only way to test the new
   LORO-variance claim with more than 6 farms.

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

### Mechanism test — planted swath, does the correlation term buy anything?

Synthetic cache with pass-coherent label noise (`swath=4.0`: `n_i` constant within a
pass, one shared error draw per pass), LSTM-32, CV10 (3 folds), 40 epochs, lr 3e-3,
mean over seeds 0/1/2. Pixel-level, because the synthetic's field-level structure is
too thin for a stable field R².

| loss | pixel R² (mean) | per-seed | RMSE | NLL | PICP@90 |
|---|---|---|---|---|---|
| `mse` | 0.502 | 0.57 / 0.58 / 0.37 | 0.592 | — | — |
| `axle` (M1) | 0.546 | 0.66 / 0.38 / 0.60 | 0.564 | 0.671 | 0.900 |
| **`axle_spatial` (M2)** | **0.793** | 0.80 / 0.81 / 0.77 | **0.399** | **0.304** | 0.822 |

Fitted noise model (M2): `rho` **0.849**, `ell_along` **8.87 px**, `ell_across`
**3.14 px**.

### Are the kernel parameters identifiable? (init control)

`ell_across` ending at 3.14 px next to a planted swath width of 4.0 px looks like
recovery — but its *initial* value was 3.0. Re-running seed 0 from three
initialisations settles it:

| `ell_across_init` | fitted | pixel R² |
|---|---|---|
| 1.0 | 1.41 | 0.804 |
| 3.0 | 3.15 | 0.801 |
| 12.0 | 10.99 | 0.794 |

Each run drifts *toward* 4.0 (1.0 rises, 12.0 falls) and none arrives. The likelihood
is close to flat in the length scales, so **`ell_across` is not identifiable here and
must be reported as a hyper-parameter, not a measurement** — no claim that AXLE
"recovers the swath width". `ell_along` likewise sat at 8.8–9.2 from an init of 8.0.

Two things survive this:

- **`rho` is well identified**: 0.873 / 0.852 / 0.833 across the same 12x sweep. The
  share of acquisition variance placed in the correlated block is a stable, high
  number no matter where the kernel starts.
- **The gain is insensitive to the length scale**: pixel R² 0.804 / 0.801 / 0.794 over
  a 12x range. M2 needs no length-scale tuning, which is worth stating positively —
  but it is the same fact as the non-identifiability, not a separate result.

### Reading

1. **The correlation term is what pays.** M1 over MSE is +0.04 and inside the seed
   spread; M2 over M1 is **+0.25 with no seed overlap** (M2's worst seed 0.77 beats
   M1's best 0.66). NLL more than halves. When the noise really is pass-coherent,
   modelling *only* its magnitude (M1) leaves most of the damage on the table.
2. **`rho` = 0.85, not 0.** The model is choosing to put 85% of the acquisition
   variance in the correlated block. Had the off-diagonal been useless, `rho -> 0`
   would have collapsed M2 back to M1 — the ablation is built into the parameter.
3. **M2 slightly under-covers** (PICP 0.822 vs M1's 0.900). Expected direction: the
   marginal variance is unchanged while the mean fits better, so intervals are now a
   touch narrow. Worth watching on real data, not alarming.
4. **This is a verification, not evidence about real harvesters.** The synthetic is
   generated by exactly the process M2 assumes, so the honest claim is "the estimator
   recovers structure it is given", i.e. the implementation is correct and the
   objective is identifiable. Whether real YieldSAT noise has this structure is what
   Germany/Argentina must answer.

### Next

- **Germany, transformer, `loss=mse,axle,axle_spatial` x cv10/loyo/loro** — the first
  real-data read on M2. Bar to clear: M1's Run 001 LOYO field R² of 0.176.
- Then Argentina soybean (the plan's go/no-go), then Brazil + Uruguay.

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
| lstm | loyo | **0.062** | −0.023 | −0.142 | −0.126 |
| lstm | loro | −0.329 | −0.408 | **−0.052** | −0.036 |
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
3. **Backbone-dependent, and on the LSTM it is a loss.** On TempCNN, LOYO axle (0.002)
   only ties the best baseline; on the LSTM, axle LOYO (−0.126) is clearly *worse* than
   MSE (0.062). The gain concentrates where the backbone has the capacity to otherwise
   overfit the label noise — consistent with the coherence gate (`docs/METHOD.md`), but
   it means the honest claim is conditional, not universal: **AXLE helps high-capacity
   backbones under year-shift and hurts the weak one.**
4. **LORO is inconclusive here.** All losses collapse to strongly negative R² with huge
   fold variance (std up to ~3.5 over 6 farms) — Germany has only 6 farms, so a single
   bad held-out farm dominates. Needs per-fold inspection; do not read a loss ordering
   from it yet.
5. **Deep Ensemble adds calibration to point losses** (mse members=5 PICP@90 ≈ 0.17–0.29
   vs 0.00 single-model), confirming the ensemble uncertainty path works.

### Deep Ensemble (members=5, LSTM) — recovered 2026-08-07

The members-aware collector finally separates these rows. Field R²:

| protocol | `mse` x5 (the paper's strongest baseline) | `axle` x5 (reliability-aware DE) |
|---|---|---|
| loyo | **0.049** | 0.021 |
| loro | **−1.265** | −1.816 |

**AXLE loses to the plain Deep Ensemble on the LSTM, on both shift protocols.** Report
it as such. Two things keep this from being fatal:

- Ensembling helps AXLE far more than it helps MSE (LOYO: −0.126 → 0.021, i.e. +0.15,
  versus 0.062 → 0.049 for MSE), so the two are converging, not diverging.
- It is the **LSTM**, the backbone where single-model AXLE also loses (LOYO −0.126 vs
  MSE 0.062). The coherence gate predicts exactly this. The DE test was never run on
  the transformer — where single-model AXLE *wins* by +0.22 — so **the decisive
  DE comparison does not exist yet**. `configs/experiment/deep_ensemble.yaml` now
  defaults to the transformer for that reason.

### Caveats / to redo

- The first `collect_results.py` merged members=1 and members=5 (both LSTM) into the same
  row, so **LSTM axle/mse under loyo/loro are ambiguous** above, and the Deep-Ensemble
  comparison (axle-DE vs mse-DE — the "stand on DE's shoulders" test) could not be read.
  Fixed in commit adding a `members` key. **Re-run** to get the clean split:
  ```bash
  git pull && python scripts/collect_results.py outputs/ --metric field_r2
  ```
  Scan **`outputs/`**, not `multirun/`: `metrics.json` is written to `cfg.output_dir`,
  while `hydra.sweep.dir` only holds hydra's own logs. (This log previously said
  `multirun/`, which finds nothing — corrected 2026-08-07, along with making
  `output_dir` self-describing so a sweep's results sit together and cannot collide.)

### Next

- Re-collect with the members fix; record the DE-vs-AXLE-DE numbers here.
- Add Germany **rapeseed**; then **Argentina soybean** LOYO — the plan's go/no-go.
- LORO: inspect per-fold to see which farm collapses and whether AXLE reduces the worst-fold damage.
