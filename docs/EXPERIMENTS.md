# Experiment log

A running, append-only record of every training campaign: date, environment,
exact command, full results, and reading. Newest first.

> **Protocol change — 2026-08-09, commit `2744e44`.** Runs 001–004 selected the best
> epoch *on the held-out fold*, i.e. selection on the test set. Every number in those
> runs is oracle-stopped and optimistic. From this point on `train.inner_val_frac=0.15`
> holds out a field-grouped slice of each fold's **training** fields, selects the epoch
> there, and scores the outer fold once with the restored weights. **Runs 001–004 are
> exploratory and must not be quoted in the paper**; the tables have to come from
> re-runs under the new protocol. `inner_val_frac=0` reproduces the old behaviour.
>
> Trigger: the Argentina LOYO curve — train loss falling monotonically (0.778 → 0.383)
> while held-out field R² peaked at epoch 5 (−0.047) and decayed through epoch 15
> (−0.171). Under oracle stopping that fold reports −0.047; the honest number is
> whatever the inner split picks. The bias is not uniform across losses, so it can
> distort the loss comparison itself, not just the absolute level.
>
> **Amendment — 2026-08-09, commit `<shift-matched selection>`.** The first honest
> implementation split the inner selection set by *field*, which shares the fold's years
> and farms and is therefore in-distribution. Selecting an epoch in-distribution and
> reporting out-of-distribution reliably picks the over-trained model — see Run 005,
> where every OOD number collapsed. The selection set now mirrors the protocol's own
> shift (`SELECTION_KEY`: a held-out **year** for LOYO, a held-out **farm** for LORO,
> held-out **fields** for CV10). Run 005 is therefore also superseded.

---

## Run 007 — Full-band Germany + the first Argentina CV10 against the published 0.84

- **Date**: 2026-08-11. Commit `e3427e3`. Transformer, shift-matched selection, 30 epochs.
- Caches (fp16, static/dynamic split) all built: ARG 5.1 GB, BRA 4.1, URU 2.1, DE 1.2 —
  **12.5 GB for all four countries at 120 bands**, as designed.

### A. Argentina soybean CV10 — the direct comparison (12 bands, seed 0)

| loss | field R² | pixel R² | fold σ |
|---|---|---|---|
| `axle` | **0.718** | 0.616 | 0.118 |
| `mse` | 0.700 | 0.607 | 0.098 |
| `axle_spatial` | 0.696 | 0.614 | 0.126 |

**Published YieldSAT: 0.84.** Our baseline reaches **0.700** — respectable, not matching.
The 0.14 gap is attributable to three things we have not yet varied: 12 bands vs 120,
transformer capacity (hidden 64, 2 layers), and a 30-epoch budget. `axle`'s +0.018 is
inside the fold spread at one seed; not a claim yet.

### B. Germany full-band (45 configs, 3 seeds) — bands do not rescue Germany

| loss | cv10 | LOYO | LORO |
|---|---|---|---|
| `mse` | 0.534 ± 0.073 | **−0.536 ± 0.213** | −4.397 ± 0.431 |
| `axle` | **0.546 ± 0.019** | −0.755 ± 0.217 | −3.700 ± 0.395 |
| `hetero` | 0.529 ± 0.042 | −0.689 ± 0.190 | −3.773 ± 0.375 |
| `axle_spatial` | 0.474 ± 0.075 | −0.786 ± 0.319 | −3.431 ± 0.155 |
| `invvar` | 0.421 ± 0.054 | −0.878 ± 0.224 | −3.321 ± 0.914 |

Against the 12-band Run 006, going to 120 bands buys **+0.05 on cv10** and nothing under
shift (LORO is *worse*: −3.35 → −4.40). Germany is 188 wheat fields on 6 farms; no
feature set fixes that. **Germany is the small-country ablation, not a main-table row.**

One thing did move: `axle_spatial` cv10 goes 0.284 → 0.474 with the full band set, so
M2's in-distribution penalty shrinks when the features are strong.

### C. Calibration — the result that does not need a significance test

Deep Ensemble (the benchmark's strongest baseline) versus AXLE, Germany full-band:

| model | pixel NLL | PICP@90 |
|---|---|---|
| `mse` ×5, cv10 | **20.8** | 0.336 |
| `mse` ×5, LOYO | **51.4** | 0.348 |
| `mse` ×5, LORO | **51.3** | 0.277 |
| `axle` ×5, cv10 | **2.22** | **0.842** |
| `axle` ×1, all protocols | 2.4–5.2 | 0.60–0.80 |

**A nominal 90% interval that covers 28%.** The Deep Ensemble's variance estimate fails
under shift by an order of magnitude in NLL, while AXLE — at statistically identical
accuracy — stays usable. This holds across three protocols, two countries, single models
and ensembles, and every selection protocol we have tried. It is the strongest claim in
the project and it is a *calibration* claim, not an accuracy one.

### Next — tune the baseline before comparing losses again

Comparing objectives on a backbone that reaches 0.70 where the literature reaches 0.84
risks measuring a shared bottleneck rather than the objectives. Before any further loss
comparison on Argentina: sweep capacity (hidden 64/128/256) and budget (30/60 epochs) with
`mse` alone on ARG soybean CV10 and find what closes the gap. Then run the full loss grid
at that configuration.

---

## Run 006 — Germany, corrected protocol, 27/27: no shift effect, but calibration survives

- **Date**: 2026-08-10. Commit `1e79558`. Transformer, 12 S2 bands, 3 seeds, shift-matched
  epoch selection (LOYO selects on a held-out year, LORO on a held-out farm, CV10 on
  held-out fields). **This is the first fully valid Germany table.**

### Results — field R² (mean ± across-seed σ)

| loss | cv10 | LOYO | LORO |
|---|---|---|---|
| `mse` | 0.485 ± 0.017 | −0.600 ± 0.140 | −3.347 ± 1.083 |
| `axle` (M1) | **0.510 ± 0.038** | −0.676 ± 0.214 | −3.231 ± 1.416 |
| `axle_spatial` (M2) | 0.284 ± 0.036 | −0.605 ± 0.056 | −3.860 ± 2.380 |

### Reading

1. **No shift effect, in either direction.** LOYO: −0.600 / −0.676 / −0.605 against seed
   σ of 0.06–0.21 — the three losses are indistinguishable. The Run 004 claim ("M1 wins
   year shift") is dead, and so is its mirror image; there is simply no effect to report.
2. **LORO is not a measurement.** Every method lands at −3.2 to −3.9 with seed σ up to
   2.4. Under the corrected protocol the selection set also removes a farm, leaving 4 of
   6 to fit — Germany does not have the farms to support this protocol at all.
3. **M2's in-distribution cost is the one robust effect**: cv10 0.284 vs 0.485/0.510,
   ~5σ. It has now replicated under three different selection protocols.
4. **M1 vs MSE in-distribution is a tie**: 0.510 vs 0.485, gap 0.025 against seed σ
   0.017/0.038 (~0.6σ). Not a win. Report it as parity.

### What survives every protocol change

`pixel_picp90`: `axle` 0.70–0.80, `axle_spatial` 0.75–0.81, `mse` **0.000** by
construction. **At matched accuracy, AXLE supplies calibrated per-pixel uncertainty and
the benchmark's objective supplies none** — from metadata the dataset already ships, with
no architecture change. This has held across the oracle-stopping, field-selection and
shift-matched protocols, and across Germany and Argentina. It is the only claim that has
never moved, and it is a different (weaker but defensible) paper from "we beat SOTA on R²".

### Consequence for the campaign

Germany is 188 wheat fields and 6 farms; under either shift protocol *every* method has
negative field R². It cannot adjudicate a robustness claim and should be reported as the
small-country ablation, not the main table. Argentina (751 fields, 57 farms, 5.3M px,
`mse` LOYO 0.219 / LORO 0.464 even at 12 bands) is the real testbed.

---

## Run 005 — Germany under honest (but in-distribution) early stopping: the story does not survive

- **Date**: 2026-08-09. Commit `691bd3a`. Transformer, 3 seeds, 25/27 configs finished
  (the sweep was still running when collected; `axle_spatial`/LORO has only 1 seed).
- **Change from Run 004**: epoch selected on a field-grouped slice of the *training*
  fields instead of on the held-out fold.

### Results — field R², oracle stop (Run 004) → honest stop (Run 005)

| loss | cv10 | LOYO | LORO |
|---|---|---|---|
| `mse` | 0.611 → **0.485** | 0.003 → **−0.572** | −1.007 → **−1.450** |
| `axle` (M1) | 0.593 → **0.510** | 0.171 → **−0.697** | −0.913 → **−1.427** |
| `axle_spatial` (M2) | 0.475 → **0.284** | 0.077 → **−0.866** | −0.318 → **−2.240**¹ |

¹ one seed only — not comparable.

### Reading — this is a negative result and must be treated as one

1. **The LOYO ordering inverts.** Under oracle stopping `axle` (0.171) beat `mse`
   (0.003); under honest stopping `mse` (−0.572) beats `axle` (−0.697). **Run 004's
   headline — "M1 wins under year shift" — was an artifact of selecting the epoch on the
   evaluation fold.**
2. **The LORO claim goes too.** M2's −0.318 becomes −2.240 (one seed) against MSE's
   −1.450. The "M2 removes the catastrophic-farm failure" finding does not survive.
3. **Only the in-distribution row keeps its shape**: cv10 `axle` 0.510 ≳ `mse` 0.485 >
   `axle_spatial` 0.284. M2's in-distribution cost is the one Run 004 claim that
   replicates.
4. **Everything dropped by 0.1–1.4 R², including the baseline.** A uniform collapse of
   that size is a protocol artifact, not a property of the losses — which is what led to
   diagnosing the *second* flaw below.

### Why the collapse — the selection set was in-distribution

The inner split held out random *fields*, which share the fold's years and farms. So the
epoch was chosen on in-distribution data while the score is reported out-of-distribution.
Under shift those two disagree by construction: in-distribution performance keeps rising
with training while OOD performance peaks early and decays (the Argentina curve shows
exactly this). Selecting on the former systematically returns an over-trained model, and
the deeper a loss can overfit, the more it is punished — so the comparison is distorted
again, in the opposite direction from oracle stopping.

**Fix**: the selection set now mirrors the protocol's shift — a held-out *year* for LOYO,
a held-out *farm* for LORO, held-out *fields* for CV10. Both Run 004 and Run 005 are
superseded; the paper's numbers must come from a re-run under the corrected protocol.

### What survives all three protocols so far

- M2 costs in-distribution accuracy (cv10), consistently.
- The direction signal is real and replicates across countries (Run 002/004) — this is a
  data finding, independent of the training protocol.
- Nothing about M1 or M2 winning under shift is established. Treat the whole
  shift-robustness claim as open.

---

## Run 004 — Germany wheat, 3 seeds: the two modules answer to different shifts

- **Date**: 2026-08-08. Commit `a675f9d`. Server `ubuntu-server`, A40 GPU 1.
- **Setup**: transformer, 30 epochs, batch 4096 px / 64 patches, seeds **0, 1, 2**
  (Run 003's seed 0 plus 18 new configs). Single model (members=1).
- **Collected with**: `collect_results.py outputs/de_m2_* outputs/de_seeds_*`, which now
  also reports the **across-seed** spread, distinct from the fold spread.

### Results — field R², mean over 3 seeds

| loss | cv10 | seed σ | LOYO | seed σ | LORO | seed σ | LORO fold σ |
|---|---|---|---|---|---|---|---|
| `mse` | **0.611** | 0.023 | 0.003 | 0.068 | −1.007 | 0.118 | 2.004 |
| `axle` (M1) | 0.593 | 0.007 | **0.171** | 0.022 | −0.913 | 0.216 | 1.922 |
| `axle_spatial` (M2) | 0.475 | 0.020 | 0.077 | 0.043 | **−0.318** | 0.197 | **0.574** |

### Reading

Seed spread is 0.007–0.22, well below the gaps between losses, so the ordering is now
resolved rather than suggestive:

1. **M1 beats MSE under year shift**: 0.171 vs 0.003, a 0.168 gap against a combined
   seed σ of ≈0.07 (~2.4σ).
2. **M1 beats M2 under year shift**: 0.171 vs 0.077, gap 0.094 against combined σ
   ≈0.048 (~2σ). Run 003 called this "inside fold noise, not resolved" — with three
   seeds it *is* resolved, and it goes against M2. Correction recorded.
3. **M2 beats everything under farm shift**: −0.318 vs −1.007/−0.913, gap ≈0.69 against
   combined σ ≈0.23 (~3σ), and the fold σ collapses from ~2.0 to 0.57.
4. **M2 costs in-distribution accuracy**: cv10 0.475 vs 0.611, gap 0.136 at σ ≈0.03.
   Real, not noise.

**The shape of the contribution changed.** M2 is not "M1 but better" — the two modules
answer to *different shifts*:

| module | mechanism | wins under |
|---|---|---|
| M1 acquisition-anchored variance | how noisy is this pixel's label | **year shift (LOYO)** |
| M2 swath-correlated covariance | how is that noise *arranged* | **farm shift (LORO)** |

Mechanistically consistent: `d_f` is a field/farm-level property, so holding out a farm
means holding out a machine and a driving pattern — modelling the stripe stops the model
transporting the wrong geometry. Holding out a *year* keeps the same farms and much the
same geometry, so M2 has little to earn and pays the correlated-likelihood cost.

This is a sharper claim than "M2 improves on M1", and a harder one to collide with. It
rests on **6 farms**, so it is a hypothesis until Argentina (57 farms) confirms it.

### Argentina cache — verified, ready

`prepare.py --data-root … --country Argentina` + `estimate_directions.py`:

- **5,325,807 px / 751 fields / 57 farms / 2017–2024** — matches the paper's Table 2.
- soybean 3,134,175 px · corn 1,266,445 · wheat 925,187.
- reliability-signal coverage **100.0%** of pixels (Germany: 99%).
- `d_f` resolved for **75.6%** of 751 fields, median strength **0.144** — against
  Germany's 76.3% / 0.131. **The stripe signal replicates on a different country and a
  different data provider**, which is the strongest evidence so far that it is a real
  acquisition artifact and not a Germany-specific quirk.

`loro` on 57 farms would be 57 trainings per config; `protocol.n_splits=8` now holds out
8 disjoint farm *groups* instead (no farm straddles a split), keeping fold counts
comparable with Germany.

### Next

- Argentina soybean LOYO + LORO (the plan's go/no-go, and the test of the
  M1-temporal / M2-spatial split above).
- Germany full-band (`--bands all`) to check whether the 12-band handicap is what keeps
  our baselines below the published numbers.

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
