# AXLE — method

## Setup

Each 10 m pixel `i` has a season-aligned time series `x_i` and an observed yield
`y_i`. The observed label is an **instrument observation** of the latent yield
`f*(x_i)`: a combine-harvester monitor averages `n_i` support points (spread
`s_i`) inside the pixel; headlands/turns are interpolated (`n_i` small or 0) and
flow-lag smears grain mass along the travel direction. So

```
y_i = f*(x_i) + ε_i,      ε has heteroscedastic, spatially-correlated variance.
```

The benchmark minimises equal-weight MSE, i.e. it assumes `ε_i = 0`. AXLE instead
supplies the noise structure from the instrument metadata YieldSAT already ships.

## M1 — acquisition-anchored heteroscedastic loss  *(implemented)*

Per-pixel acquisition-noise variance, anchored to the metadata:

```
σ²_acq,i = ( s_i² / max(n_i, 1) ) · g(q_i)
```

- `s_i² / max(n_i,1)` is the standard error of the mean of the `n_i` points.
- `q_i ∈ {Good, Average, Bad}` is the field quality grade; `g` is a **learnable**
  per-grade scalar (softplus, init 1), the run's only added hyper-parameters.

A variance-predicting head emits `μ(x_i)` and a model variance `σ²_m,i`. The loss
is a Gaussian NLL with **total** variance:

```
v_i = σ²_m,i + σ²_acq,i
L   = mean_i [ ½ (y_i − μ_i)² / v_i + ½ log v_i ]
```

Because `σ²_acq,i` is *supplied*, the model cannot reduce the loss by inflating
its own variance on noisy pixels — it stops fitting swath artifacts. Where the
reliability signal is missing (`~1%` of pixels) the aleatoric term vanishes and
the loss degrades gracefully to standard heteroscedastic regression. Because the
metadata exists on held-out years/regions, the reliability signal transfers under
distribution shift by construction — the property the naive, model-inferred
noise estimators lack.

**Baselines / ablation** (`losses/objectives.py`):
`mse` (equal weight) → `invvar` (fixed `1/σ²_acq` weighting, still independent) →
`hetero` (learned variance, *unanchored*) → `axle` (anchored).

## M2 — swath-correlated loss  *(implemented, `loss=axle_spatial`)*

Independent down-weighting cannot remove a coherent stripe. M2 gives the noise a
field-patch covariance

```
Σ = diag(σ²_m) + D^½ R D^½,   D = diag(σ²_acq),   R = ρ·exp(−dist_{d_f}/ℓ) + (1−ρ)·I
```

and trains with the correlated NLL `½ rᵀΣ⁻¹r + ½ log|Σ|`, which whitens (de-stripes)
the residual. `dist_{d_f}` is the distance **along the harvester direction** `d_f`;
`ρ ∈ (0,1)` is the share of acquisition noise that is correlated and `ℓ` the
correlation length in pixels. Both are learned in unconstrained space.

**M2 is a strict superset of M1**: at `ρ = 0` the matrix is diagonal and the loss is
exactly `axle`, so M2-vs-M1 is a one-parameter ablation rather than a different model
(asserted in `tests/test_spatial.py::test_rho_zero_reduces_exactly_to_m1`).

Three pieces make it trainable:

1. **Direction `d_f`** (`data/direction.py`) — a discrete Radon scan. For each
   candidate angle we project the pixels onto the *across*-track axis, bin at
   one-pixel spacing, and score the angle by the share of the `n_i` variance carried
   by that across-track profile; `d_f` is the argmax and the score is a *strength*.
   Below `min_strength` (default 0.10) a field gets **no** direction and the kernel
   falls back to isotropic — an honest abstention instead of a fabricated angle.
   Run once per cache: `python scripts/estimate_directions.py data/cache/Germany`.
2. **Field patches** (`data/patches.py`) — each field raster is tiled into
   `tile × tile` blocks (default 16, so ≤256 pixels) and one block is one training
   item, carrying `(row, col)` and its field's `d_f`. Tiling keeps the dense
   `k × k` Cholesky exact and cheap while `tile ≫ ℓ`, so almost no real correlation
   is cut; ragged blocks are padded and the padding is neutralised to identity.
3. **Batched solve** (`losses/spatial.py`) — a batched Cholesky over `(B, K, K)`.
   At `K ≤ 256` this is ~10⁹ flops per step, well below the backbone's cost, so the
   conjugate-gradient + Lanczos path the plan sketched is not needed at this patch
   size; it would only be required for whole-field patches (up to ~10⁴ pixels).

Only the *objective* changes — validation stays on the ordinary per-pixel path with
the marginal variance `σ²_m + σ²_acq`, so every metric stays comparable to M1.

### Is `d_f` really there? (Germany, 299 fields)

The estimator is calibrated against a null in which `n_i` is permuted *within* each
field (same geometry, same marginal, no stripes):

| | median strength | fields ≥ 0.10 |
|---|---|---|
| real `n_i` | **0.131** | 76.3% |
| shuffled `n_i` (null) | 0.062 | 10.7% |

Real fields score **1.97×** their own null and beat it in **96.0%** of cases, so the
stripe orientation is a genuine signal — but a modest one, which is why the default
threshold is set at the null's 90th percentile (≈0.10, i.e. a ~10% false-positive
rate) and unresolved fields stay isotropic. On synthetic data with a planted swath
the estimator recovers 100% of angles (median strength 0.92, error ≤5°).

## Evaluation

- **Accuracy**: RMSE, R² at pixel and field level (field = mean over its pixels).
- **Calibration**: Gaussian NLL, PICP@90 (interval coverage).
- **Reliability-stratified gap**: `R²_pooled − R²_trustworthy` (`n_i ≥ 5`). A method
  that gains by *not* fitting unreliable labels shows a positive gap — this
  separates a genuine robustness gain from fitting noise and makes "SOTA under
  shift" well-posed.

## Falsification (per the research plan)

Train AXLE vs. the equal-weight Deep-Ensemble baseline on Argentina soybean, LOYO.
**Negative control**: permute the instrument metadata within each field → the OOD
gain over the baseline should vanish (a mis-anchored covariance is *not* helpful,
and can be worse — verified numerically in `tests/test_spatial.py`). **Positive
control**: freeze `σ²_m`, keep only the anchored term → most of the OOD gain
should remain, isolating the anchoring as the causal ingredient.
