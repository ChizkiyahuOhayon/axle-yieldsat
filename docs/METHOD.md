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

## M2 — swath-correlated loss  *(module + tests; wiring on the roadmap)*

Independent down-weighting cannot remove a coherent stripe. M2 gives the noise a
field-patch covariance

```
Σ = diag(σ²_m) + D^½ R(ρ, ℓ, d_f) D^½,   D = diag(σ²_acq)
```

with `R` a Matern-1/2 correlation **along the harvester direction** `d_f`
(estimated from the stripe orientation of the support-count raster). Training uses
the correlated NLL `½ rᵀΣ⁻¹r + ½ log|Σ|`, which whitens (de-stripes) the residual.
`losses/spatial.py` implements and unit-tests this; the field-patch dataloader
that feeds it is the next milestone (`docs/ROADMAP.md`).

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
