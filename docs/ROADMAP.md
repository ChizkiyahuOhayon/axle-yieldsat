# Roadmap

Staging mirrors the research plan (P1 → P2).

## v0.1 — released (this repo)

- [x] Data pipeline: `Both.zip` → memmap cache + reliability join (verified target-exact).
- [x] Backbones: LSTM, TempCNN, Transformer (backbone-agnostic).
- [x] **AXLE-M1**: acquisition-anchored heteroscedastic NLL + learnable grade scale.
- [x] Baselines/ablation: `mse`, `invvar`, `hetero`, `axle`.
- [x] Protocols: CV10, LOYO, LORO (field-grouped, leakage-checked).
- [x] Evaluation: pixel/field RMSE·R², NLL, PICP@90, reliability-stratified gap.
- [x] Reproducibility: Hydra configs, seeds, per-fold predictions + metrics, W&B hook.
- [x] Tests: objectives, grade-scale, and the M2 mechanism claim on a synthetic swath.

## v0.2 — AXLE-M2 wiring (swath-correlated loss)

The correlated NLL is implemented and unit-tested in `losses/spatial.py`. What
remains is to feed it field patches:

1. **Field-patch dataset**: group a field's pixels into a patch, carry `(row, col)`
   coordinates and the harvester direction `d_f`.
2. **Direction estimation**: Radon transform on the support-count raster (one angle
   per field, cached at `prepare` time); fall back to isotropic where no stripe.
3. **Batched solve**: replace the dense Cholesky with conjugate-gradient +
   stochastic-Lanczos-quadrature, exploiting the along-track Toeplitz structure
   (O(k log k) matvec) so it scales to large fields.
4. **Trainer hook**: a patch collate + a `SpatialAXLE` loss selectable via `loss=axle_spatial`.

## v0.3 — foundation-model probes & full grid

- [ ] Frozen EO-FM probe backbones (Presto for pixel-time-series parity; Galileo linear probe).
- [ ] Cross-country / cross-crop protocols.
- [ ] `scripts/run_all.sh` producing every benchmark table from logged runs.
