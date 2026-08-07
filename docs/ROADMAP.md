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

## v0.2 — AXLE-M2 wiring (swath-correlated loss) — released

- [x] **Direction estimation** (`data/direction.py`, `scripts/estimate_directions.py`):
      Radon scan of the support-count raster, one angle + strength per field, with an
      isotropic fallback below threshold. Runs off `meta.parquet`, so no re-`prepare`.
      Calibrated against a within-field shuffle null (see `docs/METHOD.md`).
- [x] **Field-patch dataset** (`data/patches.py`): fields tiled into `tile × tile`
      blocks carrying `(row, col)` and `d_f`; ragged blocks padded, padding neutralised.
- [x] **Batched solve**: batched dense Cholesky over `(B, K, K)`. Tiling caps `K` at
      `tile²` (256 by default), which keeps the exact solve cheaper than the backbone —
      so the CG + stochastic-Lanczos path is **deferred**, not needed at this patch size.
      It becomes necessary only if we move to whole-field patches (~10⁴ pixels), which
      the correlation length (a few pixels) does not currently justify.
- [x] **Trainer hook**: `loss=axle_spatial` switches the *training* set to patches
      (validation stays per-pixel); `patch.tile` / `patch.min_pixels` /
      `train.patch_batch_size` in `configs/config.yaml`.

## v0.3 — foundation-model probes & full grid

- [ ] Frozen EO-FM probe backbones (Presto for pixel-time-series parity; Galileo linear probe).
- [ ] Cross-country / cross-crop protocols.
- [ ] `scripts/run_all.sh` producing every benchmark table from logged runs.
