# Data

Full field-by-field dissection lives in the project's `docs/DATA_DICTIONARY.md`
(one level up, outside this repo). This file is the short version needed to run AXLE.

## Layout of `Both.zip`

```
Both.zip                                   # 23 GB, zip64 — use Python zipfile, not `unzip`
├── Preprocessed/<Country>/<Country>.zip   # → merge_s2-soil-dem-weather-coords.nc (one per country)
└── Raw/<Country>/<Country>.zip            # → one directory per field (rasters + metadata)
```

Countries: Argentina, Brazil, Germany, Uruguay. 2,173 fields, ~12.2 M pixels total.
`scripts/prepare.py` reads the **Preprocessed** NetCDF for model inputs and the
**Raw** `yield_masks/` for the AXLE reliability signals (via `Both.zip` directly —
no full extraction needed; it reads the stored inner zips with a seekable slice).

## Preprocessed NetCDF (`merge_s2-soil-dem-weather-coords.nc`)

Dimensions `(index = pixel, time_step = 24, band = 120)`.

- `sample (index,24,120) float32` — fused input, **raw physical values** (S2 reflectance
  ×10000, soil/DEM/weather raw, coords pre-normalised); ~9% NaN (cloud/padding).
- `target (index,) float32` — per-pixel yield (t/ha), clipped to [0, 20].
- `row, col, field_shared_name, crop, year, farm_identifier` — reconstruct fields and
  build splits (categorical vars are integer-coded; the string map is in their `attrs`).
- `stats-mean/std/min/max (band,)` — per-band normalisation constants.

The 120 bands = 12 Sentinel-2 + 5 DEM + 96 soil (8 properties × 6 depths × [value,
uncertainty]) + 4 weather (temp mean/max/min, total_prec) + 3 coords. AXLE's default
input is the 12 S2 bands (matches the benchmark's S2-only setting); change with `--bands`.

## Raw reliability signals (`Raw/<Country>/<field>/yield_masks/`)

Three co-registered rasters (float32, nodata −1):

| raster (`*_scaled_yield_masked_regional_statistical_outlier.tif`) | symbol | meaning |
|---|---|---|
| `mean_...`   | —     | per-pixel yield (**identical to NetCDF `target`**) |
| `number_...` | `n_i` | harvester support-point count (0–27; 0 = interpolated headland) |
| `std_...`    | `s_i` | within-pixel spread of the support points |

plus `yieldmap_quality ∈ {Good, Average, Bad}` in `metadata-<field>.json`.

## The join (verified exact)

`prepare.py` joins these to each training pixel by `(field_shared_name, row, col)`.
On Germany the join reproduces the NetCDF `target` for **100.00%** of pixels
(max abs diff 0), with **~99%** reliability-signal coverage; `prepare.py` prints
this integrity check every run. The ~1% of pixels without a signal fall back to
`has_rel = 0` (AXLE degrades to heteroscedastic regression for them).

## Local vs. server

The Preprocessed NetCDFs are large (Argentina 63 GB, Brazil 50 GB, Uruguay 26 GB,
Germany 7 GB). Prepare them on the server (`/home/smbu/dy/nas/yieldsat`); a laptop
can handle Germany. `sample.npy` for the 12 S2 bands is ~6 GB (Argentina) down to
0.7 GB (Germany).
