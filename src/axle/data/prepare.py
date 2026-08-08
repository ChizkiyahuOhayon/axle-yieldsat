"""Convert a country's preprocessed NetCDF into a fast, training-ready cache.

The preprocessed YieldSAT release stores one NetCDF per country with a flattened
``(index, time_step, band)`` sample tensor. Loading multi-GB NetCDF per batch is
slow, so we materialise once into:

* ``sample.npy``  -- float32 memmap ``(N, T, C_dyn)`` of the time-varying bands (raw values),
* ``static.npy``  -- float32 memmap ``(N, C_static)``, written only when static bands
  are selected (see below),
* ``meta.parquet`` -- per-pixel table: target, field/crop/year/farm, row/col, and the
  joined AXLE reliability signals (n_i, s_i, quality, sigma2_acq_raw),
* ``norm.json``   -- per-band mean/std (from the NetCDF ``stats-*`` coords) for z-scoring,
* ``bands.json``  -- the ordered band names: a plain list (dynamic only) or
  ``{"dynamic": [...], "static": [...]}``.

**Static bands.** The release's 120-band tensor repeats soil (96), DEM (5) and
coordinates (3) across all 24 time steps -- 104 of 120 channels are one value copied
24 times. Storing that verbatim costs 24x more than it should (all four countries at
120 bands is ~140 GB, which does not fit on a typical scratch disk), so we detect
constant-over-time bands empirically and store them once per pixel. Same information,
~5.9x smaller, and the model can use them as what they are (a static context vector)
instead of a sequence.

Run once per country; training then reads the memmap with near-zero overhead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .reliability import build_reliability_table

# Default backbone input: the 12 Sentinel-2 L2A bands (matches the benchmark's S2-only setting).
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

# The release's four time-varying weather channels (the other 7 CSV columns are dropped
# upstream). S2 + these are the only bands that actually change across time_step.
WEATHER_BANDS = ["temp_mean", "temp_max", "temp_min", "total_prec"]

ALL_BANDS = "all"  # sentinel for "every band in the NetCDF"


def split_static_bands(ds, bands: list[str], bidx: list[int], probe: int = 2000,
                       tol: float = 1e-6) -> tuple[list[str], list[str]]:
    """Split ``bands`` into (dynamic, static) by measuring variation across time.

    A band is static when its largest *within-pixel* variation over ``time_step`` is
    negligible against its *between-pixel* variation. The ratio is scale-free, so the
    test does not depend on a channel's units -- reflectance in the thousands and
    coordinates in [-1, 1] are judged the same way. Decided from the data rather than
    from band names, so an unexpected release layout cannot silently mislabel a channel.

    A band that is constant everywhere (degenerate, no information) counts as static:
    storing it once is the right call either way.
    """
    n = ds.sizes["index"]
    take = np.unique(np.linspace(0, n - 1, min(probe, n)).astype(int))
    x = ds["sample"].isel(index=take, band=bidx).values.astype(np.float64)  # (p, T, C)
    with np.errstate(invalid="ignore"):
        within = np.nan_to_num(np.nanmax(np.nanmax(x, axis=1) - np.nanmin(x, axis=1), axis=0))
        overall = np.nan_to_num(np.nanstd(x, axis=(0, 1)))                  # (C,)
    is_static = within <= tol * np.maximum(overall, np.finfo(np.float64).tiny)
    dynamic = [b for b, s in zip(bands, is_static) if not s]
    static = [b for b, s in zip(bands, is_static) if s]
    return dynamic, static


def _decode(attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        try:
            out[int(k)] = v
        except (ValueError, TypeError):
            pass
    return out


def prepare_country(
    netcdf: str,
    out_dir: str,
    both_zip: str | None = None,
    bands: list[str] | None = None,
    chunk: int = 50_000,
) -> None:
    """Materialise one country's NetCDF (+ optional reliability join) into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(netcdf)
    n, t = ds.sizes["index"], ds.sizes["time_step"]
    all_bands = [str(b) for b in ds["band"].values]

    bands = all_bands if bands in (ALL_BANDS, [ALL_BANDS]) else (bands or S2_BANDS)
    missing = [b for b in bands if b not in all_bands]
    if missing:
        raise ValueError(f"bands not in {Path(netcdf).name}: {missing}")
    bidx = [all_bands.index(b) for b in bands]

    dyn_names, static_names = split_static_bands(ds, bands, bidx)
    dyn_idx = [all_bands.index(b) for b in dyn_names]
    static_idx = [all_bands.index(b) for b in static_names]
    print(f"[prepare] {len(bands)} bands -> {len(dyn_names)} time-varying + "
          f"{len(static_names)} static (stored once per pixel, not x{t})")

    # --- dynamic memmap (chunked to bound memory) ---
    mm = np.lib.format.open_memmap(out / "sample.npy", mode="w+", dtype=np.float32,
                                   shape=(n, t, len(dyn_names)))
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        mm[i:j] = ds["sample"].isel(index=slice(i, j), band=dyn_idx).values.astype(np.float32)
    mm.flush()

    # --- static memmap: one value per pixel, read from the first time step ---
    if static_names:
        sm = np.lib.format.open_memmap(out / "static.npy", mode="w+", dtype=np.float32,
                                       shape=(n, len(static_names)))
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            block = ds["sample"].isel(index=slice(i, j), band=static_idx).values.astype(np.float32)
            # a pixel's static value is constant in time, but the first slot can be NaN
            # padding -- take the first finite entry instead of blindly slot 0
            sm[i:j] = _first_finite(block)
        sm.flush()

    # --- per-pixel metadata table ---
    fmap = _decode(ds["field_shared_name"].attrs)
    cmap = _decode(ds["crop"].attrs)
    ymap = _decode(ds["year"].attrs)
    amap = _decode(ds["farm_identifier"].attrs)
    field = np.vectorize(fmap.get)(ds["field_shared_name"].values)
    meta = pd.DataFrame(
        {
            "index": ds["index"].values.astype(str),
            "target": ds["target"].values.astype(np.float32),
            "field_shared_name": field,
            "farm": np.vectorize(amap.get)(ds["farm_identifier"].values),
            "crop": np.vectorize(cmap.get)(ds["crop"].values),
            "year": np.vectorize(ymap.get)(ds["year"].values).astype(np.int32),
            "row": ds["row"].values.astype(np.int32),
            "col": ds["col"].values.astype(np.int32),
        }
    )

    # --- join AXLE reliability signals (n_i, s_i, quality) by (field, row, col) ---
    if both_zip is not None:
        country = _country_of(netcdf, meta)
        rel = build_reliability_table(both_zip, country)
        cols = ["field_shared_name", "row", "col", "n_i", "s_i", "quality", "quality_idx", "sigma2_acq_raw"]
        meta = meta.merge(rel[cols], on=["field_shared_name", "row", "col"], how="left")
        # verify the join reproduces the target exactly (data integrity gate)
        chk = meta.dropna(subset=["n_i"])
        if len(chk):
            merged = chk.merge(rel[["field_shared_name", "row", "col", "mean_target"]],
                               on=["field_shared_name", "row", "col"], how="left")
            d = np.abs(merged["target"].values - merged["mean_target"].values)
            frac = float((d < 1e-4).mean()) * 100
            print(f"[prepare] reliability join: {frac:.2f}% target-exact, "
                  f"coverage n_i={meta['n_i'].notna().mean()*100:.1f}%")
    meta.to_parquet(out / "meta.parquet", index=False)

    # --- normalisation stats (from NetCDF stats-* coords) ---
    norm = {
        b: {"mean": float(ds["stats-mean"].isel(band=k)), "std": float(ds["stats-std"].isel(band=k))}
        for b, k in zip(bands, bidx)
    }
    (out / "norm.json").write_text(json.dumps(norm, indent=2))
    # plain list stays the format when there is nothing static (old caches keep working)
    manifest = {"dynamic": dyn_names, "static": static_names} if static_names else dyn_names
    (out / "bands.json").write_text(json.dumps(manifest, indent=2))
    gb = (n * t * len(dyn_names) + n * len(static_names)) * 4 / 1e9
    print(f"[prepare] wrote {out}  (N={n:,}  T={t}  C_dyn={len(dyn_names)} "
          f"C_static={len(static_names)}  {gb:.2f} GB)")


def _first_finite(block: np.ndarray) -> np.ndarray:
    """(B, T, C) -> (B, C): each pixel's first finite value over time, NaN if it has none."""
    finite = np.isfinite(block)
    first = np.argmax(finite, axis=1)                       # (B, C), 0 where all-NaN
    out = np.take_along_axis(block, first[:, None, :], axis=1)[:, 0, :]
    return np.where(finite.any(axis=1), out, np.nan)


def _country_of(netcdf: str, meta: pd.DataFrame) -> str:
    for c in ("Argentina", "Brazil", "Germany", "Uruguay"):
        if c in str(netcdf) or meta["field_shared_name"].iloc[0].startswith(c):
            return c
    raise ValueError(f"cannot infer country from {netcdf}")
