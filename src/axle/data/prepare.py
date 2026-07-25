"""Convert a country's preprocessed NetCDF into a fast, training-ready cache.

The preprocessed YieldSAT release stores one NetCDF per country with a flattened
``(index, time_step, band)`` sample tensor. Loading multi-GB NetCDF per batch is
slow, so we materialise once into:

* ``sample.npy``  -- float32 memmap ``(N, T, C)`` of the selected bands (raw values),
* ``meta.parquet`` -- per-pixel table: target, field/crop/year/farm, row/col, and the
  joined AXLE reliability signals (n_i, s_i, quality, sigma2_acq_raw),
* ``norm.json``   -- per-band mean/std (from the NetCDF ``stats-*`` coords) for z-scoring,
* ``bands.json``  -- the ordered band names.

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
    bands = bands or S2_BANDS
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(netcdf)
    n, t = ds.sizes["index"], ds.sizes["time_step"]
    all_bands = [str(b) for b in ds["band"].values]
    bidx = [all_bands.index(b) for b in bands]

    # --- sample memmap (chunked to bound memory) ---
    mm = np.lib.format.open_memmap(out / "sample.npy", mode="w+", dtype=np.float32, shape=(n, t, len(bands)))
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        mm[i:j] = ds["sample"].isel(index=slice(i, j), band=bidx).values.astype(np.float32)
    mm.flush()

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
    (out / "bands.json").write_text(json.dumps(bands, indent=2))
    print(f"[prepare] wrote {out}  (N={n:,}  T={t}  C={len(bands)})")


def _country_of(netcdf: str, meta: pd.DataFrame) -> str:
    for c in ("Argentina", "Brazil", "Germany", "Uruguay"):
        if c in str(netcdf) or meta["field_shared_name"].iloc[0].startswith(c):
            return c
    raise ValueError(f"cannot infer country from {netcdf}")
