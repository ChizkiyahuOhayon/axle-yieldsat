"""CLI: build the AXLE per-pixel reliability table for a country and (optionally)
verify it joins exactly to that country's preprocessed NetCDF.

Thin wrapper over ``axle.data.reliability`` — the same code the training cache
uses, so there is a single source of truth for the join.

Examples
--------
    python scripts/extract_reliability_table.py --both /data/Both.zip \
        --country Germany --netcdf /data/.../Germany/merge_*.nc \
        --out reliability_Germany.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from axle.data.reliability import COUNTRIES, build_reliability_table


def verify(table: pd.DataFrame, netcdf: str) -> None:
    import xarray as xr

    ds = xr.open_dataset(netcdf)
    fmap = {int(k): v for k, v in ds["field_shared_name"].attrs.items() if str(k).isdigit()}
    nc = pd.DataFrame(
        {
            "field_shared_name": np.vectorize(fmap.get)(ds["field_shared_name"].values),
            "row": ds["row"].values.astype(int),
            "col": ds["col"].values.astype(int),
            "nc_target": ds["target"].values,
        }
    )
    m = table.merge(nc, on=["field_shared_name", "row", "col"], how="inner")
    d = np.abs(m["mean_target"].values - m["nc_target"].values)
    print(f"[verify] joined {len(m):,} px | target-exact {float((d < 1e-4).mean())*100:.2f}% "
          f"| max_abs_diff {np.nanmax(d):.6g} | n_i coverage {float(m['n_i'].notna().mean())*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", required=True, help="path to Both.zip")
    ap.add_argument("--country", default="Germany", help=f"one of {COUNTRIES} or 'all'")
    ap.add_argument("--netcdf", help="optional NetCDF to verify the join against")
    ap.add_argument("--out", help="output parquet (single country)")
    ap.add_argument("--out-dir", default=".", help="output dir when --country all")
    ap.add_argument("--limit", type=int, default=None, help="max fields (debug)")
    args = ap.parse_args()

    countries = COUNTRIES if args.country == "all" else [args.country]
    for c in countries:
        table = build_reliability_table(args.both, c, limit=args.limit)
        print(f"[{c}] {len(table):,} px | {table['field_shared_name'].nunique()} fields | "
              f"quality {table.groupby('quality')['row'].count().to_dict()}")
        if args.netcdf and len(countries) == 1:
            verify(table, args.netcdf)
        out = args.out or f"{args.out_dir}/reliability_{c}.parquet"
        table.to_parquet(out, index=False)
        print(f"[{c}] wrote {out}")


if __name__ == "__main__":
    main()
