"""CLI: build the training cache for one or all countries.

Reads a country's preprocessed NetCDF (and, optionally, ``Both.zip`` to join the
AXLE reliability signals) and writes a memmap + metadata cache under ``--out``.

Examples
--------
    # Germany, S2 bands, with reliability join, verifying the join integrity:
    python scripts/prepare.py \
        --netcdf /data/Preprocessed/Germany/merge_s2-soil-dem-weather-coords.nc \
        --both /data/Both.zip --country Germany --out data/cache/Germany

    # All four countries at once (NetCDFs laid out as <root>/<Country>/merge_*.nc):
    python scripts/prepare.py --root /data/Preprocessed --both /data/Both.zip --out data/cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

from axle.data.prepare import S2_BANDS, prepare_country
from axle.data.reliability import COUNTRIES

NC_NAME = "merge_s2-soil-dem-weather-coords.nc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netcdf", help="single-country NetCDF path")
    ap.add_argument("--root", help="root holding <Country>/merge_*.nc (all-countries mode)")
    ap.add_argument("--country", help="country name (single mode; inferred from path if omitted)")
    ap.add_argument("--both", help="path to Both.zip for the reliability join (recommended)")
    ap.add_argument("--out", required=True, help="output cache dir (or parent dir in --root mode)")
    ap.add_argument("--bands", nargs="*", default=S2_BANDS, help="bands to cache (default: 12 S2)")
    args = ap.parse_args()

    if args.root:
        for c in COUNTRIES:
            nc = Path(args.root) / c / NC_NAME
            if not nc.exists():
                print(f"[skip] {nc} not found")
                continue
            prepare_country(str(nc), str(Path(args.out) / c), both_zip=args.both, bands=args.bands)
    else:
        assert args.netcdf, "provide --netcdf or --root"
        prepare_country(args.netcdf, args.out, both_zip=args.both, bands=args.bands)


if __name__ == "__main__":
    main()
