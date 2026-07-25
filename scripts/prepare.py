"""CLI: build the training cache from YieldSAT.

The recommended path is **one command straight from ``Both.zip``** -- it extracts
each country's preprocessed NetCDF on the fly, joins the AXLE reliability signals,
writes the memmap cache, and (by default) deletes the large intermediate NetCDF:

    # all four countries, cache under data/cache/<Country>:
    python scripts/prepare.py --both /nas/yieldsat/Both.zip --out data/cache

    # just one country to get started (Germany is smallest):
    python scripts/prepare.py --both /nas/yieldsat/Both.zip --country Germany --out data/cache

Notes
-----
* The extracted NetCDF is large (Germany 7 GB ... Argentina 63 GB). It is written to
  ``--nc-dir`` (default: ``<out>/_netcdf``) and removed after the cache is built
  unless ``--keep-nc`` is given. The resulting cache (12 S2 bands) is far smaller
  (Germany ~0.7 GB, Argentina ~6 GB).
* ``--nc-dir`` can point at a fast local disk to avoid extracting onto slow NAS.

Alternative inputs (if you already extracted the NetCDFs yourself):
    python scripts/prepare.py --netcdf <path>/merge_*.nc --both Both.zip --country Germany --out data/cache/Germany
    python scripts/prepare.py --root <dir with <Country>/merge_*.nc> --both Both.zip --out data/cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

from axle.data.prepare import S2_BANDS, prepare_country
from axle.data.reliability import COUNTRIES, extract_preprocessed_netcdf

NC_NAME = "merge_s2-soil-dem-weather-coords.nc"


def _prepare_from_both(both, country, out, nc_dir, keep_nc, bands):
    print(f"=== {country}: extracting NetCDF from Both.zip ===")
    nc = extract_preprocessed_netcdf(both, country, nc_dir)
    try:
        prepare_country(nc, str(Path(out) / country), both_zip=both, bands=bands)
    finally:
        if not keep_nc:
            Path(nc).unlink(missing_ok=True)
            print(f"[cleanup] removed intermediate {nc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", help="path to Both.zip (recommended one-command mode)")
    ap.add_argument("--country", help="single country; omit with --both to do all four")
    ap.add_argument("--out", required=True, help="output cache dir (parent when doing several countries)")
    ap.add_argument("--nc-dir", help="where to extract NetCDFs (default: <out>/_netcdf)")
    ap.add_argument("--keep-nc", action="store_true", help="keep the extracted NetCDF (default: delete it)")
    ap.add_argument("--bands", nargs="*", default=S2_BANDS, help="bands to cache (default: 12 S2)")
    # alternative inputs when NetCDFs are already extracted
    ap.add_argument("--netcdf", help="single-country NetCDF path (skips extraction)")
    ap.add_argument("--root", help="dir holding <Country>/merge_*.nc (skips extraction)")
    args = ap.parse_args()

    if args.netcdf:                       # explicit single NetCDF
        country = args.country or _infer_country(args.netcdf)
        prepare_country(args.netcdf, str(Path(args.out) / country) if args.country else args.out,
                        both_zip=args.both, bands=args.bands)
    elif args.root:                       # pre-extracted tree
        for c in COUNTRIES:
            nc = Path(args.root) / c / NC_NAME
            if nc.exists():
                prepare_country(str(nc), str(Path(args.out) / c), both_zip=args.both, bands=args.bands)
            else:
                print(f"[skip] {nc} not found")
    elif args.both:                       # one-command from Both.zip (recommended)
        nc_dir = args.nc_dir or str(Path(args.out) / "_netcdf")
        countries = [args.country] if args.country else list(COUNTRIES)
        for c in countries:
            _prepare_from_both(args.both, c, args.out, nc_dir, args.keep_nc, args.bands)
    else:
        ap.error("provide --both (recommended), or --netcdf, or --root")


def _infer_country(path):
    for c in COUNTRIES:
        if c in str(path):
            return c
    return "cache"


if __name__ == "__main__":
    main()
