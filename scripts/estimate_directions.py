#!/usr/bin/env python3
"""Estimate the harvester travel direction d_f for every field in a prepared cache.

Reads only ``meta.parquet`` (row/col/n_i) and writes ``directions.parquet`` next to it,
which ``loss=axle_spatial`` picks up automatically.

    python scripts/estimate_directions.py data/cache/Germany
    python scripts/estimate_directions.py data/cache/*/ --signal s_i --min-strength 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from axle.data.direction import estimate_field_directions
from axle.data.patches import DIRECTIONS_FILE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", nargs="+", help="prepared cache dir(s), e.g. data/cache/Germany")
    ap.add_argument("--signal", default="n_i", help="column whose stripes reveal d_f (default: n_i)")
    ap.add_argument("--min-strength", type=float, default=0.10,
                    help="below this variance fraction a field is left isotropic (default: 0.10)")
    ap.add_argument("--angles", type=int, default=60, help="angles scanned over [0,180) (default: 60)")
    args = ap.parse_args()

    for cache in args.cache:
        cache = Path(cache)
        meta = pd.read_parquet(cache / "meta.parquet")
        d = estimate_field_directions(meta, signal=args.signal,
                                      min_strength=args.min_strength, n_angles=args.angles)
        out = cache / DIRECTIONS_FILE
        d.to_parquet(out, index=False)
        found = d["has_direction"]
        print(f"[{cache.name}] {len(d)} fields | d_f found for {found.sum()} ({found.mean():.1%}) "
              f"| median strength {d['strength'].median():.3f} -> {out}")
        if found.any():
            print("  angle histogram (deg, 30-deg bins):",
                  d.loc[found, "theta_deg"].pipe(lambda s: pd.cut(s, bins=range(0, 181, 30)).value_counts().sort_index().to_dict()))


if __name__ == "__main__":
    main()
