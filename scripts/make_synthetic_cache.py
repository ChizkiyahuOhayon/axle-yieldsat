"""CLI: generate a tiny synthetic cache in the exact YieldSAT format.

Lets you run the full pipeline (prepare-free) in seconds without the 23 GB
dataset -- for CI, smoke tests, and a clone-and-run demo. The cache mirrors what
``scripts/prepare.py`` writes (``sample.npy`` memmap, ``meta.parquet``,
``norm.json``, ``bands.json``), including season-alignment padding and the AXLE
reliability columns, so every backbone / loss / protocol path exercises real code.

    python scripts/make_synthetic_cache.py --out data/cache/Synthetic

The target is a learnable function of the (masked) inputs plus a field-coherent
offset and harvester-style low-support noise, so R^2 climbs in a few epochs --
enough to prove the loop learns, not to benchmark anything.
"""
from __future__ import annotations

import argparse

from axle.data.synthetic import write_synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cache/Synthetic")
    ap.add_argument("--fields", type=int, default=120, help="number of synthetic fields")
    ap.add_argument("--pixels-per-field", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    write_synthetic(args.out, args.fields, args.pixels_per_field, args.seed)


if __name__ == "__main__":
    main()
