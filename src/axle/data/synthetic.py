"""Synthetic YieldSAT-format cache generator (importable).

Produces a tiny cache identical in schema to ``scripts/prepare.py`` output so the
full pipeline runs without the real dataset -- used by the CI smoke test and the
clone-and-run demo. See ``scripts/make_synthetic_cache.py`` for the CLI.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare import S2_BANDS
from .reliability import QUALITY_LEVELS


def write_static(cache: str, n_static: int = 8, seed: int = 0) -> str:
    """Add a synthetic static block (soil/DEM-like) to an existing synthetic cache.

    Mirrors the full-band cache layout: ``static.npy`` plus a ``{"dynamic","static"}``
    band manifest, so the static path can be exercised without the 60 GB release file.
    """
    rng = np.random.default_rng(seed)
    cache = Path(cache)
    meta = pd.read_parquet(cache / "meta.parquet")
    bands = json.loads((cache / "bands.json").read_text())
    dyn = bands["dynamic"] if isinstance(bands, dict) else bands
    names = [f"static_{i}" for i in range(n_static)]

    # one value per *field* (soil and terrain vary between fields, not within a pixel)
    codes = pd.factorize(meta["field_shared_name"])[0]
    per_field = rng.normal(0, 1, size=(codes.max() + 1, n_static)).astype(np.float32)
    arr = per_field[codes] + rng.normal(0, 0.05, size=(len(meta), n_static)).astype(np.float32)
    np.save(cache / "static.npy", arr)

    norm = json.loads((cache / "norm.json").read_text())
    norm.update({b: {"mean": 0.0, "std": 1.0} for b in names})
    (cache / "norm.json").write_text(json.dumps(norm, indent=2))
    (cache / "bands.json").write_text(json.dumps({"dynamic": dyn, "static": names}, indent=2))
    return str(cache)


def write_synthetic(out: str, fields: int = 120, pixels_per_field: int = 200, seed: int = 0,
                    swath: float = 0.0) -> str:
    """Write a synthetic cache to ``out`` in the exact prepare() format; return the path.

    ``swath > 0`` switches the label noise from per-pixel independent to the AXLE-M2
    generative model: each field gets a random harvester angle, and pixels are grouped
    into parallel passes ``swath`` pixels wide. Support count ``n_i`` is constant within
    a pass (so the stripes are visible to the direction estimator) and the yield error
    is *shared* by the whole pass -- coherent along track, independent across it. This
    is the structure M2 claims to exploit and M1 cannot see.
    """
    rng = np.random.default_rng(seed)
    bands, T, C = S2_BANDS, 24, len(S2_BANDS)
    farms = [f"farm{i}" for i in range(4)]
    crops = ["wheat", "rapeseed"]
    years = [2018, 2019, 2020, 2021]

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(fields=fields, pixels_per_field=pixels_per_field)
    n = fields * pixels_per_field
    sample = np.lib.format.open_memmap(out / "sample.npy", mode="w+", dtype=np.float32, shape=(n, T, C))

    # a fixed linear "truth" over band means, so the target is learnable
    w = rng.normal(size=C)
    valid_slots = np.arange(6, 20)  # real acquisitions live mid-frame (season alignment)

    rows = []
    p = 0
    for f in range(args.fields):
        field = f"Synthetic_DUP0_{rng.choice(farms)}_field{f}_{rng.choice(crops)}_{rng.choice(years)}"
        parts = field.split("_")
        farm, crop, year = parts[2], parts[4], int(parts[5])
        side = int(np.ceil(np.sqrt(args.pixels_per_field)))
        field_effect = float(rng.normal(0, 1.5))   # per-field yield offset -> learnable field structure
        theta = rng.uniform(0, np.pi)              # this field's harvester angle (swath mode)
        passes: dict[int, tuple[float, float]] = {}

        def pass_props(row_i: int, col_i: int) -> tuple[float, float]:
            """(n_i, shared error) for the harvester pass this pixel belongs to."""
            t = -np.sin(theta) * row_i + np.cos(theta) * col_i     # across-track coordinate
            p = int(np.floor(t / swath))
            if p not in passes:
                n = float(rng.integers(0, 28))
                s = 0.9
                passes[p] = (n, float(rng.normal(0, np.sqrt(s**2 / max(n, 1)))))
            return passes[p]

        for k in range(args.pixels_per_field):
            x = np.full((T, C), np.nan, np.float32)
            n_obs = rng.integers(8, len(valid_slots) + 1)
            slots = np.sort(rng.choice(valid_slots, size=n_obs, replace=False))
            sig = rng.normal(0, 1, size=(n_obs, C)).astype(np.float32)
            # field_effect is injected into the FEATURES (a field-coherent spectral offset),
            # so field-level yield structure is learnable from the input, not free noise.
            x[slots] = sig * 500 + 1500 + field_effect * 300
            sample[p] = x

            feat = np.nanmean(x, axis=0)          # crude signal summary
            clean = 6.0 + 0.002 * float(feat @ w)  # feat carries the field-coherent offset
            if swath > 0:
                n_i, shared = pass_props(k // side, k % side)
                s_i = 0.9
                sigma2 = s_i**2 / max(n_i, 1)
                target = clean + shared + rng.normal(0, 0.05)  # error coherent along the pass
            else:
                n_i = float(rng.integers(0, 28))  # harvester support count 0..27
                s_i = float(abs(rng.normal(0.9, 0.4)))
                sigma2 = s_i**2 / max(n_i, 1)
                target = clean + rng.normal(0, np.sqrt(sigma2 + 0.05))  # low-support -> noisier
            q = QUALITY_LEVELS[min(2, int(n_i < 4) + int(n_i < 10))]
            rows.append({
                "index": str(uuid.uuid4()), "target": np.float32(np.clip(target, 0, 20)),
                "field_shared_name": field, "farm": farm, "crop": crop, "year": year,
                "row": np.int32(k // side), "col": np.int32(k % side),
                "n_i": np.float32(n_i), "s_i": np.float32(s_i),
                "quality": q, "quality_idx": np.int8(QUALITY_LEVELS.index(q)),
                "sigma2_acq_raw": np.float32(sigma2),
            })
            p += 1
    sample.flush()

    pd.DataFrame(rows).to_parquet(out / "meta.parquet", index=False)
    (out / "norm.json").write_text(json.dumps({b: {"mean": 1500.0, "std": 500.0} for b in bands}, indent=2))
    (out / "bands.json").write_text(json.dumps(bands, indent=2))
    print(f"[synthetic] wrote {out}  (N={n:,}  T={T}  C={C}  fields={fields})")
    return str(out)
