"""Memmap-backed pixel-sequence dataset for the prepared YieldSAT cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class YieldSATPixels(Dataset):
    """One item = one 10 m pixel's season-aligned time series and its yield.

    Reads the ``prepare_country`` cache (``sample.npy`` memmap + ``meta.parquet``),
    z-scores with the stored per-band stats, and returns the AXLE reliability
    signals alongside the target so the loss can consume them.

    Returns per item::

        sample        FloatTensor (T, C)   -- normalised, NaN-filled inputs
        mask          FloatTensor (T,)     -- 1 = real observation, 0 = padded step
        target        FloatTensor ()       -- yield (t/ha)
        sigma2_acq    FloatTensor ()       -- s_i^2 / max(n_i, 1); 0 if signal missing
        has_rel       FloatTensor ()       -- 1.0 if the reliability signal is present
        quality_idx   LongTensor  ()       -- 0/1/2 (Good/Average/Bad), 3 if missing
    """

    def __init__(self, cache_dir: str, indices: np.ndarray | None = None, nan_fill: float = 0.0):
        cache = Path(cache_dir)
        self.sample = np.load(cache / "sample.npy", mmap_mode="r")  # (N, T, C)
        self.meta = pd.read_parquet(cache / "meta.parquet")
        norm = json.loads((cache / "norm.json").read_text())
        bands = json.loads((cache / "bands.json").read_text())
        self.mean = np.array([norm[b]["mean"] for b in bands], dtype=np.float32)
        self.std = np.array([norm[b]["std"] for b in bands], dtype=np.float32) + 1e-6
        self.nan_fill = float(nan_fill)

        self.rows = np.arange(len(self.meta)) if indices is None else np.asarray(indices)
        m = self.meta
        self._target = m["target"].to_numpy(np.float32)
        self._s2acq = np.nan_to_num(m.get("sigma2_acq_raw", pd.Series(np.zeros(len(m)))).to_numpy(np.float32))
        self._has_rel = m["sigma2_acq_raw"].notna().to_numpy(np.float32) if "sigma2_acq_raw" in m else np.zeros(len(m), np.float32)
        qi = m["quality_idx"].to_numpy() if "quality_idx" in m else np.full(len(m), -1)
        self._qidx = np.where(qi < 0, 3, qi).astype(np.int64)  # 3 = missing bucket

    @property
    def num_features(self) -> int:
        return self.sample.shape[2]

    @property
    def seq_len(self) -> int:
        return self.sample.shape[1]

    def __len__(self) -> int:
        return len(self.rows)

    def _inputs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalise raw band values and derive the valid-timestep mask."""
        mask = np.isfinite(x).any(axis=-1).astype(np.float32)
        x = (x - self.mean) / self.std
        return np.nan_to_num(x, nan=self.nan_fill, posinf=self.nan_fill, neginf=self.nan_fill), mask

    def __getitem__(self, i: int) -> dict:
        r = int(self.rows[i])
        x, mask = self._inputs(np.asarray(self.sample[r], dtype=np.float32))  # (T, C), (T,)
        return {
            "sample": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "target": torch.tensor(self._target[r]),
            "sigma2_acq": torch.tensor(self._s2acq[r]),
            "has_rel": torch.tensor(self._has_rel[r]),
            "quality_idx": torch.tensor(self._qidx[r]),
        }

    def gather(self, positions: np.ndarray) -> dict:
        """Read many items at once (leading axis = ``len(positions)``).

        One vectorised memmap read instead of a Python loop. Once the cache is warm
        this is ~12x cheaper per item than repeated ``__getitem__`` (1.6 vs 19 us on
        Germany); cold reads are disk-bound either way. Used by the M2 patch dataset,
        where a single item is hundreds of pixels (see :mod:`axle.data.patches`).
        """
        r = self.rows[np.asarray(positions, dtype=np.int64)]
        x, mask = self._inputs(np.asarray(self.sample[r], dtype=np.float32))  # (k, T, C), (k, T)
        return {
            "sample": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "target": torch.from_numpy(self._target[r]),
            "sigma2_acq": torch.from_numpy(self._s2acq[r]),
            "has_rel": torch.from_numpy(self._has_rel[r]),
            "quality_idx": torch.from_numpy(self._qidx[r]),
        }
