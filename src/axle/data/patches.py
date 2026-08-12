"""Field-patch dataset for AXLE-M2 (one item = a spatially contiguous block of pixels).

M1's objective is separable, so a batch can be any bag of pixels. M2's correlated NLL
is *not*: it needs the pixels that share a swath to arrive together, with their
``(row, col)`` positions and the field's harvester direction :math:`d_f`.

So we tile each field's raster into ``tile x tile`` blocks and make each block one
training item. Tiling (rather than whole fields) is a deliberate trade: a block of
:math:`k \\le \\texttt{tile}^2` pixels keeps the dense :math:`k \\times k` Cholesky cheap and
exactly solvable, while ``tile`` stays far larger than the correlation length
:math:`\\ell` (a few pixels), so almost no real correlation is cut. Blocks vary in size
(fields are not rectangles), so the collate pads to the batch maximum and marks the
padding; :class:`~axle.losses.spatial.SpatialAXLE` neutralises padded entries.

Validation is unaffected -- M2 changes the *objective*, not the predictor -- so the
trainer keeps evaluating with the plain per-pixel :class:`YieldSATPixels`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset import YieldSATPixels

DIRECTIONS_FILE = "directions.parquet"


def load_directions(cache_dir: str | Path, path: str | Path | None = None) -> pd.DataFrame | None:
    """Load per-field directions, defaulting to ``<cache_dir>/directions.parquet``.

    Returns ``None`` when the file is absent -- M2 then runs fully isotropic, which is
    a valid (weaker) model rather than an error.
    """
    p = Path(path) if path else Path(cache_dir) / DIRECTIONS_FILE
    return pd.read_parquet(p) if p.exists() else None


class YieldSATPatches(Dataset):
    """One item = one field-tile: ``k`` pixels with coordinates and a travel direction.

    Args:
        cache_dir: prepared cache (same as :class:`YieldSATPixels`).
        indices: global row positions to draw from (a fold's training rows).
        tile: block side in pixels; a block holds at most ``tile**2`` pixels.
        min_pixels: drop blocks smaller than this (slivers carry no swath structure).
        directions: per-field direction table, or ``None`` for isotropic.

    Item keys mirror :class:`YieldSATPixels` with a leading pixel axis, plus
    ``coords`` (k, 2) and ``direction`` (2,).
    """

    def __init__(
        self,
        cache_dir: str,
        indices: np.ndarray | None = None,
        *,
        tile: int = 16,
        min_pixels: int = 32,
        directions: pd.DataFrame | None = None,
        **pixel_kw,
    ):
        self.pixels = YieldSATPixels(cache_dir, indices=indices, **pixel_kw)
        self.tile = int(tile)
        rows = self.pixels.rows
        m = self.pixels.meta.iloc[rows]

        # tile id = (field, row block, col block); group positions *within* self.pixels
        block = pd.DataFrame({
            "field": m["field_shared_name"].to_numpy(),
            "br": (m["row"].to_numpy() // self.tile),
            "bc": (m["col"].to_numpy() // self.tile),
            "pos": np.arange(len(rows)),
        })
        groups = block.groupby(["field", "br", "bc"], sort=True)["pos"].apply(np.asarray)
        self.patches = [g for g in groups if len(g) >= min_pixels]
        self.fields = [f for (f, _, _), g in groups.items() if len(g) >= min_pixels]
        if not self.patches:
            raise ValueError(
                f"no field tile reached min_pixels={min_pixels} (tile={tile}); "
                "lower min_pixels or raise tile"
            )

        self._coords = np.stack([m["row"].to_numpy(np.float32), m["col"].to_numpy(np.float32)], 1)
        self._dir = self._direction_lookup(directions)

    def _direction_lookup(self, directions: pd.DataFrame | None) -> np.ndarray:
        """(n_patches, 2) travel direction per patch; (0, 0) = isotropic."""
        d = np.zeros((len(self.patches), 2), np.float32)
        if directions is None:
            return d
        table = directions.set_index("field_shared_name")
        for i, field in enumerate(self.fields):
            if field in table.index:
                row = table.loc[field]
                d[i] = (row["dir_row"], row["dir_col"])
        return d

    @property
    def num_features(self) -> int:
        return self.pixels.num_features

    @property
    def seq_len(self) -> int:
        return self.pixels.seq_len

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, i: int) -> dict:
        pos = self.patches[i]
        out = self.pixels.gather(pos)
        out["coords"] = torch.from_numpy(self._coords[pos])
        out["direction"] = torch.from_numpy(self._dir[i])
        return out

    @staticmethod
    def collate_padded(batch: list[dict]) -> dict:
        """Pad a list of variable-size patches to the batch maximum.

        Adds ``pix_mask`` (B, K): 1 for real pixels, 0 for padding. Padded targets and
        reliabilities are zero and ``quality_idx`` is the "missing" bucket, so a padded
        entry is inert even before the loss masks it.
        """
        k = max(b["target"].shape[0] for b in batch)
        out: dict[str, torch.Tensor] = {}
        for key in batch[0]:
            if key == "direction":
                out[key] = torch.stack([b[key] for b in batch])
                continue
            ref = batch[0][key]
            padded = []
            for b in batch:
                t = b[key]
                pad = k - t.shape[0]
                if pad:
                    fill = torch.zeros((pad, *t.shape[1:]), dtype=t.dtype)
                    if key == "quality_idx":
                        fill += 3  # missing bucket
                    t = torch.cat([t, fill])
                padded.append(t)
            out[key] = torch.stack(padded).to(ref.dtype)
        out["pix_mask"] = torch.stack([
            torch.cat([torch.ones(b["target"].shape[0]), torch.zeros(k - b["target"].shape[0])])
            for b in batch
        ])
        return out

    collate_fn = collate_padded  # picked up by the trainer's DataLoader


class YieldSATTiles(YieldSATPatches):
    """A field tile as a *dense* ``tile x tile`` raster, for spatial-temporal backbones.

    The benchmark's strongest models are 3D-CNN based (3D-LSTM, 3D-ConvLSTM, AFF), and
    those need a grid, not a bag of pixels. Fields are irregular polygons, so a tile is
    only ~58% occupied at 16x16 and ~69% at 8x8; the empty cells are zero-filled and
    flagged in ``pix_mask``, exactly as padded pixels already are, so every loss and
    metric downstream is unchanged.

    Items keep :class:`YieldSATPatches`' keys with a *fixed* ``K = tile**2`` in row-major
    order, which lets a 3D backbone reshape ``sample`` to ``(T, C, tile, tile)`` without
    the dataset carrying a second copy of the inputs.
    """

    def __getitem__(self, i: int) -> dict:
        pos = self.patches[i]
        k = self.tile * self.tile
        coords = self._coords[pos]
        r0, c0 = coords[:, 0].min(), coords[:, 1].min()
        cell = ((coords[:, 0] - r0).astype(np.int64) * self.tile
                + (coords[:, 1] - c0).astype(np.int64))

        src = self.pixels.gather(pos)
        out: dict[str, torch.Tensor] = {}
        for key, v in src.items():
            dense = torch.zeros((k, *v.shape[1:]), dtype=v.dtype)
            if key == "quality_idx":
                dense += 3                                   # empty cells -> "missing" grade
            dense[cell] = v
            out[key] = dense

        grid_rc = torch.stack([torch.arange(k) // self.tile, torch.arange(k) % self.tile], 1)
        out["coords"] = (grid_rc + torch.tensor([r0, c0])).float()
        out["direction"] = torch.from_numpy(self._dir[i])
        out["pix_mask"] = torch.zeros(k)
        out["pix_mask"][cell] = 1.0
        return out

    @staticmethod
    def collate_dense(batch: list[dict]) -> dict:
        """Every tile already has the same K, so this is a plain stack."""
        return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}

    collate_fn = collate_dense
