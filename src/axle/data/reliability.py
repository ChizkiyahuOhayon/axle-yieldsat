"""Extract the AXLE acquisition-reliability signals from the YieldSAT Raw release.

For every field, the raw release ships three co-registered rasters under
``yield_masks/``:

* ``mean``   -- the per-pixel yield (identical to the preprocessed ``target``),
* ``number`` -- ``n_i``, the number of harvester support points behind the pixel,
* ``std``    -- ``s_i``, the within-pixel spread of those points,

plus a field-level ``yieldmap_quality`` grade (Good/Average/Bad) in the metadata
JSON. AXLE turns these into a per-pixel acquisition-noise variance and joins them
to the preprocessed training pixels by ``(field_shared_name, row, col)``.

The join is exact: on Germany it matches the preprocessed ``target`` for
100.00% of pixels (max abs diff 0). See ``docs/DATA.md``.
"""
from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

_MASK = {
    "mean": "yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif",
    "number": "yield_masks/number_scaled_yield_masked_regional_statistical_outlier.tif",
    "std": "yield_masks/std_scaled_yield_masked_regional_statistical_outlier.tif",
}
COUNTRIES = ("Argentina", "Brazil", "Germany", "Uruguay")
QUALITY_LEVELS = ("Good", "Average", "Bad")  # index 0/1/2; "unknown" -> -1


class _StoredSlice(io.RawIOBase):
    """A seekable view over a STORED (uncompressed) member of an outer zip.

    ``Both.zip`` stores each country zip without compression, so an inner zip's
    bytes are contiguous and we can read its central directory without extracting
    the 23 GB archive.
    """

    def __init__(self, path: str, offset: int, size: int):
        self._f = open(path, "rb")
        self._base, self._size, self._pos = offset, size, 0

    def seek(self, off, whence=0):
        self._pos = off if whence == 0 else self._pos + off if whence == 1 else self._size + off
        return self._pos

    def tell(self):
        return self._pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def read(self, n=-1):
        if n < 0 or n > self._size - self._pos:
            n = self._size - self._pos
        self._f.seek(self._base + self._pos)
        data = self._f.read(n)
        self._pos += len(data)
        return data

    def close(self):
        self._f.close()
        super().close()


def open_inner_zip(source: str, country: str, release: str = "Raw") -> zipfile.ZipFile:
    """Open the per-country inner zip for ``release`` from either data packaging.

    ``source`` may be:
      * ``Both.zip`` (a file) -- reads the inner ``{release}/{country}/{country}.zip``
        from within it via a seekable slice (no full extraction), or
      * a directory that already holds the extracted ``{release}/{country}/{country}.zip``
        (e.g. the folder where ``YieldSAT.tar.gz`` was unpacked).
    """
    p = Path(source)
    if p.is_dir():
        inner = p / release / country / f"{country}.zip"
        if not inner.exists():
            raise FileNotFoundError(f"{inner} not found (expected extracted {release} tree)")
        return zipfile.ZipFile(inner)
    outer = zipfile.ZipFile(source)
    info = outer.getinfo(f"{release}/{country}/{country}.zip")
    outer.fp.seek(info.header_offset)
    name_len, extra_len = struct.unpack("<HH", outer.fp.read(30)[26:30])
    data_off = info.header_offset + 30 + name_len + extra_len
    return zipfile.ZipFile(_StoredSlice(source, data_off, info.file_size))


def extract_preprocessed_netcdf(source: str, country: str, dest_dir: str,
                                overwrite: bool = False, bufsize: int = 1 << 24) -> str:
    """Extract a country's preprocessed NetCDF to ``dest_dir/<country>/merge_*.nc``.

    ``source`` is either ``Both.zip`` or an extracted dir with ``Preprocessed/`` (see
    :func:`open_inner_zip`). Streams the (large) file without extracting the whole
    archive; returns the path; skips if present unless ``overwrite``.
    """
    import shutil

    zf = open_inner_zip(source, country, "Preprocessed")
    member = zf.infolist()[0]  # each Preprocessed inner zip holds exactly one .nc
    out_dir = Path(dest_dir) / country
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(member.filename).name
    if target.exists() and not overwrite:
        return str(target)
    tmp = target.with_suffix(target.suffix + ".part")
    with zf.open(member) as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=bufsize)
    tmp.rename(target)  # atomic: a partial extraction never looks complete
    return str(target)


def _read_raster(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as f:
        buf = f.read()
    with rasterio.MemoryFile(buf) as mf, mf.open() as src:
        arr = src.read(1).astype("float64")
        nd = src.nodata
    return np.where(arr == nd, np.nan, arr) if nd is not None else arr


def _field_rows(field: str, meta: dict, mean, number, std) -> pd.DataFrame | None:
    """One row per valid (mean-present) pixel of a field."""
    rr, cc = np.where(np.isfinite(mean))
    if rr.size == 0:
        return None
    n_i = number[rr, cc]
    s_i = std[rr, cc]
    sigma2 = np.square(s_i) / np.maximum(np.nan_to_num(n_i, nan=0.0), 1.0)
    quality = meta.get("yieldmap_quality", "unknown")
    return pd.DataFrame(
        {
            "field_shared_name": field,
            "row": rr.astype(np.int32),
            "col": cc.astype(np.int32),
            "mean_target": mean[rr, cc].astype(np.float32),
            "n_i": n_i.astype(np.float32),
            "s_i": s_i.astype(np.float32),
            "quality": quality,
            "quality_idx": np.int8(QUALITY_LEVELS.index(quality) if quality in QUALITY_LEVELS else -1),
            "sigma2_acq_raw": sigma2.astype(np.float32),
        }
    )


def build_reliability_table(both_zip: str, country: str, limit: int | None = None) -> pd.DataFrame:
    """Build the per-pixel reliability table for one country, read from ``Both.zip``."""
    zf = open_inner_zip(both_zip, country, "Raw")
    fields = sorted({n.split("/")[0] for n in zf.namelist() if "/" in n})
    if limit:
        fields = fields[:limit]
    rows = []
    for field in fields:
        try:
            meta = json.loads(zf.read(f"{field}/metadata-{field}.json"))
            mean = _read_raster(zf, f"{field}/{_MASK['mean']}")
            number = _read_raster(zf, f"{field}/{_MASK['number']}")
            std = _read_raster(zf, f"{field}/{_MASK['std']}")
        except KeyError:
            continue
        r = _field_rows(field, meta, mean, number, std)
        if r is not None:
            rows.append(r)
    return pd.concat(rows, ignore_index=True)
