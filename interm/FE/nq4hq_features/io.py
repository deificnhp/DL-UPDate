"""Thin I/O layer. Polars is preferred; pyarrow is the fallback so the pipeline and its
leakage tests also run where Polars isn't installed. Feature math never touches the frame
library (see core.py), so both backends produce identical numbers."""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
    HAVE_POLARS = True
except ImportError:  # pragma: no cover
    pl = None
    HAVE_POLARS = False
import pyarrow as pa
import pyarrow.parquet as pq


class Frame:
    """Holds the original table untouched (all identifiers/flags preserved) plus numpy views."""

    def __init__(self, obj):
        self.obj = obj
        self.backend = "polars" if (HAVE_POLARS and isinstance(obj, pl.DataFrame)) else "arrow"

    # ---- construction
    @classmethod
    def read(cls, path, prefer_polars=True):
        if prefer_polars and HAVE_POLARS:
            return cls(pl.read_parquet(path))
        return cls(pq.read_table(path))

    @classmethod
    def from_any(cls, obj):
        if HAVE_POLARS and isinstance(obj, pl.DataFrame):
            return cls(obj)
        if isinstance(obj, pa.Table):
            return cls(obj)
        try:  # pandas
            return cls(pa.Table.from_pandas(obj, preserve_index=False))
        except Exception as e:
            raise TypeError(f"unsupported frame type {type(obj)}") from e

    # ---- access
    @property
    def columns(self):
        return list(self.obj.columns) if self.backend == "polars" else self.obj.column_names

    def __len__(self):
        return self.obj.height if self.backend == "polars" else self.obj.num_rows

    def np(self, name):
        if self.backend == "polars":
            return self.obj.get_column(name).to_numpy()
        return self.obj.column(name).to_numpy()

    def np_bool(self, name):
        """Boolean flag with nulls -> False."""
        if self.backend == "polars":
            return self.obj.get_column(name).fill_null(False).cast(pl.Boolean).to_numpy()
        a = self.obj.column(name).to_pandas()
        return a.fillna(False).astype(bool).to_numpy()

    def np_time_ns(self, name):
        """Timestamp/date column -> int64 nanoseconds (UTC for tz-aware)."""
        if self.backend == "polars":
            s = self.obj.get_column(name)
            if s.dtype == pl.Date:
                return s.cast(pl.Int64).to_numpy().astype(np.int64) * 86_400_000_000_000
            return s.dt.epoch("ns").to_numpy().astype(np.int64)   # UTC epoch, tz-aware or naive
        col = self.obj.column(name)
        if pa.types.is_date(col.type):
            return col.cast(pa.date32()).to_numpy().astype("datetime64[D]").astype("datetime64[ns]").astype(np.int64)
        if pa.types.is_timestamp(col.type):
            return col.cast(pa.timestamp("ns", tz=col.type.tz)).to_numpy().astype("datetime64[ns]").astype(np.int64)
        return np.asarray(col.to_numpy(), dtype="datetime64[ns]").astype(np.int64)

    # ---- transforms
    def take(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        if self.backend == "polars":
            return Frame(self.obj[idx])
        return Frame(self.obj.take(pa.array(idx)))

    def with_columns(self, cols: dict):
        if self.backend == "polars":
            return Frame(self.obj.with_columns([pl.Series(k, v) for k, v in cols.items()]))
        t = self.obj
        for k, v in cols.items():
            arr = pa.array(v, from_pandas=True)  # NaN -> null for floats
            if k in t.column_names:
                t = t.set_column(t.column_names.index(k), k, arr)
            else:
                t = t.append_column(k, arr)
        return Frame(t)

    def filter(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if self.backend == "polars":
            return Frame(self.obj.filter(pl.Series(mask)))
        return Frame(self.obj.filter(pa.array(mask)))

    def write(self, path):
        if self.backend == "polars":
            self.obj.write_parquet(path, compression="zstd", statistics=True)
        else:
            pq.write_table(self.obj, path, compression="zstd")
