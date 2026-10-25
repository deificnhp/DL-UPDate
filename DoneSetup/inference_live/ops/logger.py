"""Operational logger for inference predictions and health diagnostics.

Appends rows to DoneSetup/ops_store/forecast_log.parquet (and optional JSONL fallback).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..logging_setup import get_logger

log = get_logger("ops.logger")

DEFAULT_OPS_STORE = Path(__file__).resolve().parents[2] / "ops_store"


class OperationalLogger:
    """Lightweight, append-only operational logger."""

    def __init__(self, store_dir: Path | str | None = None):
        self.store_dir = Path(store_dir) if store_dir is not None else DEFAULT_OPS_STORE
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.forecast_parquet = self.store_dir / "forecast_log.parquet"

    def log_forecast(self, row: dict[str, Any]) -> None:
        """Append a prediction record to forecast_log.parquet."""
        # Convert row to DataFrame
        df_new = pd.DataFrame([row])
        if self.forecast_parquet.exists():
            try:
                df_existing = pd.read_parquet(self.forecast_parquet)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception as exc:
                log.warning("Failed to read existing forecast_log: %s. Overwriting.", exc)
                df_combined = df_new
        else:
            df_combined = df_new

        df_combined.to_parquet(self.forecast_parquet, index=False)
        log.info("Forecast logged to %s (total rows: %d)", self.forecast_parquet, len(df_combined))

    def read_forecasts(self) -> pd.DataFrame:
        """Read all logged forecasts."""
        if not self.forecast_parquet.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.forecast_parquet)
