"""Realization join: match pending forecasts with realized bar t+1 and score.

When bar t+1 closes, match it to pending forecast from bar t, compute:
    realized_logret = log(C_{t+1} / C_t)
    z = realized_logret / scale
    hit[7] = 1 if realized_logret < q_final[tau] else 0
    pinball[7] = max(tau * e, (tau - 1) * e)
    in_50 = 1 if (q_final[2] <= realized_logret <= q_final[4]) else 0
    in_80 = 1 if (q_final[1] <= realized_logret <= q_final[5]) else 0
    in_90 = 1 if (q_final[0] <= realized_logret <= q_final[6]) else 0
    baseline_pinball[7] = pinball of scale-only baseline (q_base = z_base * scale)
Append to scored_log.parquet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..logging_setup import get_logger
from .logger import DEFAULT_OPS_STORE

log = get_logger("ops.realization_join")

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
# Standard normal quantiles as standard scale-only baseline representation
Z_BASELINE = np.array([norm.ppf(t) for t in TAUS], dtype=np.float64)


def compute_pinball(y: float, q: np.ndarray, taus: np.ndarray | list[float] = TAUS) -> np.ndarray:
    """Compute pinball loss for a scalar realization y and quantile predictions q."""
    y_arr = float(y)
    q_arr = np.asarray(q, dtype=np.float64)
    t_arr = np.asarray(taus, dtype=np.float64)
    e = y_arr - q_arr
    return np.maximum(t_arr * e, (t_arr - 1.0) * e)


class RealizationJoiner:
    """Manages matching pending forecasts to realized outcomes."""

    def __init__(self, store_dir: Path | str | None = None):
        self.store_dir = Path(store_dir) if store_dir is not None else DEFAULT_OPS_STORE
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.scored_parquet = self.store_dir / "scored_log.parquet"
        self.pending_file = self.store_dir / "pending_forecasts.parquet"

    def register_forecast(self, forecast_dict: dict[str, Any]) -> None:
        """Register a forecast awaiting future realization."""
        df_new = pd.DataFrame([forecast_dict])
        if self.pending_file.exists():
            try:
                df_prev = pd.read_parquet(self.pending_file)
                df_all = pd.concat([df_prev, df_new], ignore_index=True)
            except Exception:
                df_all = df_new
        else:
            df_all = df_new
        df_all.to_parquet(self.pending_file, index=False)

    def process_closed_bar(self, current_bar: dict[str, Any] | pd.Series) -> Optional[dict[str, Any]]:
        """Attempt to join newly closed bar t+1 with a pending forecast made at bar t.

        current_bar has keys:
            close_mid, bar_start_utc, bar_end_utc, trade_date, slot
        """
        if not self.pending_file.exists():
            return None

        pending_df = pd.read_parquet(self.pending_file)
        if pending_df.empty:
            return None

        bar_close = float(current_bar["close_mid"])
        bar_start = pd.Timestamp(current_bar["bar_start_utc"])
        bar_end = pd.Timestamp(current_bar["bar_end_utc"])
        slot = int(current_bar["slot"])
        trade_date = str(current_bar["trade_date"])

        # Match condition:
        # A forecast made at bar t forecasts target bar t+1.
        # It may match on target_bar_start_utc, or (target_trade_date, target_slot).
        match_idx = None
        for i, row in pending_df.iterrows():
            # Check target timestamp match or slot/date match
            if pd.notna(row.get("target_bar_start_utc")) and row.get("target_bar_start_utc") is not None:
                if pd.Timestamp(row["target_bar_start_utc"]) == bar_start:
                    match_idx = i
                    break
            elif str(row.get("target_trade_date")) == trade_date and int(row.get("target_slot", -1)) == slot:
                match_idx = i
                break

        if match_idx is None:
            # Also try matching if input_bar_end_utc <= bar_start and target_slot == slot
            for i, row in pending_df.iterrows():
                if int(row.get("target_slot", -1)) == slot and str(row.get("trade_date")) <= trade_date:
                    match_idx = i
                    break

        if match_idx is None:
            return None

        matched_row = pending_df.iloc[match_idx].to_dict()
        # Remove matched row from pending
        remaining = pending_df.drop(index=match_idx).reset_index(drop=True)
        remaining.to_parquet(self.pending_file, index=False)

        # Compute scoring
        c_t = float(matched_row["last_close"])
        realized_logret = float(np.log(bar_close / c_t))
        scale = float(matched_row.get("scale", 1.0))
        z = realized_logret / scale if scale > 0 else 0.0

        q_final = np.asarray(matched_row["q_final"], dtype=np.float64)
        taus = np.array(TAUS)
        hit = (realized_logret < q_final).astype(int).tolist()
        pinball = compute_pinball(realized_logret, q_final, taus).tolist()

        in_50 = int(q_final[2] <= realized_logret <= q_final[4])
        in_80 = int(q_final[1] <= realized_logret <= q_final[5])
        in_90 = int(q_final[0] <= realized_logret <= q_final[6])

        # Scale-only baseline: quantiles = Z_BASELINE * scale
        q_base = Z_BASELINE * scale
        base_pinball = compute_pinball(realized_logret, q_base, taus).tolist()

        scored_record = {
            "evaluated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "target_bar_start_utc": bar_start.isoformat(),
            "target_bar_end_utc": bar_end.isoformat(),
            "input_bar_end_utc": str(matched_row.get("bar_end_utc", matched_row.get("input_bar_end_utc"))),
            "trade_date": trade_date,
            "slot": slot,
            "c_t": c_t,
            "c_next": bar_close,
            "realized_logret": realized_logret,
            "scale": scale,
            "z": z,
            "q_final": q_final.tolist(),
            "hit": hit,
            "pinball": pinball,
            "in_50": in_50,
            "in_80": in_80,
            "in_90": in_90,
            "baseline_pinball": base_pinball,
            "model_version": matched_row.get("model_version", "unknown"),
            "health_state": matched_row.get("health_state", "GREEN"),
            "aci_state": matched_row.get("aci_state", {}),
        }

        # Append to scored_log.parquet
        df_new_scored = pd.DataFrame([scored_record])
        if self.scored_parquet.exists():
            try:
                df_curr_scored = pd.read_parquet(self.scored_parquet)
                df_scored_all = pd.concat([df_curr_scored, df_new_scored], ignore_index=True)
            except Exception:
                df_scored_all = df_new_scored
        else:
            df_scored_all = df_new_scored

        df_scored_all.to_parquet(self.scored_parquet, index=False)
        log.info("Matched and scored forecast for bar %s (realized_ret=%.5f, z=%.3f)",
                 bar_start, realized_logret, z)
        return scored_record
