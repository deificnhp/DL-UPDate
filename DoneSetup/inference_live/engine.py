"""The live inference engine.

Orchestration, in one place:

    minutes -> session slots -> 4h bars -> features -> scaled windows
            -> ensemble forward -> raw log-return quantiles -> calibrated quantiles

The engine is deliberately fail-closed. If anything is missing -- not enough history,
a bar that failed the validity gates, a NaN feature, a missing artifact -- it returns
NO prediction and says why. It never substitutes a stale window or an unfitted
transform.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .core import bar_builder, calendar_cme, features as featmod, slots as slotmod
from .core.registry import resolve_registry_pointer
from .data.mt5_source import detect_broker_utc_offset
from .logging_setup import get_logger
from .model.calibration import ACICalibrator, ACISymmetricWidthTracker, StaticCalibrator
from .model.runtime import LoadedModel, find_artifacts, load_model, load_provenance, predict_members
from .ops.logger import OperationalLogger
from .ops.realization_join import RealizationJoiner

log = get_logger("engine")


def _as_date(v) -> dt.date:
    """Normalize a trade_date (Timestamp, date, or numpy datetime) to a plain date."""
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    return pd.Timestamp(v).date()


@dataclass
class QuantilePrediction:
    """One prediction: the quantiles of the next 4-hour bar's log return."""
    # identity of the INPUT bar (the last bar the model saw)
    input_bar_start_utc: pd.Timestamp
    input_bar_end_utc: pd.Timestamp
    input_trade_date: dt.date
    input_slot: int
    input_close: float
    # the TARGET bar (the bar whose close-to-close return is forecast)
    target_bar_start_utc: Optional[pd.Timestamp]
    target_bar_end_utc: Optional[pd.Timestamp]
    target_trade_date: Optional[dt.date]
    target_slot: Optional[int]
    target_horizon_hours: float
    # the quantiles
    taus: List[float]
    quantiles: List[float]           # ensemble mean, calibrated, raw log-return units
    member_quantiles: np.ndarray     # (K, 7) calibrated, per member
    # provenance
    model: str
    fold: str
    calibrated: bool
    scale: float
    sigma_ewma: float
    slot_next: int
    dow_next: int
    generated_at_utc: pd.Timestamp
    # Health and operational monitoring fields
    forecast_valid: bool = True
    health_state: str = "GREEN"      # GREEN | YELLOW | RED | BLACK
    health_reasons: List[str] = field(default_factory=list)
    sizing_multiplier_cap: float = 1.0  # 1.0 (GREEN), 0.5 (YELLOW), 0.25 (RED), 0.0 (BLACK)
    model_version: str = "champion"
    calib_version: str = "static_cqr"
    ens_dispersion: List[float] = field(default_factory=list)  # std across K members per quantile

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "fold": self.fold,
            "calibrated": self.calibrated,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "input_bar": {
                "bar_start_utc": self.input_bar_start_utc.isoformat(),
                "bar_end_utc": self.input_bar_end_utc.isoformat(),
                "trade_date": str(self.input_trade_date),
                "slot": self.input_slot,
                "close": self.input_close,
            },
            "target": {
                "bar_start_utc": self.target_bar_start_utc.isoformat()
                if self.target_bar_start_utc is not None else None,
                "bar_end_utc": self.target_bar_end_utc.isoformat()
                if self.target_bar_end_utc is not None else None,
                "trade_date": str(self.target_trade_date) if self.target_trade_date else None,
                "slot": self.target_slot,
                "horizon_hours": self.target_horizon_hours,
            },
            "quantiles": {f"q{int(t * 100):02d}": float(q)
                          for t, q in zip(self.taus, self.quantiles)},
            "scale": self.scale,
            "sigma_ewma": self.sigma_ewma,
            "slot_next": self.slot_next,
            "dow_next": self.dow_next,
            "forecast_valid": self.forecast_valid,
            "health_state": self.health_state,
            "health_reasons": self.health_reasons,
            "sizing_multiplier_cap": self.sizing_multiplier_cap,
            "model_version": self.model_version,
            "calib_version": self.calib_version,
            "ens_dispersion": self.ens_dispersion,
        }


@dataclass
class EngineState:
    """Everything the engine carries between runs."""
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    seed_bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    minutes: pd.DataFrame = field(default_factory=pd.DataFrame)
    models: Dict[str, LoadedModel] = field(default_factory=dict)
    calibrators: Dict[str, object] = field(default_factory=dict)
    provenance: Dict[str, dict] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)
    # the CME closure / early-close sets in force for the current bar history
    calendar: dict = field(default_factory=lambda: {"closed": set(), "early": set()})
    detected_broker_offset: Optional[float] = None
    width_tracker: Optional[ACISymmetricWidthTracker] = None


class LiveEngine:
    """The live inference system."""

    def __init__(self, config: Config, state: Optional[EngineState] = None):
        self.config = config
        self.state = state or EngineState()
        self._artifacts: Optional[Dict[str, Path]] = None
        self.ops_logger = OperationalLogger()
        self.realization_joiner = RealizationJoiner()
        if self.state.width_tracker is None:
            self.state.width_tracker = ACISymmetricWidthTracker()

    # -- artifacts ----------------------------------------------------------
    def artifact_paths(self) -> Dict[str, Path]:
        if self._artifacts is None:
            # Check if pointer registry exists
            reg_ptr = resolve_registry_pointer(self.config.checkpoint_dir, role="active")
            if reg_ptr is not None and reg_ptr["checkpoint_path"].exists() and reg_ptr["provenance_path"].exists():
                arch = self.config.architectures[0] if self.config.architectures else "patchtst"
                self._artifacts = {
                    f"checkpoint:{arch}": reg_ptr["checkpoint_path"],
                    f"provenance:{arch}": reg_ptr["provenance_path"],
                    "_model_version": reg_ptr["model_version"],
                }
            else:
                self._artifacts = find_artifacts(
                    self.config.checkpoint_dir, self.config.fold, self.config.architectures)
        return self._artifacts

    def load_models(self) -> None:
        """Load every configured architecture and its frozen calibration."""
        paths = self.artifact_paths()
        m_version = paths.get("_model_version")
        for arch in self.config.architectures:
            ck = paths[f"checkpoint:{arch}"]
            pr = paths[f"provenance:{arch}"]
            model = load_model(ck, self.config.model)
            prov = load_provenance(pr)
            cal = self._build_calibrator(prov)
            self.state.models[arch] = model
            self.state.calibrators[arch] = cal
            self.state.provenance[arch] = prov
            self.state.model_versions[arch] = str(m_version or prov.get("fold", model.fold))
            log.info("model %s ready: fold=%s L=%d K=%d median_bias=%.3e cqr_offsets=%s",
                     arch, prov.get("fold", model.fold), model.lookback, model.K,
                     cal.static.median_bias, [round(v, 8) for v in cal.static.cqr_offsets])

    def _build_calibrator(self, provenance: dict):
        """Build the calibrator for the configured mode."""
        mode = self.config.calibration.mode
        cal_meta = provenance["calibration"]
        if mode == "static":
            return StaticCalibrator.from_metadata(cal_meta)
        if mode == "aci":
            alpha = self.config.calibration.aci_alpha
            hist = provenance.get("aci_alpha_after_observation")
            if hist:
                alpha = hist[-1]
            return ACICalibrator.from_metadata(
                cal_meta, aci_alpha=alpha, gamma=self.config.calibration.gamma,
                width_tracker=self.state.width_tracker)
        raise ValueError(f"unknown calibration mode: {mode!r}")

    # -- seeding and live updates -------------------------------------------
    def seed_history_from_bars(self, bars: pd.DataFrame) -> None:
        """Seed the 4-hour bar history with an existing clean bar dataset."""
        b = bars.copy()
        for c in ("bar_start_utc", "bar_end_utc"):
            b[c] = pd.to_datetime(b[c], utc=True)
        b["trade_date"] = pd.to_datetime(b["trade_date"]).dt.date
        b = b.sort_values("bar_start_utc", ignore_index=True)
        self.state.seed_bars = b
        self.state.bars = b
        self._refresh_calendar()

    def update_from_minutes(self, minutes: pd.DataFrame,
                            contract_id: str = "NDXUSD",
                            last_tick_epoch: Optional[float] = None,
                            system_utc_epoch: Optional[float] = None) -> bar_builder.BarBuildResult:
        """Append a new batch of 1-minute bars and recompute 4-hour bars."""
        # 1. DST / Broker UTC Offset Guard
        if last_tick_epoch is not None and system_utc_epoch is not None:
            detected = detect_broker_utc_offset(last_tick_epoch, system_utc_epoch)
            self.state.detected_broker_offset = detected

        m = minutes.copy()
        m["ts"] = pd.to_datetime(m["ts"], utc=True)
        m = m.sort_values("ts", ignore_index=True)
        if not self.state.minutes.empty:
            m = pd.concat([self.state.minutes, m], ignore_index=True).drop_duplicates(
                subset=["ts"], keep="last").sort_values("ts", ignore_index=True)
        self.state.minutes = m

        res = bar_builder.build_bars(m, self.config.slots, self.config.features)
        new_bars = res.bars
        if not self.state.seed_bars.empty:
            # Combine seed with new live bars
            seed = self.state.seed_bars
            cutoff = new_bars["bar_start_utc"].min() if not new_bars.empty else pd.Timestamp.max.tz_localize("UTC")
            kept_seed = seed.loc[seed["bar_end_utc"] <= cutoff].copy()
            comb = pd.concat([kept_seed, new_bars], ignore_index=True).drop_duplicates(
                subset=["bar_start_utc"], keep="last").sort_values("bar_start_utc", ignore_index=True)
            self.state.bars = comb
        else:
            self.state.bars = new_bars

        # Check if the most recent bar just closed and trigger realization join
        if not self.state.bars.empty:
            last_closed = self.state.bars.iloc[-1]
            try:
                self.realization_joiner.process_closed_bar(last_closed)
            except Exception as exc:
                log.warning("Realization join check encountered error: %s", exc)

        self._refresh_calendar()
        return res

    def _refresh_calendar(self) -> None:
        if self.state.bars.empty:
            return
        td = self.state.bars["trade_date"]
        y_min = int(min(d.year for d in td))
        y_max = int(max(d.year for d in td)) + 1
        cl, ea = calendar_cme.cme_calendar(y_min, y_max)
        self.state.calendar = {"closed": cl, "early": ea}

    def compute_features(self) -> pd.DataFrame:
        if self.state.bars.empty:
            return pd.DataFrame()
        return featmod.compute_features(self.state.bars, self.config.features)

    def _live_target(self, in_bar: pd.Series):
        st = self.config.slots
        w = slotmod.scheduled_successor(
            _as_date(in_bar["trade_date"]), int(in_bar["slot"]), st,
            closed=self.state.calendar["closed"],
            early=self.state.calendar["early"],
            max_gap_days=self.config.features.max_session_gap_days)
        if w is None:
            log.warning("no live target: no scheduled successor found")
            return None
        tgt_start = slotmod.bar_start_utc(w.trade_date, w.slot, st)
        tgt_end = slotmod.bar_end_utc(w.trade_date, w.slot, st)
        tgt_dummy = pd.Series({
            "bar_start_utc": tgt_start,
            "bar_end_utc": tgt_end,
            "trade_date": w.trade_date,
            "slot": w.slot,
        })
        dow = int(w.trade_date.weekday())
        return tgt_dummy, int(w.slot), dow

    def _is_just_closed(self, in_bar: pd.Series, now_utc: pd.Timestamp) -> bool:
        end = pd.Timestamp(in_bar["bar_end_utc"])
        if now_utc < end:
            log.warning("the last bar in history (%s) has not closed yet (now=%s)", end, now_utc)
            return False
        max_age = pd.Timedelta(hours=self.config.slots.bar_hours)
        if now_utc - end > max_age:
            log.warning("the last bar closed at %s, which is older than one bar period (%s)", end, now_utc - end)
            return False
        return True

    # -- inference ----------------------------------------------------------
    def predict(self, arch: str = "patchtst",
                decision_index: Optional[int] = None,
                live: bool = True,
                now_utc: Optional[pd.Timestamp] = None) -> Optional[QuantilePrediction]:
        """Produce one prediction."""
        if arch not in self.state.models:
            raise KeyError(f"model {arch!r} is not loaded; call load_models() first")
        model = self.state.models[arch]
        calibrator = self.state.calibrators[arch]
        model_ver = self.state.model_versions.get(arch, model.fold)

        fdf = self.compute_features()
        n = len(fdf)
        if n == 0:
            return None
        if decision_index is None:
            decision_index = n - 1
        if decision_index < 0:
            decision_index = n + decision_index
        t = int(decision_index)

        health_reasons = []
        health_state = "GREEN"
        forecast_valid = True

        # Check DST Clock Guard
        cfg_offset = float(self.config.mt5.server_utc_offset_hours)
        detected_offset = self.state.detected_broker_offset
        if detected_offset is not None and detected_offset != cfg_offset:
            forecast_valid = False
            health_state = "BLACK"
            health_reasons.append(
                f"Broker offset mismatch: configured UTC+{cfg_offset:g} vs detected UTC+{detected_offset:g}")

        # Check decision bar features
        decision_feats = fdf[featmod.FEATURE_ORDER].iloc[t].to_numpy(dtype=float)
        if not np.isfinite(decision_feats).all():
            forecast_valid = False
            health_state = "BLACK"
            health_reasons.append("Decision bar contains non-finite features.")

        # Timing check
        if now_utc is not None:
            ok = self._is_just_closed(self.state.bars.iloc[t], now_utc)
            if not ok:
                return None

        L = model.lookback
        need = self.effective_min_history()
        if t - L + 1 < 0:
            log.warning("history too short (have %d, need %d)", t + 1, need)
            return None

        window = fdf.iloc[t - L + 1: t + 1]
        win_feats = window[featmod.FEATURE_ORDER].to_numpy(dtype=float)
        if not np.isfinite(win_feats).all():
            forecast_valid = False
            health_state = "BLACK"
            health_reasons.append(f"The {L}-bar lookback window contains NaN/Inf values.")

        bars = self.state.bars
        in_bar = bars.iloc[t]

        # Target covariates
        if live and t == n - 1:
            target = self._live_target(in_bar)
            if target is None:
                return None
            tgt_bar, slot_next, dow_next = target
        else:
            if not bool(fdf["next_valid"].iloc[t]):
                log.warning("decision bar at %s has no valid successor", self.state.bars["bar_end_utc"].iloc[t])
                return None
            slot_next = int(fdf["slot_next"].iloc[t])
            dow_next = int(fdf["dow_next"].iloc[t])
            tgt_bar = bars.iloc[t + 1] if t + 1 < n else None

        # Build window
        raw = featmod.feature_matrix(fdf.iloc[t - L + 1: t + 1])
        x = model.transform.scale_features(raw)
        cats = np.array([slot_next, dow_next], dtype=np.int64)
        sigma = float(fdf["sigma_ewma"].iloc[t]) if np.isfinite(fdf["sigma_ewma"].iloc[t]) else 1e-4
        scale = float(model.transform.target_scale(np.array([sigma]), np.array([slot_next]))[0])

        if not forecast_valid:
            # Emit fail-closed prediction with BLACK health state
            sizing_cap = 0.0
            nan_quantiles = [float("nan")] * 7
            nan_members = np.full((model.K, 7), np.nan)
            ens_disp = [0.0] * 7
            pred = QuantilePrediction(
                input_bar_start_utc=pd.Timestamp(in_bar["bar_start_utc"]),
                input_bar_end_utc=pd.Timestamp(in_bar["bar_end_utc"]),
                input_trade_date=_as_date(in_bar["trade_date"]),
                input_slot=int(in_bar["slot"]),
                input_close=float(in_bar["close_mid"]),
                target_bar_start_utc=pd.Timestamp(tgt_bar["bar_start_utc"]) if tgt_bar is not None else None,
                target_bar_end_utc=pd.Timestamp(tgt_bar["bar_end_utc"]) if tgt_bar is not None else None,
                target_trade_date=_as_date(tgt_bar["trade_date"]) if tgt_bar is not None else None,
                target_slot=int(tgt_bar["slot"]) if tgt_bar is not None else slot_next,
                target_horizon_hours=float(self.config.slots.bar_hours),
                taus=list(model.taus),
                quantiles=nan_quantiles,
                member_quantiles=nan_members,
                model=arch,
                fold=model.fold,
                calibrated=False,
                scale=scale,
                sigma_ewma=sigma,
                slot_next=int(slot_next),
                dow_next=int(dow_next),
                generated_at_utc=pd.Timestamp.now(tz="UTC"),
                forecast_valid=False,
                health_state=health_state,
                health_reasons=health_reasons,
                sizing_multiplier_cap=sizing_cap,
                model_version=model_ver,
                calib_version=self.config.calibration.mode,
                ens_dispersion=ens_disp,
            )
            return pred

        members = predict_members(model, x[None, :, :], cats[None, :], np.array([scale], dtype=float))  # (K, 1, 7)
        cal = calibrator.apply(members)                                # (K, 1, 7)
        ens = cal.mean(axis=0)[0]                                      # (7,)
        ens_disp = np.std(cal[:, 0, :], axis=0).tolist()              # (7,)

        sizing_cap = 1.0 if health_state == "GREEN" else (0.5 if health_state == "YELLOW" else 0.25)

        pred = QuantilePrediction(
            input_bar_start_utc=pd.Timestamp(in_bar["bar_start_utc"]),
            input_bar_end_utc=pd.Timestamp(in_bar["bar_end_utc"]),
            input_trade_date=_as_date(in_bar["trade_date"]),
            input_slot=int(in_bar["slot"]),
            input_close=float(in_bar["close_mid"]),
            target_bar_start_utc=pd.Timestamp(tgt_bar["bar_start_utc"]) if tgt_bar is not None else None,
            target_bar_end_utc=pd.Timestamp(tgt_bar["bar_end_utc"]) if tgt_bar is not None else None,
            target_trade_date=_as_date(tgt_bar["trade_date"]) if tgt_bar is not None else None,
            target_slot=int(tgt_bar["slot"]) if tgt_bar is not None else slot_next,
            target_horizon_hours=float(self.config.slots.bar_hours),
            taus=list(model.taus),
            quantiles=[float(v) for v in ens],
            member_quantiles=cal[:, 0, :],
            model=arch,
            fold=model.fold,
            calibrated=True,
            scale=scale,
            sigma_ewma=sigma,
            slot_next=int(slot_next),
            dow_next=int(dow_next),
            generated_at_utc=pd.Timestamp.now(tz="UTC"),
            forecast_valid=forecast_valid,
            health_state=health_state,
            health_reasons=health_reasons,
            sizing_multiplier_cap=sizing_cap,
            model_version=model_ver,
            calib_version=self.config.calibration.mode,
            ens_dispersion=ens_disp,
        )

        # Operational logging: forecast_log.parquet and register pending forecast
        try:
            slot_fac = float(model.transform.factors[slot_next])
            raw_q = members[:, 0, :].mean(axis=0).tolist()
            aci_st = self.state.width_tracker.get_state() if self.state.width_tracker else {}
            log_row = {
                "generated_at_utc": pred.generated_at_utc.isoformat(),
                "broker_offset_detected": self.state.detected_broker_offset,
                "trade_date": str(pred.input_trade_date),
                "slot_current": pred.input_slot,
                "slot_next": pred.slot_next,
                "last_close": pred.input_close,
                "sigma_ewma": pred.sigma_ewma,
                "slot_factor": slot_fac,
                "scale": pred.scale,
                "q_raw": raw_q,
                "q_final": pred.quantiles,
                "ens_dispersion": pred.ens_dispersion,
                "aci_state": aci_st,
                "health_state": pred.health_state,
                "model_version": pred.model_version,
                "target_bar_start_utc": pred.target_bar_start_utc.isoformat() if pred.target_bar_start_utc else None,
                "target_trade_date": str(pred.target_trade_date) if pred.target_trade_date else None,
                "target_slot": pred.target_slot,
                "bar_end_utc": pred.input_bar_end_utc.isoformat(),
            }
            self.ops_logger.log_forecast(log_row)
            self.realization_joiner.register_forecast(log_row)
        except Exception as exc:
            log.warning("Operational logging encountered an issue: %s", exc)

        log.info("prediction: model=%s input_bar_end=%s target_slot=%s q50=%.6f q05=%.6f q95=%.6f (state=%s)",
                 arch, pred.input_bar_end_utc, pred.target_slot,
                 ens[3], ens[0], ens[6], pred.health_state)
        return pred

    def predict_all(self) -> Dict[str, QuantilePrediction]:
        out = {}
        for arch in self.config.architectures:
            p = self.predict(arch)
            if p is not None:
                out[arch] = p
        return out

    # -- warm-up ------------------------------------------------------------
    def effective_min_history(self) -> int:
        return self.config.model.lookback + self.config.warmup_offset

    def last_decision_index(self) -> Optional[int]:
        if self.state.bars.empty:
            return None
        fdf = self.compute_features()
        ok = fdf["next_valid"].to_numpy().astype(bool)
        idx = np.flatnonzero(ok)
        return int(idx[-1]) if idx.size else None
