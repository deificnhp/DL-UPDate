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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .core import bar_builder, calendar_cme, features as featmod, slots as slotmod
from .logging_setup import get_logger
from .model.calibration import ACICalibrator, StaticCalibrator
from .model.runtime import LoadedModel, find_artifacts, load_model, load_provenance, predict_members

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
        }


@dataclass
class EngineState:
    """Everything the engine carries between runs.

    `history_bars` is the accumulated 4-hour bar history. It is seeded from the
    training project's clean bar file (or a live bootstrap) and appended to on every
    run, so the recursive feature state (EWMA, EMA120, same-slot volume median)
    continues exactly as it would have in the training pipeline.
    """
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    seed_bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    minutes: pd.DataFrame = field(default_factory=pd.DataFrame)
    models: Dict[str, LoadedModel] = field(default_factory=dict)
    calibrators: Dict[str, object] = field(default_factory=dict)
    provenance: Dict[str, dict] = field(default_factory=dict)
    # the CME closure / early-close sets in force for the current bar history
    calendar: dict = field(default_factory=lambda: {"closed": set(), "early": set()})


class LiveEngine:
    """The live inference system."""

    def __init__(self, config: Config, state: Optional[EngineState] = None):
        self.config = config
        self.state = state or EngineState()
        self._artifacts: Optional[Dict[str, Path]] = None

    # -- artifacts ----------------------------------------------------------
    def artifact_paths(self) -> Dict[str, Path]:
        if self._artifacts is None:
            self._artifacts = find_artifacts(
                self.config.checkpoint_dir, self.config.fold, self.config.architectures)
        return self._artifacts

    def load_models(self) -> None:
        """Load every configured architecture and its frozen calibration."""
        paths = self.artifact_paths()
        for arch in self.config.architectures:
            ck = paths[f"checkpoint:{arch}"]
            pr = paths[f"provenance:{arch}"]
            model = load_model(ck, self.config.model)
            prov = load_provenance(pr)
            cal = self._build_calibrator(prov)
            self.state.models[arch] = model
            self.state.calibrators[arch] = cal
            self.state.provenance[arch] = prov
            log.info("model %s ready: fold=%s L=%d K=%d median_bias=%.3e cqr_offsets=%s",
                     arch, prov.get("fold", model.fold), model.lookback, model.K,
                     cal.static.median_bias, [round(v, 8) for v in cal.static.cqr_offsets])

    def _build_calibrator(self, provenance: dict):
        """Build the calibrator for the configured mode.

        "static" (the default) is the EXACT reproduction of the training project's
        offline path: the frozen median bias plus the three frozen CQR offsets.

        "aci" additionally starts from the fold's persisted ACI alpha state and
        advances it with realized outcomes. The alpha vector is seeded from the last
        entry of `aci_alpha_after_observation` when it is present, otherwise from the
        nominal alphas in the configuration.
        """
        mode = self.config.calibration.mode
        cal_meta = provenance["calibration"]
        if mode == "static":
            return StaticCalibrator.from_metadata(cal_meta)
        if mode == "aci":
            alpha = self.config.calibration.aci_alpha
            hist = provenance.get("aci_alpha_after_observation")
            if hist:
                alpha = tuple(float(v) for v in hist[-1])
                log.info("ACI alpha seeded from the persisted fold state: %s",
                         [round(v, 4) for v in alpha])
            return ACICalibrator.from_metadata(
                cal_meta, aci_alpha=alpha, gamma=self.config.calibration.gamma)
        raise ValueError(f"unknown calibration mode {mode!r} (use 'static' or 'aci')")

    # -- data ---------------------------------------------------------------
    def seed_history_from_bars(self, bars: pd.DataFrame) -> None:
        """Seed the bar history with an existing frame of 4-hour bars.

        Used to warm the recursive feature state with the trained model's own history.
        """
        if bars.empty:
            raise ValueError("cannot seed from an empty bar frame")
        b = bars.copy()
        for c in ("bar_start_utc", "bar_end_utc"):
            b[c] = pd.to_datetime(b[c], utc=True)
        b["trade_date"] = pd.to_datetime(b["trade_date"]).dt.date
        b = b.sort_values("bar_start_utc", ignore_index=True)
        # keep the pristine seed rows: the volume alignment is re-estimated from them
        # on every run, so it is never applied twice to the same rows.
        self.state.seed_bars = b.copy()
        if not self.state.bars.empty:
            b = (pd.concat([self.state.bars, b], ignore_index=True)
                 .drop_duplicates("bar_start_utc", keep="last")
                 .sort_values("bar_start_utc", ignore_index=True))
        self.state.bars = b
        # make sure the CME calendar covers the seeded range, so a live decision can
        # resolve the scheduled successor of the last bar
        y0 = int(pd.Timestamp(b["trade_date"].min()).year)
        y1 = int(pd.Timestamp(b["trade_date"].max()).year)
        closed, early = calendar_cme.cme_calendar(y0, y1)
        self.state.calendar = {"closed": set(closed), "early": set(early)}
        log.info("history seeded: %d bars (%s .. %s)", len(b),
                 b["bar_start_utc"].iloc[0], b["bar_start_utc"].iloc[-1])

    # -- seed / live seam ----------------------------------------------------
    def _volume_scale_factors(self, seed_bars: pd.DataFrame,
                              live_bars: pd.DataFrame) -> Dict[int, float]:
        """Per-slot factors k with `live_volume * k ~= seed_volume`.

        The seed's `volume` is the vendor's contract-volume share; the live feed's
        `volume` is the broker's tick count. The two are on completely different
        scales (measured: ~3 orders of magnitude apart), and `rvol` is
        log(volume / median of the previous 20 same-slot volumes). While that median
        is still vendor-derived and the numerator is broker-derived, `rvol` pins at
        the +/-5 clip -- an out-of-distribution input produced purely by the
        handover.

        The factor is estimated from the bars the two frames SHARE (the same
        bar_start_utc): for those bars the ratio is pure scale, with no session or
        regime effect. A shared bar_start_utc also requires the clock to be right --
        a mis-set server offset would leave the two frames with no bars in common.
        """
        s = seed_bars[["bar_start_utc", "slot", "volume"]].copy()
        l = live_bars[["bar_start_utc", "slot", "volume"]].copy()
        m = s.merge(l, on="bar_start_utc", how="inner", suffixes=("_s", "_l"))
        if m.empty:
            log.warning("volume alignment skipped: the seed and the live feed share no "
                        "bar_start_utc, so the volume scale cannot be estimated")
            return {}
        slot_mismatch = int((m["slot_s"] != m["slot_l"]).sum())
        if slot_mismatch:
            log.warning("volume alignment: %d shared bars disagree on the slot; the "
                        "clock may still be mis-set", slot_mismatch)
        m = m[(m["volume_l"] > 0) & (m["volume_s"] > 0)]
        if m.empty:
            log.warning("volume alignment skipped: the shared bars have no volume")
            return {}
        ratio = m["volume_s"].to_numpy() / m["volume_l"].to_numpy()
        glob = float(np.median(ratio))
        min_ov = self.config.volume_align_min_overlap
        factors: Dict[int, float] = {}
        for slot in sorted(set(m["slot_l"].astype(int))):
            sel = m["slot_l"].astype(int).to_numpy() == slot
            if int(sel.sum()) >= min_ov:
                factors[slot] = float(np.median(ratio[sel]))
            else:
                factors[slot] = glob
        log.info("volume alignment: %d shared bars with the seed; per-slot factors "
                 "seed/live = %s (global %.6g)",
                 len(m), {k: round(v, 8) for k, v in sorted(factors.items())}, glob)
        return factors

    def _align_volume_scale(self, live_bars: pd.DataFrame) -> pd.DataFrame:
        """Rescale the live bars' volume into the seed's units (see the method above)."""
        if not self.config.align_volume_scale:
            return live_bars
        seed = self.state.seed_bars
        if seed.empty or live_bars.empty:
            return live_bars
        factors = self._volume_scale_factors(seed, live_bars)
        if not factors:
            return live_bars
        out = live_bars.copy()
        k = out["slot"].astype(int).map(factors)
        out["volume"] = out["volume"].to_numpy() * k.to_numpy()
        med_before = float(np.median(live_bars["volume"]))
        med_after = float(np.median(out["volume"]))
        log.info("volume alignment applied: live median volume %.6g -> %.6g "
                 "(now in the seed's units)", med_before, med_after)
        return out

    def update_from_minutes(self, minutes: pd.DataFrame,
                            contract_id: str = "NQ_LIVE") -> pd.DataFrame:
        """Build bars from minutes and merge them into the history.

        Only bars whose window has CLOSED are admitted: the live policy accepts a
        slot-0 bar from 2 genuine minutes on, so a session window that is still
        forming would otherwise enter the history as a bar. The cutoff is derived
        from the data itself (the newest minute in the frame), so the rule is
        deterministic and replay-safe.

        Returns the merged bar history.
        """
        if not self.state.models:
            self.load_models()
        res = bar_builder.build_bars(minutes, self.config.slots, self.config.features,
                                     contract_id=contract_id)
        bars = res.bars
        if not bars.empty:
            cutoff = pd.Timestamp(minutes["ts"].max()) + pd.Timedelta(minutes=1)
            unclosed = bars["bar_end_utc"] > cutoff
            if unclosed.any():
                log.info("dropped %d bar(s) that have not closed yet (newest data %s): %s",
                         int(unclosed.sum()), cutoff,
                         [str(b) for b in bars.loc[unclosed, "bar_start_utc"]])
                bars = bars.loc[~unclosed]
        bar_builder.validate_bars(bars, self.config.slots, self.config.features)
        self.state.minutes = minutes
        self.state.calendar = {"closed": set(res.calendar.get("closed", [])),
                               "early": set(res.calendar.get("early", []))}
        if bars.empty:
            log.warning("no bar passed the validity gates in this update")
            return self.state.bars
        live = self._align_volume_scale(bars)
        if self.state.bars.empty:
            self.state.bars = live.copy()
        else:
            self.state.bars = (pd.concat([self.state.bars, live], ignore_index=True)
                                .drop_duplicates("bar_start_utc", keep="last")
                                .sort_values("bar_start_utc", ignore_index=True))
        log.info("history now %d bars (last end %s)", len(self.state.bars),
                 self.state.bars["bar_end_utc"].iloc[-1])
        return self.state.bars

    # -- features -----------------------------------------------------------
    def compute_features(self) -> pd.DataFrame:
        """Features for the whole bar history (strictly trailing, so the last row is
        the freshest possible state)."""
        if self.state.bars.empty:
            raise RuntimeError("no bar history; seed it or update from minutes first")
        return featmod.compute_features(self.state.bars, self.config.features)

    # -- target resolution --------------------------------------------------
    def _is_just_closed(self, in_bar: pd.Series, now_utc: pd.Timestamp) -> bool:
        """Check that `in_bar` is the bar that has JUST closed, per the session grid.

        Two conditions:
          1. the bar's scheduled end time is the most recent scheduled end <= now
             (so the bar is the most recent one to have closed);
          2. it closed no more than one bar period ago (so a stale bar is not
             silently re-predicted).

        Returns False and logs the reason when the check fails.
        """
        now_utc = pd.Timestamp(now_utc)
        bar_end = pd.Timestamp(in_bar["bar_end_utc"])
        if bar_end > now_utc:
            log.warning("no prediction: the bar ending %s has not closed yet (now %s)",
                        bar_end, now_utc)
            return False
        w = slotmod.bar_just_closed(now_utc, self.config.slots,
                                     max_lag_hours=self.config.slots.bar_hours)
        if w is None:
            log.warning("no prediction: no bar has closed within the last %d hours (now %s)",
                        self.config.slots.bar_hours, now_utc)
            return False
        if pd.Timestamp(w.bar_end_utc) != bar_end:
            log.warning("no prediction: the last bar to close was %s, but the history's "
                        "last bar ends %s (now %s)", w.bar_end_utc, bar_end, now_utc)
            return False
        return True


    def _live_target(self, in_bar: pd.Series):
        """Resolve the SCHEDULED target bar for a live decision.

        Returns (target_row_like, slot_next, dow_next) or None.

        The target bar has not closed yet, so its calendar covariates are taken from
        the session calendar. The rules are the training project's contiguity rules
        (see `core.slots.scheduled_successor`), which is what makes slot_next/dow_next
        exactly the covariates the model was trained on.
        """
        closed, early = self.state.calendar.get("closed", set()), self.state.calendar.get("early", set())
        win = slotmod.scheduled_successor(
            in_bar["trade_date"], int(in_bar["slot"]), self.config.slots,
            closed=closed, early=early,
            max_gap_days=self.config.features.max_session_gap_days)
        if win is None:
            log.warning("no prediction: no scheduled successor bar for %s slot %s",
                        in_bar["trade_date"], in_bar["slot"])
            return None
        dow = (pd.Timestamp(win.trade_date).weekday())
        row = {
            "bar_start_utc": win.bar_start_utc,
            "bar_end_utc": win.bar_end_utc,
            "trade_date": win.trade_date,
            "slot": win.slot,
        }
        log.info("live target: input slot %s -> target slot %s (trade_date %s, dow %d)",
                 in_bar["slot"], win.slot, win.trade_date, dow)
        return row, int(win.slot), int(dow)

    # -- prediction ---------------------------------------------------------
    def predict(self, arch: str = "patchtst",
                decision_index: Optional[int] = None,
                live: bool = True,
                now_utc: Optional[pd.Timestamp] = None) -> Optional[QuantilePrediction]:
        """Produce one prediction.

        `decision_index` is the position in the feature frame of the bar to predict
        FROM (the last input bar). Default: the most recent bar.

        `live=True` (the default) treats the decision bar as the LAST bar in the
        history: its target is the next SCHEDULED session bar, which has not closed
        yet, so the target covariates (slot_next / dow_next) are taken from the
        session calendar rather than from a materialized successor row. This is the
        only mode that can serve a live feed; `live=False` reproduces the offline
        behaviour of predicting from a bar whose successor is already in the history
        (used by the replay/verification path).

        `now_utc` is the current time, used to enforce the "a full valid bar has just
        closed" rule: the decision bar must be the bar that the session grid says
        closed most recently, and it must have closed no more than one bar period
        ago. Pass None to skip that check (offline replay).
        """
        if arch not in self.state.models:
            raise KeyError(f"model {arch!r} is not loaded; call load_models() first")
        model = self.state.models[arch]
        calibrator = self.state.calibrators[arch]

        fdf = self.compute_features()
        n = len(fdf)
        if decision_index is None:
            decision_index = n - 1
        if decision_index < 0:
            decision_index = n + decision_index
        t = int(decision_index)

        if not np.isfinite(fdf[featmod.FEATURE_ORDER].iloc[t].to_numpy(dtype=float)).all():
            log.warning("no prediction: the decision bar at %s has non-finite features",
                        self.state.bars["bar_end_utc"].iloc[t])
            return None

        # --- the decision bar must be the bar that JUST CLOSED --------------------
        if now_utc is not None:
            ok = self._is_just_closed(self.state.bars.iloc[t], now_utc)
            if not ok:
                return None

        L = model.lookback
        need = self.effective_min_history()
        if t - L + 1 < 0:
            log.warning("no prediction: the history is too short. The model needs a "
                        "%d-bar window (L=%d) AND %d rows of warm-up before it, so at "
                        "least %d bars of history are required -- not merely %d. "
                        "Have %d. Seed the history with the shipped artifact "
                        "(CKPTs/history_seed.parquet, %d bars) to reach it.",
                        L, L, self.config.warmup_offset, need, L, t + 1, need)
            return None

        # the WHOLE window must be finite, not just the decision row. A short history
        # (or a short seed) leaves the early rows of the window without the long
        # trailing features (lrv_m needs 120 bars, rvol needs 20 same-slot sessions),
        # and a NaN anywhere in the window would silently poison the forecast.
        window = fdf.iloc[t - L + 1: t + 1]
        win_feats = window[featmod.FEATURE_ORDER].to_numpy(dtype=float)
        if not np.isfinite(win_feats).all():
            n_bad = int((~np.isfinite(win_feats)).sum())
            first_bad = int(np.flatnonzero(~np.isfinite(win_feats).all(axis=1))[0])
            log.warning("no prediction: the %d-bar window contains %d non-finite feature "
                        "values (first at offset %d of the window). The long trailing "
                        "features (EMA-120 on the price index, the 120-bar trailing "
                        "variance mean) are not finite until row %d of the history, so "
                        "the effective minimum is %d bars (L=%d plus %d rows of warm-up), "
                        "not %d. Have %d. Seed the history with the shipped artifact "
                        "(CKPTs/history_seed.parquet, %d bars) to reach it.",
                        L, n_bad, first_bad, self.config.warmup_offset, need, L,
                        self.config.warmup_offset, L, len(fdf), need)
            return None

        bars = self.state.bars
        in_bar = bars.iloc[t]

        # --- resolve the target bar and its calendar covariates ---------------------
        if live and t == n - 1:
            target = self._live_target(in_bar)
            if target is None:
                return None
            tgt_bar, slot_next, dow_next = target
        else:
            if not bool(fdf["next_valid"].iloc[t]):
                log.warning("no prediction: the decision bar at %s has no valid successor",
                            self.state.bars["bar_end_utc"].iloc[t])
                return None
            slot_next = int(fdf["slot_next"].iloc[t])
            dow_next = int(fdf["dow_next"].iloc[t])
            tgt_bar = bars.iloc[t + 1] if t + 1 < n else None

        # Build the window: features[t-L+1 .. t], scaled with the FROZEN transform.
        raw = featmod.feature_matrix(fdf.iloc[t - L + 1: t + 1])
        x = model.transform.scale_features(raw)

        cats = np.array([slot_next, dow_next], dtype=np.int64)
        sigma = float(fdf["sigma_ewma"].iloc[t])
        scale = float(model.transform.target_scale(
            np.array([sigma]), np.array([slot_next]))[0])

        members = predict_members(model, x[None, :, :], cats[None, :],
                                  np.array([scale], dtype=float))     # (K, 1, 7)
        cal = calibrator.apply(members)                                # (K, 1, 7)
        ens = cal.mean(axis=0)[0]                                      # (7,)

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
        )
        log.info("prediction: model=%s input_bar_end=%s target_slot=%s q50=%.6f q05=%.6f q95=%.6f",
                 arch, pred.input_bar_end_utc, pred.target_slot,
                 ens[3], ens[0], ens[6])
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
        """The number of bars the history must contain for a valid decision.

        The model consumes L bars, but EVERY row of that window must have all 11
        features finite, and the long trailing features (the EMA-120 on the price
        index, `lrv_m`'s 120-bar trailing variance mean) are not finite until row
        `warmup_offset` of the history. The effective minimum is therefore
        L + warmup_offset = 419 bars, and the runtime says so.
        """
        return self.config.model.lookback + self.config.warmup_offset

    def last_decision_index(self) -> Optional[int]:
        """Index of the most recent bar that can serve as a decision.

        A bar can serve as a decision if it has a successor in the history (the target
        bar's slot/dow covariates come from the successor) -- the target's *return* is
        not needed, only its schedule.
        """
        if self.state.bars.empty:
            return None
        fdf = self.compute_features()
        ok = fdf["next_valid"].to_numpy().astype(bool)
        idx = np.flatnonzero(ok)
        return int(idx[-1]) if idx.size else None
