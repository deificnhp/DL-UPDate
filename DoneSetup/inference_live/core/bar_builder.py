"""1-minute -> 4-hour bar builder.

This is the live system's copy of DATA/build_nq_4h.py (training project). It reproduces
the aggregation and the VALIDITY GATES exactly, because the trained model's features
were computed over bars that passed these gates, and because the gaps left by rejected
bars are part of the input distribution the model saw.

Pipeline (training order):
  minutes (bid/ask or mid/last)
    -> spread cap (causal, on the bid-ask spread; skipped for single-sided feeds)
    -> synthetic-minute filter (zero volume, or flat-and-identical-to-previous on both sides)
    -> slot assignment (ET wall clock, 5 slots; the 14:00-18:00 block is dropped)
    -> calendar grid (weekday trade dates x 5 slots)
    -> aggregation over GENUINE minutes only
    -> validity gates -> accepted bars / rejected bars with a reason
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import FeatureConfig, SlotTable
from ..logging_setup import get_logger
from . import calendar_cme
from .slots import assign_slots, build_grid

log = get_logger("core.bar_builder")

# Rejection reasons, in the training project's order.
REASON_CLOSURE = "closure"
REASON_ZERO_VOLUME = "zero_volume_synthetic"
REASON_STALE = "stale_market_closure"
REASON_LOW_COVERAGE = "low_coverage"
REASON_MISSING_RUN = "missing_run_exceeded"

CLEAN_COLUMNS = [
    "interval_id", "bar_start_utc", "bar_end_utc", "trade_date", "calendar_date", "slot",
    "contract_id", "open_mid", "high_mid", "low_mid", "close_mid",
    "spread_open", "spread_mean", "spread_max", "spread_close", "volume",
    "expected_minutes", "observed_minutes", "quote_coverage",
    "active_seconds", "wall_seconds",
    "is_terminal_3h", "is_partial_calendar", "is_early_close", "is_holiday_session",
    "is_roll_session", "is_post_maintenance", "is_post_weekend", "is_post_holiday",
]


@dataclass
class BarBuilderResult:
    bars: pd.DataFrame            # accepted bars, chronologically sorted
    rejected: pd.DataFrame        # rejected grid bars with a reason
    minutes: pd.DataFrame         # the minute frame after slot assignment (for diagnostics)
    calendar: dict                # the closed/early sets used


# ---------------------------------------------------------------------------
def mark_synthetic(minutes: pd.DataFrame,
                   bid_cols: Optional[list[str]] = None,
                   ask_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Flag synthetic (non-tradable) minutes.

    Training rule: a minute is synthetic if its volume is <= 0, OR it is flat on both
    sides (O=H=L=C) AND identical to the previous minute on both sides.

    For a single-sided feed (MT5 mid/last), the two-sided condition degenerates to
    "flat AND identical to the previous minute", which is the closest available
    analogue and is applied to the mid OHLC.
    """
    m = minutes.copy()
    if bid_cols and ask_cols and all(c in m.columns for c in bid_cols + ask_cols):
        flat = (m["bid_open"].eq(m["bid_close"]) & m["bid_high"].eq(m["bid_low"])
                & m["bid_open"].eq(m["bid_high"])
                & m["ask_open"].eq(m["ask_close"]) & m["ask_high"].eq(m["ask_low"])
                & m["ask_open"].eq(m["ask_high"]))
        prev_same = (m["bid_close"].eq(m["bid_close"].shift())
                     & m["ask_close"].eq(m["ask_close"].shift()))
    else:
        # single-sided feed: apply the rule to the mid OHLC
        pfx = "" if {"open", "high", "low", "close"} <= set(m.columns) else "mid_"
        o, h, l, c = (m[f"{pfx}{k}"] for k in ("open", "high", "low", "close"))
        flat = o.eq(c) & h.eq(l) & o.eq(h)
        prev_same = c.eq(c.shift())
    m["is_synthetic"] = m["volume"].le(0) | (flat & prev_same)
    m["genuine"] = ~m["is_synthetic"]
    return m


def apply_spread_cap(minutes: pd.DataFrame,
                     tick: float = 0.25,
                     k: float = 5.0,
                     hard_cap: float = 25.0,
                     window: int = 7200,
                     min_periods: int = 30,
                     min_ticks: int = 4) -> pd.DataFrame:
    """Causal spread cap, copied from the training pipeline.

    The cap is the rolling (median + 5*IQR) of the spread, computed on GENUINE minutes
    and shifted by one minute so it is strictly past. The floor is median + 4 ticks and
    the ceiling is an absolute 25 points. The raw spread is clipped into [0, cap]; the
    bar is never dropped for a wide spread.

    For a single-sided feed there is no spread, so this is a no-op.
    """
    m = minutes.copy()
    if "spread_raw" not in m.columns:
        m["spread_cap"] = hard_cap
        m["spread"] = 0.0
        return m
    g = m.loc[m["genuine"], "spread_raw"]
    roll = g.rolling(window, min_periods=min_periods)
    med, q1, q3 = (roll.quantile(p).shift(1) for p in (0.5, 0.25, 0.75))
    cap = (med + k * (q3 - q1)).clip(lower=med + min_ticks * tick).clip(upper=hard_cap)
    m["spread_cap"] = cap.reindex(m.index).ffill().fillna(hard_cap)
    m["spread"] = m["spread_raw"].clip(lower=0.0, upper=m["spread_cap"])
    return m


# ---------------------------------------------------------------------------
def aggregate(grid: pd.DataFrame, minutes: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Aggregate genuine minutes onto the session grid.

    Mirrors the training `aggregate`:
      open_mid  = first genuine minute's mid open
      high_mid  = max genuine mid high
      low_mid   = min genuine mid low
      close_mid = last genuine mid close
      volume    = sum of genuine-minute volume
      observed_minutes = number of genuine minutes
      max_missing_run_minutes = longest run of missing minutes inside [start, end)
    """
    key = ["trade_date", "slot"]
    if grid.empty:
        return grid.copy()

    gm = minutes.loc[minutes["genuine"]].sort_values("ts")
    grp = gm.groupby(key, sort=False)
    agg = grp.agg(open_mid=("mid_open", "first"), high_mid=("mid_high", "max"),
                  low_mid=("mid_low", "min"), close_mid=("mid_close", "last"),
                  spread_open=("spread", "first"), spread_mean=("spread", "mean"),
                  spread_max=("spread", "max"), spread_close=("spread", "last"),
                  volume=("volume", "sum"), observed_minutes=("ts", "size"),
                  first_ts=("ts", "first"), last_ts=("ts", "last"))

    # Longest run of missing minutes inside [bar_start, bar_end).
    gm2 = gm.merge(grid[key + ["bar_start_utc"]], on=key, how="left")
    off = ((gm2["ts"] - gm2["bar_start_utc"]).dt.total_seconds() // 60).astype("int64")
    gm2["off"] = off
    gm2["gap"] = gm2.groupby(key)["off"].diff().fillna(0) - 1
    inner = gm2.groupby(key)["gap"].max().clip(lower=0)
    lead = gm2.groupby(key)["off"].min()
    trail = (cfg.expected_minutes - 1) - gm2.groupby(key)["off"].max()
    agg["max_missing_run_minutes"] = pd.concat([inner, lead, trail], axis=1).max(axis=1).astype("int64")

    raw_cnt = minutes.groupby(key).size().rename("raw_minutes")
    raw_vol = minutes.groupby(key)["volume"].sum().rename("raw_volume")

    out = (grid.merge(agg.reset_index(), on=key, how="left")
               .merge(raw_cnt.reset_index(), on=key, how="left")
               .merge(raw_vol.reset_index(), on=key, how="left"))
    out["observed_minutes"] = out["observed_minutes"].fillna(0).astype("int64")
    out["raw_minutes"] = out["raw_minutes"].fillna(0).astype("int64")
    out["volume"] = out["volume"].fillna(0.0)
    out["max_missing_run_minutes"] = out["max_missing_run_minutes"].fillna(
        cfg.expected_minutes).astype("int64")
    out["quote_coverage"] = out["observed_minutes"] / cfg.expected_minutes
    return out


def gate_thresholds(bars: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """The per-slot validity thresholds that apply to `bars`.

    Two policies:

      * the TRAINING policy (the default of a bare `FeatureConfig`) is exactly the
        one DATA/build_nq_4h.py used: 99% coverage and no more than one missing
        minute in a row. It is what the model's features were computed over.

      * the LIVE policy (`FeatureConfig.live()`) relaxes those gates for broker
        feeds, whose quote density is lower than the vendor's:
            - slots 1..4: coverage >= 0.90 (216 of 240 minutes), longest missing
              run <= 60 minutes
            - slot 0    : the coverage gate and the missing-run gate are disabled;
              a bar is accepted from 2 genuine minutes on

    The first three gates (closure, zero volume, flat/stale) are UNCHANGED in both
    policies: a bar with no minutes at all, no volume, or no range is still
    rejected. Only the "how complete must this session be" gates are relaxed.
    """
    is_slot0 = bars["slot"].eq(0).to_numpy()
    if cfg.use_live_gates:
        min_cov = np.where(is_slot0, cfg.live_slot0_min_coverage, cfg.live_min_coverage)
        max_run = np.where(is_slot0, cfg.live_slot0_max_missing_run, cfg.live_max_missing_run)
        min_obs = np.where(is_slot0, cfg.live_slot0_min_observed_minutes, 0)
    else:
        min_cov = np.full(len(bars), cfg.min_coverage, dtype=float)
        max_run = np.full(len(bars), cfg.max_missing_run, dtype=float)
        min_obs = np.zeros(len(bars), dtype=float)
    return pd.DataFrame({"min_coverage": min_cov, "max_missing_run": max_run,
                          "min_observed_minutes": min_obs}, index=bars.index)


def classify(bars: pd.DataFrame, cfg: FeatureConfig) -> pd.Series:
    """The validity gates, in the training project's order.

    Training order matters: the first matching condition wins, so the reason labels
    are comparable with the shipped exclusions file. The coverage and missing-run
    gates use the per-slot thresholds from `gate_thresholds`.
    """
    flat = (bars["high_mid"].eq(bars["low_mid"])
            | (bars["open_mid"].eq(bars["close_mid"]) & bars["high_mid"].eq(bars["open_mid"])))
    thr = gate_thresholds(bars, cfg)
    too_thin = (bars["quote_coverage"].lt(thr["min_coverage"])
                | bars["observed_minutes"].lt(thr["min_observed_minutes"]))
    conds = [
        bars["raw_minutes"].eq(0) | (bars["is_closure_day"] & bars["observed_minutes"].eq(0)),
        bars["observed_minutes"].eq(0) | bars["volume"].le(0),
        flat.fillna(True),
        too_thin,
        bars["max_missing_run_minutes"].gt(thr["max_missing_run"]),
    ]
    reasons = [REASON_CLOSURE, REASON_ZERO_VOLUME, REASON_STALE,
               REASON_LOW_COVERAGE, REASON_MISSING_RUN]
    return pd.Series(np.select(conds, reasons, default=""), index=bars.index)


def enforce_spread_invariants(b: pd.DataFrame) -> pd.DataFrame:
    """Clamp (never drop) so that max >= mean >= min(open, close) >= 0."""
    b = b.copy()
    for c in ("spread_open", "spread_close", "spread_mean", "spread_max"):
        b[c] = b[c].clip(lower=0.0)
    b["spread_max"] = np.maximum(b["spread_max"], b["spread_mean"])
    viol = np.minimum(b["spread_open"], b["spread_close"]) > b["spread_mean"]
    o_is_min = b["spread_open"] <= b["spread_close"]
    b.loc[viol & o_is_min, "spread_open"] = b.loc[viol & o_is_min, "spread_mean"]
    b.loc[viol & ~o_is_min, "spread_close"] = b.loc[viol & ~o_is_min, "spread_mean"]
    return b


# ---------------------------------------------------------------------------
def build_bars(minutes: pd.DataFrame,
               slot_table: SlotTable,
               cfg: FeatureConfig,
               contract_id: str = "NQ_LIVE",
               version: str = "1.0.0-adapted",
               use_spread_cap: bool = True) -> BarBuilderResult:
    """Build the accepted 4h bars from a frame of 1-minute rows.

    `minutes` must contain: ts (UTC datetime64) and volume, plus either
    (bid_*, ask_*) pairs, or the base open/high/low/close columns, or pre-computed
    mid_* columns.
    """
    if minutes.empty:
        raise ValueError("no minutes supplied")
    if "ts" not in minutes.columns:
        raise ValueError("minute frame is missing column 'ts'")
    if "volume" not in minutes.columns:
        raise ValueError("minute frame is missing column 'volume'")
    has_two_sided = {"bid_open", "ask_open"} <= set(minutes.columns)
    has_base = {"open", "high", "low", "close"} <= set(minutes.columns)
    if not (has_two_sided or has_base or "mid_open" in minutes.columns):
        raise ValueError(
            "minute frame must carry either (bid_*, ask_*) pairs, the base "
            "open/high/low/close columns, or pre-computed mid_* columns")

    m = minutes.copy()
    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    m = m.sort_values("ts", ignore_index=True)
    m = m.drop_duplicates("ts", keep="last").reset_index(drop=True)

    # --- mid prices ---------------------------------------------------------
    if "mid_open" not in m.columns:
        if {"bid_open", "ask_open"} <= set(m.columns):
            # crossed book -> corrupted, discarded before aggregation (training parity)
            crossed = (m["ask_close"] < m["bid_close"]) | (m["ask_open"] < m["bid_open"])
            n_crossed = int(crossed.sum())
            if n_crossed:
                log.info("crossed minutes discarded: %d", n_crossed)
                m = m.loc[~crossed].reset_index(drop=True)
            m["is_locked"] = m["ask_close"].eq(m["bid_close"])
            for f in ("open", "high", "low", "close"):
                m[f"mid_{f}"] = (m[f"bid_{f}"] + m[f"ask_{f}"]) / 2.0
            m["spread_raw"] = m["ask_close"] - m["bid_close"]
            m["spread_open_raw"] = m["ask_open"] - m["bid_open"]
        else:
            for f in ("open", "high", "low", "close"):
                m[f"mid_{f}"] = m[f].astype("float64")
    m = mark_synthetic(m, bid_cols=[f"bid_{f}" for f in ("open", "high", "low", "close")],
                       ask_cols=[f"ask_{f}" for f in ("open", "high", "low", "close")])
    if use_spread_cap:
        m = apply_spread_cap(m)

    # --- slot assignment ----------------------------------------------------
    m = assign_slots(m, slot_table)

    # --- calendar grid ------------------------------------------------------
    if m.empty:
        log.warning("no minute fell inside the trading session; no bar can be built")
        empty = pd.DataFrame(columns=CLEAN_COLUMNS)
        return BarBuilderResult(bars=empty, rejected=pd.DataFrame(), minutes=m,
                                 calendar={"closed": [], "early": []})
    y0 = int(m["trade_date"].min().year)
    y1 = int(m["trade_date"].max().year)
    closed, early = calendar_cme.cme_calendar(y0, y1)
    et_naive = m["ts"].dt.tz_convert(slot_table.timezone).dt.tz_localize(None)
    from .slots import trade_date_for_et
    tds = trade_date_for_et(et_naive, slot_table)
    lo = tds.min() - pd.Timedelta(days=2)
    hi = tds.max() + pd.Timedelta(days=2)
    dates = []
    d = lo
    while d <= hi:
        dates.append(d)
        d = d + pd.Timedelta(days=1)
    grid = build_grid(dates, slot_table, closed, early)
    if grid.empty:
        raise ValueError("empty session grid")

    # --- aggregation --------------------------------------------------------
    b = aggregate(grid, m, cfg)
    b = enforce_spread_invariants(b)
    b["rejection_reason"] = classify(b, cfg)

    b["interval_id"] = (b["trade_date"].dt.strftime("%Y-%m-%d") + "_S"
                        + b["slot"].astype(str) + "_" + version)
    b["contract_id"] = contract_id
    b["expected_minutes"] = cfg.expected_minutes
    b["wall_seconds"] = ((b["bar_end_utc"] - b["bar_start_utc"]).dt.total_seconds()).astype("int64")
    b["active_seconds"] = cfg.expected_minutes * 60
    b["is_terminal_3h"] = False
    b["is_roll_session"] = False
    b["is_post_maintenance"] = b["slot"].eq(0)
    b["is_post_weekend"] = b["slot"].eq(0) & b["trade_date"].dt.weekday.eq(0)
    hol = {pd.Timestamp(h).normalize() for h in (set(closed) | set(early))}
    tds_sorted = sorted(set(b["trade_date"]))
    post_h = {tds_sorted[i] for i in range(1, len(tds_sorted)) if tds_sorted[i - 1] in hol}
    b["is_post_holiday"] = b["slot"].eq(0) & b["trade_date"].isin(post_h)

    clean = b.loc[b["rejection_reason"].eq("")].sort_values("bar_start_utc", ignore_index=True)
    rejected = b.loc[b["rejection_reason"].ne("")].sort_values("bar_start_utc", ignore_index=True)

    log.info("bars: grid=%d accepted=%d rejected=%d", len(b), len(clean), len(rejected))
    if not rejected.empty:
        log.info("rejection reasons: %s", rejected["rejection_reason"].value_counts().to_dict())

    # --- per-slot accounting ------------------------------------------------------
    # Which slots are actually being produced? A slot that is missing from the
    # accepted set is a structural problem with the feed (or with the clock), not a
    # market event, so it is reported explicitly.
    allb = pd.concat([clean.assign(accepted=True), rejected.assign(accepted=False)],
                     ignore_index=True)
    if not allb.empty:
        tab = allb.groupby("slot").agg(
            grid=("accepted", "size"),
            accepted=("accepted", "sum"),
            mean_observed=("observed_minutes", "mean"),
            mean_coverage=("quote_coverage", "mean"),
        )
        tab["accept_rate"] = (tab["accepted"] / tab["grid"]).round(3)
        log.info("accepted by slot:\n%s", tab.round(3).to_string())
        missing = [s for s in range(slot_table.n_slots) if int(tab["accepted"].get(s, 0)) == 0]
        if missing:
            log.warning("slots with NO accepted bar in this build: %s", missing)

    clean = clean[CLEAN_COLUMNS].copy() if not clean.empty else clean
    return BarBuilderResult(bars=clean, rejected=rejected, minutes=m,
                             calendar={"closed": sorted(closed), "early": sorted(early)})


def validate_bars(bars: pd.DataFrame, slot_table: SlotTable,
                  cfg: Optional[FeatureConfig] = None) -> None:
    """Fail-closed structural checks on the accepted bars (mirrors `validate`).

    Every assertion here has a counterpart in the training project's `validate`
    function (DATA/build_nq_4h.py); the live system refuses to serve a bar the
    training pipeline would have rejected.

    The coverage / observed-minutes / missing-run checks use the same per-slot
    policy the gates used (`gate_thresholds`), so a bar that the live policy
    accepts is not rejected here by the training policy's stricter floor.
    """
    if bars.empty:
        return
    cfg = cfg or FeatureConfig()
    et = bars["bar_start_utc"].dt.tz_convert(slot_table.timezone)
    eh = bars["bar_end_utc"].dt.tz_convert(slot_table.timezone)
    problems = []

    if bars.isna().any().any():
        problems.append(f"nulls: {bars.columns[bars.isna().any()].tolist()}")
    if bars["bar_start_utc"].duplicated().any():
        problems.append("duplicate bar_start_utc")
    if not bars["slot"].isin(range(slot_table.n_slots)).all():
        problems.append("slot outside the session")
    if not bars["volume"].gt(0).all():
        problems.append("zero-volume bar accepted")
    flat = (bars["open_mid"].eq(bars["high_mid"]) & bars["high_mid"].eq(bars["low_mid"])
            & bars["low_mid"].eq(bars["close_mid"]))
    if flat.any():
        problems.append(f"{int(flat.sum())} flat OHLC bars")
    if not bars["high_mid"].ge(bars[["open_mid", "close_mid"]].max(axis=1)).all():
        problems.append("high_mid below open/close")
    if not bars["low_mid"].le(bars[["open_mid", "close_mid"]].min(axis=1)).all():
        problems.append("low_mid above open/close")
    for c in ("spread_open", "spread_close", "spread_mean"):
        if not bars[c].ge(0).all():
            problems.append(f"negative {c}")
    if not bars["spread_max"].ge(bars["spread_mean"] - 1e-12).all():
        problems.append("spread_max below spread_mean")
    if not bars["spread_mean"].ge(
            np.minimum(bars["spread_open"], bars["spread_close"]) - 1e-12).all():
        problems.append("spread_mean below min(spread_open, spread_close)")
    thr = gate_thresholds(bars, cfg)
    if not bars["quote_coverage"].ge(thr["min_coverage"] - 1e-12).all():
        problems.append("quote_coverage below the acceptance floor")
    if not bars["observed_minutes"].ge(thr["min_observed_minutes"]).all():
        problems.append("observed_minutes below the acceptance floor")
    # the missing-run gate is enforced by `classify`; the accepted frame does not
    # carry `max_missing_run_minutes` (it is not part of the training clean schema).
    if not (bars["bar_end_utc"] - bars["bar_start_utc"]).eq(pd.Timedelta(hours=4)).all():
        problems.append("bar duration is not exactly 4 hours")
    if not bars["wall_seconds"].eq(14400).all():
        problems.append("wall_seconds is not 14400")
    want_hour = bars["slot"].map(dict(zip(range(slot_table.n_slots),
                                          slot_table.start_hours)))
    if not et.dt.hour.eq(want_hour).all():
        problems.append("ET start hour does not match the slot table")
    # the calendar date must equal the ET local start date, and must be the
    # trade_date plus the slot's day offset
    cal_exp = bars["trade_date"] + pd.to_timedelta(
        bars["slot"].map(dict(zip(range(slot_table.n_slots), slot_table.day_offsets))),
        unit="D")
    if not pd.to_datetime(bars["calendar_date"]).dt.normalize().eq(
            pd.to_datetime(cal_exp).dt.normalize()).all():
        problems.append("calendar_date does not match trade_date + day offset")
    if not et.dt.date.eq(pd.to_datetime(bars["calendar_date"]).dt.date).all():
        problems.append("ET local start date does not equal calendar_date")
    # the dropped terminal block must never appear
    if (et.dt.hour == 17).any():
        problems.append("a bar starts in the dropped 17:00 ET block")
    if not bars["is_post_maintenance"].eq(bars["slot"].eq(0)).all():
        problems.append("is_post_maintenance is not slot == 0")

    if not problems:
        return
    raise ValueError("bar validation failed: " + "; ".join(problems))
