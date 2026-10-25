"""Session-anchored slot geometry: ET wall clock -> slot, trade_date, calendar_date.

This is the live system's copy of the timing rules the trained model was built on.
Source of truth: DATA/build_nq_4h.py (SLOT_START_H, SLOT_DAY_OFF, assign_slots,
build_grid) in the training project.

Rules reproduced here, exactly:
  * 5 slots per session, each exactly 4 hours, anchored to America/New_York wall clock.
  * slot 0: 18:00-22:00 ET   (calendar day = trade_date - 1)
    slot 1: 22:00-02:00 ET   (calendar day = trade_date - 1, ends on trade_date)
    slot 2: 02:00-06:00 ET   (calendar day = trade_date)
    slot 3: 06:00-10:00 ET   (calendar day = trade_date)
    slot 4: 10:00-14:00 ET   (calendar day = trade_date)
  * The 14:00-18:00 ET block is the dropped terminal slot: it is not in the session.
  * A minute's trade_date is its ET date, +1 day if the ET hour is >= 18.
  * Bar boundaries are localized in ET with ambiguous="raise" / nonexistent="raise",
    so a DST transition fails loudly instead of silently shifting a bar.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import SlotTable
from ..logging_setup import get_logger

log = get_logger("core.slots")

DAY = dt.timedelta(days=1)


@dataclass(frozen=True)
class BarWindow:
    """One scheduled bar on the session grid.

    `trade_date` and `calendar_date` are midnight-normalised pandas Timestamps
    (datetime64[ns]), matching the Arrow date32 semantics of the training project's
    clean bar file.
    """
    trade_date: pd.Timestamp
    slot: int
    calendar_date: pd.Timestamp
    bar_start_utc: pd.Timestamp
    bar_end_utc: pd.Timestamp


def _localize(naive: pd.Series, tz: str) -> pd.Series:
    """Localize naive ET wall-clock timestamps. Ambiguous/nonexistent raise."""
    return naive.dt.tz_localize(tz, ambiguous="raise", nonexistent="raise")


def trade_date_for_et(et_naive: pd.Series, slot_table: SlotTable) -> pd.Series:
    """CME business date of a minute, from its ET wall-clock timestamp.

    Reproduces `assign_slots`: trade_date = ET date, +1 day if the ET hour >= 18.
    (The session starts at 18:00, so evening minutes belong to the next business day.)

    Returns a datetime64[ns] Series of dates (midnight-normalised).
    """
    hour = et_naive.dt.hour.to_numpy()
    day_shift = np.where(hour >= slot_table.start_hours[0], 1, 0)
    d = et_naive.dt.normalize()
    return d + pd.to_timedelta(day_shift, unit="D")


def slot_for_et_hour(hour: int, slot_table: SlotTable) -> int:
    """Slot index for an ET hour, or -1 outside the session (the dropped 14:00-18:00)."""
    return slot_table.slot_for_hour(hour)


def assign_slots(minutes: pd.DataFrame, slot_table: SlotTable) -> pd.DataFrame:
    """Attach `slot` and `trade_date` to a frame of 1-minute rows.

    `minutes` must have a UTC timestamp column named `ts` (datetime64[ns, UTC]).
    Rows outside the session, and rows whose trade_date is not a weekday, are dropped
    exactly as `assign_slots` did (`keep = (slot >= 0) & weekday < 5`).
    """
    if "ts" not in minutes.columns:
        raise ValueError("minutes frame must have a 'ts' column")
    df = minutes.copy()
    et = df["ts"].dt.tz_convert(slot_table.timezone)
    et_naive = et.dt.tz_localize(None)

    hour = et_naive.dt.hour.to_numpy()
    slot = np.array([slot_table.slot_for_hour(int(h)) for h in hour], dtype=np.int64)

    df["slot"] = slot
    df["trade_date"] = trade_date_for_et(et_naive, slot_table)
    df["calendar_date"] = et_naive.dt.normalize()

    weekday_ok = df["trade_date"].dt.weekday.lt(5).to_numpy()
    keep = (slot >= 0) & weekday_ok
    n_drop = int((~keep).sum())
    if n_drop:
        log.info("minutes outside the session or on a non-weekday dropped: %d", n_drop)
    out = df.loc[keep].reset_index(drop=True)
    out["et_naive"] = et_naive.loc[keep].to_numpy()
    return out


def build_grid(trade_dates: Iterable[dt.date],
               slot_table: SlotTable,
               closed: Optional[set] = None,
               early: Optional[set] = None) -> pd.DataFrame:
    """The session grid: for every weekday trade_date, the five scheduled bars.

    Mirrors `build_grid`. `closed` / `early` are sets of dates (holiday closures and
    early-close sessions) used to flag the bars; they do not remove grid rows -- the
    validity gates do that, exactly as in training.
    """
    closed = closed or set()
    early = early or set()
    g = _windows_for_dates(list(trade_dates), slot_table)
    if g.empty:
        return g
    closed_ts = {pd.Timestamp(c).normalize() for c in closed}
    early_ts = {pd.Timestamp(e).normalize() for e in early}
    g["is_holiday_session"] = g["trade_date"].isin(closed_ts | early_ts)
    g["is_early_close"] = g["trade_date"].isin(early_ts) & g["slot"].eq(slot_table.n_slots - 1)
    g["is_partial_calendar"] = g["is_early_close"]
    g["is_closure_day"] = g["trade_date"].isin(closed_ts)
    return g


def grid_for_span(minutes: pd.DataFrame,
                  slot_table: SlotTable,
                  closed: Optional[set] = None,
                  early: Optional[set] = None) -> pd.DataFrame:
    """Build the grid covering the span of the supplied minutes."""
    if minutes.empty:
        return pd.DataFrame()
    et = minutes["ts"].dt.tz_convert(slot_table.timezone)
    et_naive = et.dt.tz_localize(None)
    tds = trade_date_for_et(et_naive, slot_table)
    lo = tds.min() - pd.Timedelta(days=2)
    hi = tds.max() + pd.Timedelta(days=2)
    dates = []
    d = lo
    while d <= hi:
        dates.append(d)
        d += pd.Timedelta(days=1)
    return build_grid(dates, slot_table, closed, early)


def slot_starts_for_trade_date(trade_date: dt.date,
                               slot_table: SlotTable) -> List[pd.Timestamp]:
    """UTC start timestamps of the five bars of one trade date."""
    w = _windows_for_dates([trade_date], slot_table)
    return list(w.sort_values("slot")["bar_start_utc"])


def _windows_for_dates(dates: Sequence,
                       slot_table: SlotTable) -> pd.DataFrame:
    """Vectorized bar windows for a list of candidate trade dates.

    `trade_date` and `calendar_date` come back as datetime64[ns] (date-normalised),
    matching the Arrow date32 semantics of the training project's clean bar file.
    """
    rows = []
    for d in dates:
        d = pd.Timestamp(d).normalize()
        if d.weekday() >= 5:
            continue
        for s in range(slot_table.n_slots):
            cal = d + pd.Timedelta(days=int(slot_table.day_offsets[s]))
            naive = pd.Timestamp(cal.year, cal.month, cal.day,
                                 int(slot_table.start_hours[s]))
            rows.append((d, s, cal, naive))
    if not rows:
        return pd.DataFrame(columns=["trade_date", "slot", "calendar_date",
                                     "bar_start_utc", "bar_end_utc"])
    df = pd.DataFrame(rows, columns=["trade_date", "slot", "calendar_date", "_naive"])
    st = _localize(df["_naive"], slot_table.timezone)
    en = _localize(df["_naive"] + pd.Timedelta(hours=slot_table.bar_hours), slot_table.timezone)
    df["bar_start_utc"] = st.dt.tz_convert("UTC")
    df["bar_end_utc"] = en.dt.tz_convert("UTC")
    return df.drop(columns=["_naive"])


def _windows_around(around: dt.date, slot_table: SlotTable, span_days: int = 10) -> pd.DataFrame:
    dates = [around + dt.timedelta(days=k) for k in range(-span_days, span_days + 1)]
    return _windows_for_dates(dates, slot_table)


def next_session_bar(after_utc: pd.Timestamp,
                     slot_table: SlotTable,
                     closed: Optional[set] = None,
                     early: Optional[set] = None) -> Optional[BarWindow]:
    """The next scheduled bar that ENDS strictly after `after_utc`.

    Used by the engine to answer "which bar has just closed?" and "what is the
    target horizon of the current decision?".
    """
    after_utc = pd.Timestamp(after_utc)
    if after_utc.tzinfo is None:
        raise ValueError("after_utc must be timezone-aware")
    w = _windows_around(after_utc.tz_convert(slot_table.timezone).date(), slot_table)
    w = w[w["bar_end_utc"] > after_utc]
    if w.empty:
        return None
    r = w.sort_values("bar_end_utc").iloc[0]
    return BarWindow(trade_date=r["trade_date"], slot=int(r["slot"]),
                     calendar_date=r["calendar_date"],
                     bar_start_utc=r["bar_start_utc"], bar_end_utc=r["bar_end_utc"])


def bar_just_closed(now_utc: pd.Timestamp,
                    slot_table: SlotTable,
                    closed: Optional[set] = None,
                    early: Optional[set] = None,
                    max_lag_hours: Optional[float] = None) -> Optional[BarWindow]:
    """The bar whose scheduled end time is the most recent one <= now_utc, or None.

    `max_lag_hours` optionally rejects a bar that ended longer than that ago (a bar
    that ended 3 days ago is not "just closed"). The engine passes the bar period
    (4 h) so that a decision is only taken in the window right after a bar closes.
    """
    now_utc = pd.Timestamp(now_utc)
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    w = _windows_around(now_utc.tz_convert(slot_table.timezone).date(), slot_table)
    w = w[w["bar_end_utc"] <= now_utc]
    if w.empty:
        return None
    r = w.sort_values("bar_end_utc").iloc[-1]
    lag_h = (now_utc - r["bar_end_utc"]).total_seconds() / 3600.0
    if max_lag_hours is not None and lag_h > max_lag_hours:
        return None
    return BarWindow(trade_date=r["trade_date"], slot=int(r["slot"]),
                     calendar_date=r["calendar_date"],
                     bar_start_utc=r["bar_start_utc"], bar_end_utc=r["bar_end_utc"])


# ---------------------------------------------------------------------------
def next_session_date(trade_date: dt.date,
                       slot_table: SlotTable,
                       closed: Optional[set] = None,
                       max_gap_days: int = 5) -> Optional[dt.date]:
    """The next CME session date after `trade_date`.

    A session date is a weekday that is not a CME closure day. `closed` holds the
    closure dates; `early` (early-close sessions) are still trading days and are
    therefore NOT skipped here.

    Returns None if the next session is more than `max_gap_days` calendar days away
    (the training project's `max_session_gap_days` bound, beyond which the
    contiguity rule breaks the session link and the row was never trainable).
    """
    closed = closed or set()
    d = pd.Timestamp(trade_date).normalize() + pd.Timedelta(days=1)
    for _ in range(max_gap_days + 1):
        if d.weekday() < 5 and d.normalize() not in {pd.Timestamp(c).normalize() for c in closed}:
            gap = (d - pd.Timestamp(trade_date).normalize()).days
            return d.date() if gap <= max_gap_days else None
        d = d + pd.Timedelta(days=1)
    return None


def scheduled_successor(trade_date,
                         slot: int,
                         slot_table: SlotTable,
                         closed: Optional[set] = None,
                         early: Optional[set] = None,
                         max_gap_days: int = 5) -> Optional[BarWindow]:
    """The SCHEDULED bar that follows (trade_date, slot) in the session calendar.

    This is the live system's analogue of "the chronological successor of row t" that
    the training pipeline used for the target bar. The difference is that in training
    the successor was a row that had already been built and accepted; live, the
    successor has not happened yet, but its SCHEDULE is a deterministic fact, so the
    target covariates (slot_next / dow_next) are still exactly the ones the model was
    trained on.

    Rules (mirroring the training project's contiguity rule):
      * slot < last slot, and the trade date is a normal session  -> (same date, slot+1)
      * slot == last slot                                         -> (next session, slot 0)
      * the trade date is an early-close session                   -> (next session, slot 0)
        (the training grid truncates slot 4 on early-close dates, so slot 4 is rejected
         by the coverage gate and slot 3 chains to the next session's slot 0)

    Returns None when the next session is further away than `max_gap_days`, i.e. when
    the session link the training pipeline required does not hold.
    """
    closed = closed or set()
    early = early or set()
    d = pd.Timestamp(trade_date).normalize()
    is_early = d in {pd.Timestamp(e).normalize() for e in early}
    last = slot_table.n_slots - 1

    if slot < last and not is_early:
        nxt = _windows_for_dates([d.date()], slot_table)
        nxt = nxt[nxt["slot"] == slot + 1]
        if nxt.empty:
            return None
        r = nxt.iloc[0]
        return BarWindow(trade_date=r["trade_date"], slot=int(r["slot"]),
                         calendar_date=r["calendar_date"],
                         bar_start_utc=r["bar_start_utc"], bar_end_utc=r["bar_end_utc"])

    nxt_date = next_session_date(d.date(), slot_table, closed, max_gap_days)
    if nxt_date is None:
        return None
    nxt = _windows_for_dates([nxt_date], slot_table)
    nxt = nxt[nxt["slot"] == 0]
    if nxt.empty:
        return None
    r = nxt.iloc[0]
    return BarWindow(trade_date=r["trade_date"], slot=int(r["slot"]),
                     calendar_date=r["calendar_date"],
                     bar_start_utc=r["bar_start_utc"], bar_end_utc=r["bar_end_utc"])

