#!/usr/bin/env python3.12
"""
NQ 4-hour bar pipeline (v1.0.0-adapted)
DATA/RAW/BID.csv + DATA/RAW/ASK.csv (1-min OHLCV, 'Time (UTC)')
  -> DATA/PROCESSED/nq_4h_clean.parquet       (29 cols, non-null)
  -> DATA/PROCESSED/nq_4h_exclusions.parquet  (rejected candidates + reason)

Deterministic, causal, no look-ahead. pandas + pyarrow (polars-free so it runs anywhere).
Usage: python build_nq_4h.py [--root .]
"""
from __future__ import annotations
import argparse, sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ------------------------------------------------------------------ config
VERSION        = "1.0.0-adapted"
CONTRACT_ID    = "NQ_CONTINUOUS"
TZ_ET          = "America/New_York"
EXPECTED_MIN   = 240
BAR_SECONDS    = 14_400
MIN_COVERAGE   = 0.99           # -> observed >= 238
MAX_MISS_RUN   = 1
TICK           = 0.25           # NQ tick size (index points)
HARD_SPREAD_CAP = 25.0          # absolute contractual sanity ceiling (points)
CAP_K          = 5.0            # median + K*IQR
CAP_MIN_TICKS  = 4              # cap never tighter than median + 4 ticks
CAP_WINDOW     = 7_200          # ~5 sessions of genuine minutes, causal (shifted)
CAP_MIN_PERIODS = 30
VOLUME_MODE    = "sum"          # "sum" (bid+ask vendor vol) | "mean"
SLOT_START_H   = {0: 18, 1: 22, 2: 2, 3: 6, 4: 10}   # ET wall clock
SLOT_DAY_OFF   = {0: -1, 1: -1, 2: 0, 3: 0, 4: 0}    # calendar day vs trade_date

# ------------------------------------------------------------------ CME calendar (equity index futures)
def _nth_wd(y, m, wd, n):          # n-th weekday (0=Mon) of month
    d = date(y, m, 1); d += timedelta((wd - d.weekday()) % 7); return d + timedelta(7 * (n - 1))
def _last_wd(y, m, wd):
    d = date(y, m + 1, 1) - timedelta(1) if m < 12 else date(y, 12, 31)
    return d - timedelta((d.weekday() - wd) % 7)
def _easter(y):
    a=y%19;b=y//100;c=y%100;d=b//4;e=b%4;f=(b+8)//25;g=(b-f+1)//3;h=(19*a+b-d-g+15)%30
    i=c//4;k=c%4;l=(32+2*e+2*i-h-k)%7;m=(a+11*h+22*l)//451;mo=(h+l-7*m+114)//31;dy=((h+l-7*m+114)%31)+1
    return date(y, mo, dy)
def _observed(d):
    return d - timedelta(1) if d.weekday() == 5 else d + timedelta(1) if d.weekday() == 6 else d

def cme_calendar(y0, y1):
    """returns (full_closure set, early_close set). Early close = holiday session halted
    at 12:00 CT / 13:00 ET (slot 4 truncated) or 12:15 CT day-before/after."""
    closed, early = set(), set()
    for y in range(y0, y1 + 1):
        closed |= {_observed(date(y, 1, 1)), _easter(y) - timedelta(2), _observed(date(y, 12, 25))}
        early |= {_nth_wd(y, 1, 0, 3), _nth_wd(y, 2, 0, 3), _last_wd(y, 5, 0), _nth_wd(y, 9, 0, 1),
                  _observed(date(y, 7, 4)), _nth_wd(y, 11, 3, 4),            # MLK, Pres, Mem, Labor, Jul4, Thanksgiving
                  _nth_wd(y, 11, 3, 4) + timedelta(1)}                        # Black Friday
        if y >= 2022: early.add(_observed(date(y, 6, 19)))
        for d in (date(y, 12, 24), date(y, 7, 3)):
            if d.weekday() < 5: early.add(d)
    return closed, early - closed

# ------------------------------------------------------------------ ingestion
def _read_side(path: Path, side: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    tcol = next(c for c in df.columns if c.lower().startswith("time"))
    raw = df[tcol].astype(str).str.replace(r"\s*(UTC|GMT)$", "", regex=True).str.strip()
    ts = None
    for fmt in ("%Y.%m.%d %H:%M:%S.%f", "%Y.%m.%d %H:%M:%S", "%d.%m.%Y %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try: ts = pd.to_datetime(raw, format=fmt, utc=True); break
        except (ValueError, TypeError): pass
    if ts is None: ts = pd.to_datetime(raw, format="mixed", utc=True)
    ren = {c: f"{side}_{c.lower()}" for c in ("Open", "High", "Low", "Close", "Volume")}
    out = df.rename(columns=ren)[list(ren.values())].astype("float64")
    out.insert(0, "ts", ts.dt.as_unit("us"))
    if out["ts"].dt.second.ne(0).any() or out["ts"].dt.microsecond.ne(0).any():
        raise ValueError(f"{path}: non minute-aligned timestamps")
    dup = out["ts"].duplicated(keep="last").sum()
    if dup: print(f"[warn] {side}: {dup} duplicate timestamps, keeping last", file=sys.stderr)
    return out.drop_duplicates("ts", keep="last").sort_values("ts", ignore_index=True)

def load_minutes(raw: Path) -> pd.DataFrame:
    b, a = _read_side(raw / "BID.csv", "bid"), _read_side(raw / "ASK.csv", "ask")
    m = b.merge(a, on="ts", how="inner", validate="one_to_one").sort_values("ts", ignore_index=True)
    print(f"[ingest] bid={len(b):,} ask={len(a):,} joined={len(m):,}")
    # crossed book -> corrupted, discarded before aggregation
    crossed = (m.ask_close < m.bid_close) | (m.ask_open < m.bid_open)
    print(f"[ingest] crossed minutes discarded: {int(crossed.sum()):,}")
    m = m.loc[~crossed].reset_index(drop=True)
    m["is_locked"] = m.ask_close.eq(m.bid_close)
    for f in ("open", "high", "low", "close"):
        m[f"mid_{f}"] = (m[f"bid_{f}"] + m[f"ask_{f}"]) / 2.0
    m["spread_raw"] = m.ask_close - m.bid_close
    m["spread_open_raw"] = m.ask_open - m.bid_open
    m["volume"] = (m.bid_volume + m.ask_volume) * (0.5 if VOLUME_MODE == "mean" else 1.0)
    # synthetic forward-fill: zero volume, or flat minute identical to previous minute on both sides
    q = ["bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"]
    flat = (m.bid_open.eq(m.bid_close) & m.bid_high.eq(m.bid_low) & m.bid_open.eq(m.bid_high) &
            m.ask_open.eq(m.ask_close) & m.ask_high.eq(m.ask_low) & m.ask_open.eq(m.ask_high))
    prev_same = m.bid_close.eq(m.bid_close.shift()) & m.ask_close.eq(m.ask_close.shift())
    m["is_synthetic"] = m.volume.le(0) | (flat & prev_same)
    m["genuine"] = ~m.is_synthetic
    print(f"[ingest] synthetic/non-tradable minutes: {int(m.is_synthetic.sum()):,}")
    return m

# ------------------------------------------------------------------ causal spread cap
def apply_spread_cap(m: pd.DataFrame) -> pd.DataFrame:
    g = m.loc[m.genuine, "spread_raw"]
    roll = g.rolling(CAP_WINDOW, min_periods=CAP_MIN_PERIODS)
    med, q1, q3 = (roll.quantile(p).shift(1) for p in (0.5, 0.25, 0.75))   # shift(1) => strictly past
    cap = (med + CAP_K * (q3 - q1)).clip(lower=med + CAP_MIN_TICKS * TICK).clip(upper=HARD_SPREAD_CAP)
    m["spread_cap"] = cap.reindex(m.index).ffill().fillna(HARD_SPREAD_CAP)
    m["spread"] = m.spread_raw.clip(lower=0.0, upper=m.spread_cap)
    print(f"[spread] minutes capped: {int((m.spread_raw > m.spread_cap).sum()):,}")
    return m

# ------------------------------------------------------------------ session mapping
def assign_slots(m: pd.DataFrame) -> pd.DataFrame:
    loc = m.ts.dt.tz_convert(TZ_ET)
    h = loc.dt.hour.to_numpy()
    ld = loc.dt.tz_localize(None).dt.normalize()
    slot = np.select([(h >= 18) & (h < 22), (h >= 22) | (h < 2), (h >= 2) & (h < 6),
                      (h >= 6) & (h < 10), (h >= 10) & (h < 14)], [0, 1, 2, 3, 4], -1)
    day_shift = np.where(h >= 18, 1, 0)                    # evening session belongs to next trade date
    m["slot"] = slot
    m["trade_date"] = (ld + pd.to_timedelta(day_shift, unit="D")).dt.date
    keep = (slot >= 0) & pd.to_datetime(m.trade_date).dt.weekday.lt(5).to_numpy()
    print(f"[slots] minutes outside slots 0-4 / halts / slot-5 dropped: {int((~keep).sum()):,}")
    return m.loc[keep].reset_index(drop=True)

def build_grid(d0: date, d1: date, closed, early) -> pd.DataFrame:
    rows = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            for s in range(5):
                cal = d + timedelta(SLOT_DAY_OFF[s])
                rows.append((d, s, cal, pd.Timestamp(cal) + pd.Timedelta(hours=SLOT_START_H[s])))
        d += timedelta(1)
    g = pd.DataFrame(rows, columns=["trade_date", "slot", "calendar_date", "local_start"])
    st = g.local_start.dt.tz_localize(TZ_ET, ambiguous="raise", nonexistent="raise")
    en = (g.local_start + pd.Timedelta(hours=4)).dt.tz_localize(TZ_ET, ambiguous="raise", nonexistent="raise")
    g["bar_start_utc"] = st.dt.tz_convert("UTC").dt.as_unit("us")
    g["bar_end_utc"] = en.dt.tz_convert("UTC").dt.as_unit("us")
    g["is_holiday_session"] = g.trade_date.isin(closed | early)
    g["is_early_close"] = g.trade_date.isin(early) & g.slot.eq(4)   # 13:00 ET halt truncates slot 4 only
    g["is_partial_calendar"] = g.is_early_close
    g["is_closure_day"] = g.trade_date.isin(closed)
    return g.drop(columns="local_start")

# ------------------------------------------------------------------ aggregation
def aggregate(m: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    key = ["trade_date", "slot"]
    raw_cnt = m.groupby(key).size().rename("raw_minutes")
    raw_vol = m.groupby(key).volume.sum().rename("raw_volume")
    gm = m.loc[m.genuine].sort_values("ts")
    grp = gm.groupby(key, sort=False)
    agg = grp.agg(open_mid=("mid_open", "first"), high_mid=("mid_high", "max"), low_mid=("mid_low", "min"),
                  close_mid=("mid_close", "last"), spread_open=("spread", "first"),
                  spread_mean=("spread", "mean"), spread_max=("spread", "max"),
                  spread_close=("spread", "last"), volume=("volume", "sum"),
                  observed_minutes=("ts", "size"), first_ts=("ts", "first"), last_ts=("ts", "last"))
    # longest run of missing genuine minutes inside [start, end)
    gm = gm.merge(grid[key + ["bar_start_utc"]], on=key, how="left")
    off = ((gm.ts - gm.bar_start_utc).dt.total_seconds() // 60).astype("int64")
    gm["off"] = off
    gm["gap"] = gm.groupby(key).off.diff().fillna(0) - 1
    inner = gm.groupby(key).gap.max().clip(lower=0)
    lead = gm.groupby(key).off.min()
    trail = (EXPECTED_MIN - 1) - gm.groupby(key).off.max()
    agg["max_missing_run_minutes"] = pd.concat([inner, lead, trail], axis=1).max(axis=1).astype("int64")
    out = grid.merge(agg.reset_index(), on=key, how="left") \
              .merge(raw_cnt.reset_index(), on=key, how="left").merge(raw_vol.reset_index(), on=key, how="left")
    out["observed_minutes"] = out.observed_minutes.fillna(0).astype("int64")
    out["raw_minutes"] = out.raw_minutes.fillna(0).astype("int64")
    out["volume"] = out.volume.fillna(0.0)
    out["max_missing_run_minutes"] = out.max_missing_run_minutes.fillna(EXPECTED_MIN).astype("int64")
    out["quote_coverage"] = out.observed_minutes / EXPECTED_MIN
    return out

def enforce_spread_invariants(b: pd.DataFrame) -> pd.DataFrame:
    """Clamp (never drop) so that max >= mean >= min(open, close) >= 0.
    All inputs are already causally capped at minute level, so this only resolves the
    shape case where a bar opens AND closes wide but trades tight in between: the narrower
    of open/close is clamped down to the mean (the wider one is left untouched)."""
    for c in ("spread_open", "spread_close", "spread_mean", "spread_max"):
        b[c] = b[c].clip(lower=0.0)
    b["spread_max"] = np.maximum(b.spread_max, b.spread_mean)
    viol = np.minimum(b.spread_open, b.spread_close) > b.spread_mean
    o_is_min = b.spread_open <= b.spread_close
    b.loc[viol & o_is_min, "spread_open"] = b.loc[viol & o_is_min, "spread_mean"]
    b.loc[viol & ~o_is_min, "spread_close"] = b.loc[viol & ~o_is_min, "spread_mean"]
    print(f"[spread] bars with open/close spread clamped to mean: {int(viol.sum()):,}")
    return b

def classify(b: pd.DataFrame) -> pd.Series:
    flat = b.high_mid.eq(b.low_mid) | (b.open_mid.eq(b.close_mid) & b.high_mid.eq(b.open_mid))
    conds = [
        b.raw_minutes.eq(0) | (b.is_closure_day & b.observed_minutes.eq(0)),
        b.observed_minutes.eq(0) | b.volume.le(0),
        flat.fillna(True),
        b.quote_coverage.lt(MIN_COVERAGE),
        b.max_missing_run_minutes.gt(MAX_MISS_RUN),
    ]
    reasons = ["closure", "zero_volume_synthetic", "stale_market_closure",
               "low_coverage", "missing_run_exceeded"]
    return pd.Series(np.select(conds, reasons, default=""), index=b.index)

# ------------------------------------------------------------------ schema + write
CLEAN_SCHEMA = pa.schema([
    ("interval_id", pa.string()), ("bar_start_utc", pa.timestamp("us", tz="UTC")),
    ("bar_end_utc", pa.timestamp("us", tz="UTC")), ("trade_date", pa.date32()), ("calendar_date", pa.date32()),
    ("slot", pa.int64()), ("contract_id", pa.string()),
    ("open_mid", pa.float64()), ("high_mid", pa.float64()), ("low_mid", pa.float64()), ("close_mid", pa.float64()),
    ("spread_open", pa.float64()), ("spread_mean", pa.float64()), ("spread_max", pa.float64()),
    ("spread_close", pa.float64()), ("volume", pa.float64()),
    ("expected_minutes", pa.int64()), ("observed_minutes", pa.int64()), ("quote_coverage", pa.float64()),
    ("active_seconds", pa.int64()), ("wall_seconds", pa.int64()),
    ("is_terminal_3h", pa.bool_()), ("is_partial_calendar", pa.bool_()), ("is_early_close", pa.bool_()),
    ("is_holiday_session", pa.bool_()), ("is_roll_session", pa.bool_()), ("is_post_maintenance", pa.bool_()),
    ("is_post_weekend", pa.bool_()), ("is_post_holiday", pa.bool_()),
])
EXCL_SCHEMA = pa.schema([
    ("interval_id", pa.string()), ("bar_start_utc", pa.timestamp("us", tz="UTC")),
    ("bar_end_utc", pa.timestamp("us", tz="UTC")), ("slot", pa.int64()), ("trade_date", pa.date32()),
    ("volume", pa.float64()), ("quote_coverage", pa.float64()), ("observed_minutes", pa.int64()),
    ("max_missing_run_minutes", pa.int64()), ("rejection_reason", pa.string()),
])

def _write(df, schema, path):
    t = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
    pq.write_table(t, path, compression="zstd")

def run(root: Path) -> None:
    raw, outd = root / "DATA" / "RAW", root / "DATA" / "PROCESSED"
    outd.mkdir(parents=True, exist_ok=True)
    m = apply_spread_cap(load_minutes(raw))
    m = assign_slots(m)
    d0, d1 = min(m.trade_date), max(m.trade_date)
    closed, early = cme_calendar(d0.year, d1.year)
    grid = build_grid(d0, d1, closed, early)
    b = aggregate(m, grid)
    b["interval_id"] = b.trade_date.map(lambda d: d.isoformat()) + "_S" + b.slot.astype(str) + f"_{VERSION}"
    b["contract_id"] = CONTRACT_ID
    b["expected_minutes"] = EXPECTED_MIN
    b["active_seconds"] = BAR_SECONDS
    b["wall_seconds"] = ((b.bar_end_utc - b.bar_start_utc).dt.total_seconds()).astype("int64")
    b["is_terminal_3h"] = False
    b["is_roll_session"] = False
    b["is_post_maintenance"] = b.slot.eq(0)
    b["is_post_weekend"] = b.slot.eq(0) & pd.to_datetime(b.trade_date).dt.weekday.eq(0)
    # first slot of the first trade date after a holiday trade date (skipping weekends)
    td = sorted(set(b.trade_date)); hol = closed | early
    post_h = {td[i] for i in range(1, len(td)) if td[i - 1] in hol}
    b["is_post_holiday"] = b.slot.eq(0) & b.trade_date.isin(post_h)
    b = enforce_spread_invariants(b)
    b["rejection_reason"] = classify(b)

    clean = b.loc[b.rejection_reason.eq("")].sort_values("bar_start_utc", ignore_index=True)
    excl = b.loc[b.rejection_reason.ne("")].sort_values("bar_start_utc", ignore_index=True)
    # trim leading/trailing grid padding outside the vendor data span from exclusions noise
    span = (excl.bar_end_utc > m.ts.min()) & (excl.bar_start_utc <= m.ts.max())
    excl = excl.loc[span].reset_index(drop=True)

    validate(clean)
    _write(clean, CLEAN_SCHEMA, outd / "nq_4h_clean.parquet")
    _write(excl, EXCL_SCHEMA, outd / "nq_4h_exclusions.parquet")
    print(f"[out] clean={len(clean):,} excluded={len(excl):,}")
    print(excl.rejection_reason.value_counts().to_string())
    yr = pd.DataFrame({"clean": clean.groupby(pd.to_datetime(clean.trade_date).dt.year).size(),
                       "excluded": excl.groupby(pd.to_datetime(excl.trade_date).dt.year).size()}).fillna(0).astype(int)
    print("[retention by trade_date year]"); print(yr.to_string())
    print(pd.crosstab(pd.to_datetime(excl.trade_date).dt.year, excl.rejection_reason).to_string())
    # trade-date audit: local ET start date must equal trade_date-1 for slots 0/1, trade_date for 2-4
    et_d = clean.bar_start_utc.dt.tz_convert(TZ_ET).dt.date
    exp = [t + timedelta(SLOT_DAY_OFF[s_]) for t, s_ in zip(clean.trade_date, clean.slot)]
    print(f"[audit] trade-date mismatches: {int((et_d.values != np.array(exp)).sum())} | "
          f"slot1 ET start hours: {sorted(clean.loc[clean.slot.eq(1), 'bar_start_utc'].dt.tz_convert(TZ_ET).dt.hour.unique())}")

# ------------------------------------------------------------------ hard gates (fail loud)
def validate(c: pd.DataFrame) -> None:
    c = c[CLEAN_SCHEMA.names]
    assert c.shape[1] == 29, "column count"
    assert not c.isna().any().any(), f"nulls: {c.columns[c.isna().any()].tolist()}"
    assert c.interval_id.is_unique and c.bar_start_utc.is_unique
    assert c.slot.isin([0, 1, 2, 3, 4]).all()
    assert (c.volume > 0).all(), "zero-volume bar leaked"
    assert not (c.open_mid.eq(c.high_mid) & c.high_mid.eq(c.low_mid) & c.low_mid.eq(c.close_mid)).any(), "flat OHLC"
    assert (c.high_mid >= c[["open_mid", "close_mid"]].max(axis=1)).all()
    assert (c.low_mid <= c[["open_mid", "close_mid"]].min(axis=1)).all()
    assert (c[["spread_open", "spread_close", "spread_mean"]] >= 0).all().all()
    assert (c.spread_max >= c.spread_mean - 1e-12).all()
    assert (c.spread_mean >= np.minimum(c.spread_open, c.spread_close) - 1e-12).all()
    assert (c.quote_coverage >= MIN_COVERAGE).all() and (c.observed_minutes >= 238).all()
    assert (c.wall_seconds == BAR_SECONDS).all() and (c.bar_end_utc - c.bar_start_utc).eq(pd.Timedelta(hours=4)).all()
    et = c.bar_start_utc.dt.tz_convert(TZ_ET)
    assert (et.dt.hour == c.slot.map(SLOT_START_H)).all(), "DST/slot misalignment"
    cal_exp = [t + timedelta(SLOT_DAY_OFF[s]) for t, s in zip(c.trade_date, c.slot)]
    assert (c.calendar_date.tolist() == cal_exp) and (et.dt.date == c.calendar_date).all()
    eh = c.bar_end_utc.dt.tz_convert(TZ_ET)
    assert not ((et.dt.hour == 17) | (eh.dt.hour.between(15, 17) & (c.slot == 4) & (eh.dt.minute > 0))).any()
    assert (c.is_post_maintenance == c.slot.eq(0)).all()
    print(f"[validate] {len(c):,} clean bars pass all internal gates")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".")
    run(Path(ap.parse_args().root).resolve())
