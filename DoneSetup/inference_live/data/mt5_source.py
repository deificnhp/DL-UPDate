"""MetaTrader 5 1-minute data source.

Uses the official `MetaTrader5` Python package. Two hard-won details of that API are
handled here:

  * `copy_rates_range` and `copy_rates_from_pos` take EPOCH SECOND INTEGERS, not
    `datetime` objects. Passing datetimes raises TypeError/SystemError on Windows.
  * The returned `rates` array is ordered OLDEST -> NEWEST, and its `time` column is
    the bar's OPEN time in epoch seconds.

The frame produced here has the columns the bar builder expects:
    ts (datetime64[ns, UTC]), open, high, low, close, volume
and, when the symbol exposes a tick-level bid/ask series, also
    bid_open/bid_high/bid_low/bid_close and ask_* so the mid-price and spread
    machinery can be reproduced.

IMPORTANT (fidelity): the trained features are built on MID prices from a dual-feed
bid/ask vendor file. An MT5 OHLC series is a single-sided (last/bid) series. The
`prefer_mid` setting uses the tick-based mid (bid+ask)/2 when ticks are available for
the symbol; otherwise the broker's OHLC is used as-is, which is a documented
approximation (see README, "Risks").

IMPORTANT (clock): MT5 timestamps are BROKER SERVER TIME, not UTC. Measured on this
account: the API's clock runs `server_utc_offset_hours` AHEAD of true UTC -- a request
whose upper bound is 6 h past the true present returns bars labelled up to
true_now + 3 h -- and the `copy_rates_range` from/to arguments live on that same
server clock. So both directions need converting:

    true UTC of a bar   = raw server epoch - server_utc_offset_hours
    epoch to send to MT5 = true UTC epoch + server_utc_offset_hours
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import MT5Config, SlotTable
from ..logging_setup import get_logger

log = get_logger("data.mt5")

M1 = 60


def server_tz(cfg: MT5Config) -> dt.tzinfo:
    """The broker's server timezone (a fixed offset from UTC)."""
    return dt.timezone(dt.timedelta(hours=cfg.server_utc_offset_hours))


def to_true_utc(epoch_seconds, cfg: MT5Config, index=None) -> pd.Series:
    """Convert MT5 server-clock epochs to true UTC timestamps.

    `index` is carried through so the result can be assigned back onto a frame
    without index alignment surprises.
    """
    naive = pd.to_datetime(np.asarray(epoch_seconds, dtype="int64"), unit="s")
    if isinstance(naive, pd.Timestamp):
        naive = pd.DatetimeIndex([naive])
    # pd.to_datetime on a numpy array yields a DatetimeIndex, not a Series; wrap it so
    # the .dt accessor below always exists.
    naive = pd.Series(naive, index=index)
    return naive.dt.tz_localize(server_tz(cfg)).dt.tz_convert("UTC")


def to_api_epoch(ts_utc, cfg: MT5Config) -> int:
    """Convert a true-UTC instant to the epoch argument the MT5 API expects.

    The API's from/to arguments are on the broker's server clock, so a UTC instant
    must be shifted FORWARD by the server offset before it is sent.
    """
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp()) + cfg.server_utc_offset_hours * 3600


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # noqa: N813
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the MetaTrader5 package is not installed. Install it with:\n"
            "    pip install MetaTrader5\n"
            "and make sure the MetaTrader terminal is running and logged in."
        ) from exc
    return mt5

def _init_error_message(cfg: MT5Config, err) -> str:
    """The operator-facing message for a failed terminal connection."""
    return (
        f"MetaTrader5.initialize() failed: error {err[0]} '{err[1]}'.\n"
        f"  configured path : {cfg.path or '(none - auto-discover)'}\n"
        "  symbol          : " + cfg.symbol + "\n"
        "Check, in this order: (1) the configured terminal path exists and is the "
        "terminal you are running (error -2 means the path is wrong; error -10003 "
        "means the process could not be started; error -10005 means it timed out); "
        "(2) the terminal is running and logged in; (3) the symbol exists in this "
        "account (Market Watch -> Symbols).")


def initialize(cfg: MT5Config) -> None:
    """Connect to the terminal and select the symbol. Fails loudly.

    The terminal path is passed ONLY when one is configured. Passing `path=None`
    explicitly is different from omitting the keyword: MT5 rejects an explicit None
    with error -2 ("Invalid 'path' argument"), while an omitted path makes the
    package auto-discover the running terminal.
    """
    mt5 = _import_mt5()

    init_kwargs = {"timeout": cfg.initialize_timeout_ms // 1000}
    if cfg.path:
        init_kwargs["path"] = cfg.path

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        if not cfg.path:
            raise RuntimeError(_init_error_message(cfg, err))
        # A terminal that is already running (or that this call just launched and
        # whose IPC endpoint was not ready yet) makes the path-based initialize fail
        # with -10005 'IPC timeout'. The package can also attach to the RUNNING
        # terminal by omitting the `path` keyword, so fall back to that and then
        # verify the terminal we attached to is the one that was configured.
        log.warning("initialize(path=%s) failed: error %d '%s'. Retrying against the "
                    "already-running terminal (auto-discover).", cfg.path, err[0], err[1])
        if not mt5.initialize(timeout=cfg.initialize_timeout_ms // 1000):
            err2 = mt5.last_error()
            raise RuntimeError(_init_error_message(cfg, err2))
        info = mt5.terminal_info()
        running = (info.path if info is not None else "") or ""
        if running and Path(running) != Path(cfg.path):
            mt5.shutdown()
            raise RuntimeError(
                "connected to a DIFFERENT terminal than the configured one.\n"
                f"  configured path : {cfg.path}\n"
                f"  running terminal: {running}\n"
                "Close the other terminal, or point MT5Config.path at the one to use.")

    log.info("MetaTrader5 initialized: %s", mt5.terminal_info())
    log.info("clock contract: MT5 timestamps are broker server time = UTC+%d; "
             "server epochs are converted to true UTC on the way in",
             cfg.server_utc_offset_hours)

    if not mt5.symbol_select(cfg.symbol, True):
        err = mt5.last_error()
        raise RuntimeError(
            f"symbol_select('{cfg.symbol}', True) failed: error {err[0]} '{err[1]}'.\n"
            "Check the exact symbol name (MT5 usually requires a broker suffix, e.g. "
            "'NDXUSD', 'NQ2006' or 'NQ.cfd').")

    info = mt5.symbol_info(cfg.symbol)
    if info is None:
        raise RuntimeError(f"symbol_info('{cfg.symbol}') returned None")

    log.info("symbol %s: digits=%d point=%s volume_min=%s trade_mode=%s",
             cfg.symbol, info.digits, info.point, info.volume_min, info.trade_mode)

def shutdown() -> None:
    try:
        mt5 = _import_mt5()
        mt5.shutdown()
    except Exception:  # pragma: no cover - best effort
        pass


def _epoch(ts: dt.datetime) -> int:
    return int(pd.Timestamp(ts).timestamp())


def copy_minutes(cfg: MT5Config,
                 start_utc: Optional[dt.datetime] = None,
                 count: Optional[int] = None) -> pd.DataFrame:
    """Pull 1-minute bars.

    `start_utc` / `count` default to "the last `count` minutes" where count comes from
    the config. The range arguments are sent on the broker's SERVER clock and the
    returned timestamps are converted back to true UTC.
    """
    mt5 = _import_mt5()
    if count is None:
        count = cfg.bootstrap_minutes
    now_utc = pd.Timestamp.now(tz="UTC")
    if start_utc is None:
        start_utc = now_utc - pd.Timedelta(minutes=count + 10)
    frm = to_api_epoch(start_utc, cfg)
    to = to_api_epoch(now_utc, cfg)

    rates = mt5.copy_rates_range(cfg.symbol, mt5.TIMEFRAME_M1, frm, to)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(
            f"copy_rates_range('{cfg.symbol}', M1, {frm}, {to}) returned no data "
            f"(error {err[0]} '{err[1]}').\n"
            "  the arguments are on the broker's server clock (UTC+"
            f"{cfg.server_utc_offset_hours}).\n"
            "Usual causes: the symbol has no downloaded history in this terminal "
            "(open the symbol's history and press 'Download'), the terminal is "
            "offline, the requested range is outside the broker's history, or the "
            "range is too large -- MT5 rejects a single call of more than ~100 000 "
            "minutes with 'Terminal: Invalid params'.")
    df = pd.DataFrame(rates)
    df = df.rename(columns={"time": "ts"})
    # server clock -> true UTC
    df["ts"] = to_true_utc(df["ts"].to_numpy(), cfg, index=df.index)
    df = df[["ts", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    df = df.rename(columns={"tick_volume": "volume", "real_volume": "real_volume"})
    df["volume"] = df["volume"].astype("float64")
    if cfg.volume_mode == "mean":
        df["volume"] = df["volume"] / 2.0
    df = df.sort_values("ts", ignore_index=True)
    log.info("MT5 pulled %d M1 bars %s .. %s UTC (symbol=%s, server clock UTC+%d)",
             len(df), df["ts"].iloc[0], df["ts"].iloc[-1], cfg.symbol,
             cfg.server_utc_offset_hours)
    return df


def copy_ticks_mid(cfg: MT5Config,
                   start_utc: Optional[dt.datetime] = None,
                   count: Optional[int] = None) -> pd.DataFrame:
    """Best-effort bid/ask minute mid prices from tick data.

    Returns a frame with ts, open/high/low/close computed from the mid of
    (bid+ask)/2 and the per-minute volume (tick count). If the symbol has no tick
    data, returns an empty frame and the caller falls back to the OHLC series.
    """
    mt5 = _import_mt5()
    if count is None:
        count = cfg.bootstrap_minutes
    now_utc = pd.Timestamp.now(tz="UTC")
    if start_utc is None:
        start_utc = now_utc - pd.Timedelta(minutes=count + 10)
    frm = to_api_epoch(start_utc, cfg)
    to = to_api_epoch(now_utc, cfg)

    ticks = mt5.copy_ticks_range(cfg.symbol, frm, to, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        log.warning("no tick data for %s; falling back to the terminal's OHLC series",
                    cfg.symbol)
        return pd.DataFrame()

    t = pd.DataFrame(ticks)
    t["ts"] = to_true_utc(t["time"].to_numpy(), cfg, index=t.index)
    t = t[(t["bid"] > 0) & (t["ask"] > 0)].copy()
    if t.empty:
        log.warning("tick data for %s has no valid bid/ask; falling back to OHLC", cfg.symbol)
        return pd.DataFrame()
    t["mid"] = (t["bid"] + t["ask"]) / 2.0
    t["minute"] = t["ts"].dt.floor("min")
    g = t.groupby("minute")
    out = pd.DataFrame({
        "ts": g.size().index,
        "open": g["mid"].first().to_numpy(),
        "high": g["mid"].max().to_numpy(),
        "low": g["mid"].min().to_numpy(),
        "close": g["mid"].last().to_numpy(),
        "volume": g.size().to_numpy().astype("float64"),
    })
    out = out.sort_values("ts", ignore_index=True)
    log.info("MT5 ticks -> %d mid-price minutes (UTC %s .. %s)",
             len(out), out["ts"].iloc[0], out["ts"].iloc[-1])
    return out


def fetch_minutes(cfg: MT5Config,
                  start_utc: Optional[dt.datetime] = None,
                  count: Optional[int] = None) -> pd.DataFrame:
    """The single entry point: the best available 1-minute frame.

    Tries the tick-derived mid first (best fidelity to the training data), then the
    terminal's OHLC series.
    """
    if cfg.prefer_mid:
        mid = copy_ticks_mid(cfg, start_utc, count)
        if not mid.empty:
            return mid
    return copy_minutes(cfg, start_utc, count)


# ---------------------------------------------------------------------------
def clock_diagnostic(minutes: pd.DataFrame,
                     cfg: MT5Config,
                     slot_table: SlotTable,
                     n_samples: int = 5) -> dict:
    """Report the clock interpretation and check it against the expected session.

    Prints, for a few sample minutes: the RAW MT5 epoch, the interpreted true UTC,
    the interpreted America/New_York wall clock, and the slot that wall clock falls
    into. Then it profiles the feed by ET hour: the broker's daily maintenance break
    must land near ET 17:00-18:00 (the CME maintenance window) for the slot
    assignment to be right. A mis-set offset moves that hole by exactly the offset,
    so this is a direct, cheap check on the clock contract.

    Returns a dict with the derived numbers so the caller can assert on them.
    """
    if minutes.empty:
        raise ValueError("clock diagnostic needs a non-empty minute frame")

    off = cfg.server_utc_offset_hours
    ts = minutes["ts"]
    et = ts.dt.tz_convert(slot_table.timezone)
    et_naive = et.dt.tz_localize(None)
    slots = np.array([slot_table.slot_for_hour(int(h)) for h in et_naive.dt.hour],
                      dtype=np.int64)

    log.info("=== clock diagnostic (MT5 timestamps are broker server time) ===")
    log.info("  server offset applied : UTC+%d", off)
    step = max(1, len(minutes) // n_samples)
    for i in range(0, len(minutes), step):
        raw_server = ts.iloc[i] + pd.Timedelta(hours=off)
        log.info("    raw %s (server) -> UTC %s -> ET %s -> slot %d",
                 raw_server.isoformat(), ts.iloc[i].isoformat(),
                 et.iloc[i].isoformat(), int(slots[i]))
        if i // step >= n_samples - 1:
            break

    # --- the daily break, by ET hour ------------------------------------------------
    cnt = et_naive.dt.hour.value_counts().reindex(range(24), fill_value=0)
    empty = [h for h in range(24) if int(cnt[h]) == 0]
    profile = {h: int(cnt[h]) for h in range(24)}
    log.info("  minutes per ET hour  : %s", profile)
    log.info("  ET hours with NO data : %s", empty)

    # The session is 18:00 -> 14:00 ET. Hours outside it are empty by definition
    # (the 14:00-18:00 block is dropped, and the broker does not quote it either).
    out_of_session = set(range(14, 18))
    inside = [h for h in empty if h not in out_of_session]
    ok = (17 in inside) or (not inside)
    if inside:
        log.info("  daily break inside the session at ET hour(s): %s", inside)
    else:
        log.info("  no empty hour inside the 18:00-14:00 ET session in this sample")
    if 17 in inside:
        log.info("  OK: the break lands in the CME maintenance window (ET 17:00-18:00)")
    elif inside:
        log.warning("  WARNING: the break is at ET %s, not 17:00. If the server "
                    "offset is wrong by n hours, the break moves by n hours.",
                    inside)
    return {
        "server_utc_offset_hours": off,
        "et_hour_profile": profile,
        "empty_et_hours": empty,
        "empty_et_hours_inside_session": inside,
        "break_at_et_17": 17 in inside,
        "session_break_aligned": ok,
    }


def detect_broker_utc_offset(last_tick_epoch: float, system_utc_epoch: float) -> float:
    """Detect broker server UTC offset by comparing last tick server epoch to system UTC epoch.

    Rounds the difference to the nearest half-hour (or hour) during market open hours.
    For instance, if server tick is 10800s ahead of UTC, offset is +3.0 hours.
    """
    diff_sec = float(last_tick_epoch) - float(system_utc_epoch)
    diff_hours = diff_sec / 3600.0
    return round(diff_hours * 2.0) / 2.0
