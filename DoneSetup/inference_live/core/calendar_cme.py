"""CME equity-index-futures holiday calendar.

Vendored verbatim (logic) from DATA/build_nq_4h.py in the training project
(`cme_calendar`). It produces:
  closed : full-closure days (no session)
  early  : early-close days (session halts at 13:00 ET, truncating the last slot)

The calendar is used to FLAG grid bars, exactly as in training. It does not remove
them: a holiday session bar still has to pass the minute-coverage gates, which is what
actually excludes it (reason "closure" / "stale_market_closure" in the training run).
"""
from __future__ import annotations

from datetime import date, timedelta


def _nth_wd(y: int, m: int, wd: int, n: int) -> date:
    """n-th weekday (0=Mon) of month m in year y."""
    d = date(y, m, 1)
    d += timedelta((wd - d.weekday()) % 7)
    return d + timedelta(7 * (n - 1))


def _last_wd(y: int, m: int, wd: int) -> date:
    d = date(y, m + 1, 1) - timedelta(1) if m < 12 else date(y, 12, 31)
    return d - timedelta((d.weekday() - wd) % 7)


def _easter(y: int) -> date:
    a = y % 19; b = y // 100; c = y % 100; d = b // 4; e = b % 4
    f = (b + 8) // 25; g = (b - f + 1) // 3; h = (19 * a + b - d - g + 15) % 30
    i = c // 4; k = c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451; mo = (h + l - 7 * m + 114) // 31
    dy = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, mo, dy)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(1)
    if d.weekday() == 6:
        return d + timedelta(1)
    return d


def cme_calendar(y0: int, y1: int) -> tuple[set[date], set[date]]:
    """(closed, early) sets of dates for the years y0..y1 inclusive.

    Early close = holiday session halted at 12:00 CT / 13:00 ET (slot 4 truncated) or
    12:15 CT the day before/after. Identical to the training project.
    """
    closed: set[date] = set()
    early: set[date] = set()
    for y in range(y0, y1 + 1):
        closed |= {_observed(date(y, 1, 1)), _easter(y) - timedelta(2),
                   _observed(date(y, 12, 25))}
        early |= {_nth_wd(y, 1, 0, 3), _nth_wd(y, 2, 0, 3), _last_wd(y, 5, 0),
                  _nth_wd(y, 9, 0, 1), _observed(date(y, 7, 4)), _nth_wd(y, 11, 3, 4),
                  _nth_wd(y, 11, 3, 4) + timedelta(1)}
        if y >= 2022:
            early.add(_observed(date(y, 6, 19)))
        for d in (date(y, 12, 24), date(y, 7, 3)):
            if d.weekday() < 5:
                early.add(d)
    return closed, early - closed
