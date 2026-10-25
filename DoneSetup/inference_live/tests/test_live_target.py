"""The live target resolution: the SCHEDULED successor of the last bar.

In training, the target bar's covariates (slot_next / dow_next) came from the next
row of the accepted-bar frame, and the row was only trainable when that successor was
"contiguous" under the training project's rules. Live, the successor has not happened
yet, but its SCHEDULE is deterministic, so the live system derives the same covariates
from the session calendar.

This test pins that contract:
  1. the live successor agrees with the training covariates on ~98% of trainable rows
     (the residual is the historical cases where the next scheduled bar was rejected
     for data quality -- unknowable live, and documented as a risk);
  2. the successor crosses the session boundary correctly (slot 4 -> next session 0);
  3. holidays are skipped when rolling to the next session;
  4. a session gap longer than max_session_gap_days yields no successor.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from ..config import SlotTable
from ..core import calendar_cme
from ..core.slots import next_session_date, scheduled_successor

TRAINING_ROOT = Path(r"D:\DataScience\interm\RED SPRITE\DL-Layer")
FEATS = TRAINING_ROOT / "DATA/PROCESSED/features_targets.parquet"

pytestmark = pytest.mark.skipif(
    not FEATS.exists(), reason="the training project's feature file is not available")


def test_successor_matches_training_covariates():
    """The live successor reproduces the training covariates on ~98% of trainable rows."""
    d = pd.read_parquet(FEATS)
    d["bar_start_utc"] = pd.to_datetime(d["bar_start_utc"], utc=True)
    d = d.sort_values("bar_start_utc", kind="stable").reset_index(drop=True)
    d["trade_date"] = pd.to_datetime(d["trade_date"])

    st = SlotTable()
    closed, early = calendar_cme.cme_calendar(
        int(d["trade_date"].min().year), int(d["trade_date"].max().year) + 1)

    n = len(d)
    ok = tot = 0
    for i in range(n - 1):
        row, nxt = d.iloc[i], d.iloc[i + 1]
        if not bool(nxt["ret_valid"]):
            continue          # the successor was not contiguous -> the row was never trainable
        w = scheduled_successor(row["trade_date"], int(row["slot"]), st,
                                closed=closed, early=early, max_gap_days=5)
        if w is None:
            continue
        tot += 1
        ok += int(w.slot == int(row["slot_next"])
                  and w.trade_date.weekday() == int(row["dow_next"]))
    assert tot > 5000, f"only {tot} trainable rows checked"
    ratio = ok / tot
    assert ratio > 0.95, f"live successor matches the training covariates only {ratio:.3f}"


def test_successor_steps_within_the_session():
    st = SlotTable()
    for slot in range(4):
        w = scheduled_successor(dt.date(2024, 6, 3), slot, st)
        assert w is not None
        assert w.slot == slot + 1
        assert w.trade_date == pd.Timestamp('2024-06-03')


def test_successor_crosses_the_session_boundary():
    st = SlotTable()
    # Monday's slot 4 -> Tuesday's slot 0
    w = scheduled_successor(dt.date(2024, 6, 3), 4, st)
    assert w is not None
    assert w.slot == 0
    assert w.trade_date == pd.Timestamp('2024-06-04')
    # Friday's slot 4 -> Monday's slot 0
    w = scheduled_successor(dt.date(2024, 6, 7), 4, st)
    assert w is not None
    assert w.trade_date == pd.Timestamp('2024-06-10')


def test_successor_skips_a_closed_holiday():
    st = SlotTable()
    closed = {dt.date(2024, 6, 4)}          # pretend Tuesday was a closure
    w = scheduled_successor(dt.date(2024, 6, 3), 4, st, closed=closed)
    assert w is not None
    assert w.trade_date == pd.Timestamp('2024-06-05')


def test_successor_is_none_beyond_the_session_gap_limit():
    st = SlotTable()
    closed = {dt.date(2024, 6, 4), dt.date(2024, 6, 5), dt.date(2024, 6, 6),
              dt.date(2024, 6, 7)}
    assert scheduled_successor(dt.date(2024, 6, 3), 4, st, closed=closed,
                               max_gap_days=5) is None


def test_next_session_date_skips_weekends_and_closures():
    st = SlotTable()
    # Monday -> Tuesday
    assert next_session_date(dt.date(2024, 6, 3), st) == dt.date(2024, 6, 4)
    # Friday -> Monday
    assert next_session_date(dt.date(2024, 6, 7), st) == dt.date(2024, 6, 10)
    # a closure is skipped
    assert next_session_date(dt.date(2024, 6, 3), st,
                             closed={dt.date(2024, 6, 4)}) == dt.date(2024, 6, 5)
