"""The warm-up contract: how much history the engine needs, and what it does without it.

Two properties are pinned here:

  1. the engine needs at least 419 bars before it can produce a forecast, because the
     model's 300-bar window must contain no non-finite feature, and the long
     trailing features (`lrv_m` needs 120 bars, `rvol` needs 20 same-slot sessions)
     are not finite until row 119;
  2. a 419-bar seed reproduces the full-history forecast to ~2e-6, so the recursive
     state handover is correct.

The test needs the training project's clean bar file; it is skipped when it is not
available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ..config import Config
from ..core.features import FEATURE_ORDER
from ..engine import LiveEngine

TRAINING_ROOT = Path(r"D:\DataScience\interm\RED SPRITE\DL-Layer")
CLEAN = TRAINING_ROOT / "DATA/PROCESSED/nq_4h_clean.parquet"

pytestmark = pytest.mark.skipif(
    not CLEAN.exists(), reason="the training project's clean bar file is not available")

MIN_SEED_BARS = 419


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    b = pd.read_parquet(CLEAN)
    for c in ("bar_start_utc", "bar_end_utc"):
        b[c] = pd.to_datetime(b[c], utc=True)
    b["trade_date"] = pd.to_datetime(b["trade_date"]).dt.date
    return b.sort_values("bar_start_utc", ignore_index=True)


def _engine(bars_: pd.DataFrame) -> LiveEngine:
    e = LiveEngine(Config(architectures=("patchtst",)))
    e.load_models()
    e.seed_history_from_bars(bars_)
    return e


def test_a_short_seed_produces_no_prediction(bars):
    """Below the minimum, the engine declines rather than emitting a broken forecast."""
    e = _engine(bars.tail(400))
    pred = e.predict("patchtst")
    assert pred is None, "the engine produced a forecast from a window with NaN features"


def test_the_minimum_seed_produces_a_prediction(bars):
    e = _engine(bars.tail(MIN_SEED_BARS))
    pred = e.predict("patchtst")
    assert pred is not None, "the engine declined with a sufficient seed"
    assert np.isfinite(pred.quantiles).all()


def test_the_minimum_seed_matches_the_full_history(bars):
    """The recursive state handover is correct: a sufficient seed reproduces the
    full-history forecast to ~2e-6 (the residual is the EMA-120 price index anchor)."""
    full = _engine(bars).predict("patchtst")
    seeded = _engine(bars.tail(MIN_SEED_BARS)).predict("patchtst")
    assert full is not None and seeded is not None
    assert full.input_bar_end_utc == seeded.input_bar_end_utc
    assert full.target_bar_start_utc == seeded.target_bar_start_utc
    err = float(np.max(np.abs(np.array(full.quantiles) - np.array(seeded.quantiles))))
    assert err < 1e-4, f"the seeded forecast differs from the full-history one by {err:.3e}"


def test_the_minimum_is_what_the_warm_up_needs(bars):
    """Pin the exact boundary: 418 bars is not enough, 419 is."""
    for n, expect_prediction in ((MIN_SEED_BARS - 1, False), (MIN_SEED_BARS, True)):
        e = _engine(bars.tail(n))
        f = e.compute_features()
        win = f.iloc[len(f) - 300:][FEATURE_ORDER].to_numpy(dtype=float)
        finite = bool(np.isfinite(win).all())
        assert finite == expect_prediction, (
            f"{n} bars: the window is {'finite' if finite else 'not finite'}, "
            f"expected the opposite")
        assert (e.predict("patchtst") is not None) == expect_prediction
