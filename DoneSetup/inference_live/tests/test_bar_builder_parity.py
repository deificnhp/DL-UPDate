"""Bar-builder parity against the training project, on the REAL vendor minutes.

This is the strongest available check of the live bar builder: it runs the live
`build_bars` on the training project's raw BID/ASK minute CSVs and requires the
result to match the training project's `nq_4h_clean.parquet` (accepted bars) and
`nq_4h_exclusions.parquet` (rejected bars + reasons) exactly.

It needs the training project's raw data (636 MB of minute CSVs), so it is skipped
when they are not available. Run it with:

    python -m pytest inference_live/tests/test_bar_builder_parity.py -q -s
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ..config import FeatureConfig, SlotTable
from ..core import bar_builder

TRAINING_ROOT = Path(r"D:\DataScience\interm\RED SPRITE\DL-Layer")
RAW = TRAINING_ROOT / "DATA/RAW"
CLEAN = TRAINING_ROOT / "DATA/PROCESSED/nq_4h_clean.parquet"
EXCL = TRAINING_ROOT / "DATA/PROCESSED/nq_4h_exclusions.parquet"

pytestmark = pytest.mark.skipif(
    not (RAW.exists() and CLEAN.exists() and EXCL.exists()),
    reason="the training project's raw minute data is not available")


def _read_side(path: Path, side: str) -> pd.DataFrame:
    """The training project's `_read_side`, copied verbatim in behaviour."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    tcol = next(c for c in df.columns if c.lower().startswith("time"))
    raw = df[tcol].astype(str).str.replace(r"\s*(UTC|GMT)$", "", regex=True).str.strip()
    ts = None
    for fmt in ("%Y.%m.%d %H:%M:%S.%f", "%Y.%m.%d %H:%M:%S", "%d.%m.%Y %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            ts = pd.to_datetime(raw, format=fmt, utc=True)
            break
        except (ValueError, TypeError):
            pass
    if ts is None:
        ts = pd.to_datetime(raw, format="mixed", utc=True)
    ren = {c: f"{side}_{c.lower()}" for c in ("Open", "High", "Low", "Close", "Volume")}
    out = df.rename(columns=ren)[list(ren.values())].astype("float64")
    out.insert(0, "ts", ts)
    return out.drop_duplicates("ts", keep="last").sort_values("ts", ignore_index=True)


def _load_minutes() -> pd.DataFrame:
    b = _read_side(RAW / "BID.csv", "bid")
    a = _read_side(RAW / "ASK.csv", "ask")
    m = b.merge(a, on="ts", how="inner", validate="one_to_one").sort_values("ts", ignore_index=True)
    crossed = (m["ask_close"] < m["bid_close"]) | (m["ask_open"] < m["bid_open"])
    m = m.loc[~crossed].reset_index(drop=True)
    m["volume"] = m["bid_volume"] + m["ask_volume"]
    return m


@pytest.fixture(scope="module")
def live_bars():
    m = _load_minutes()
    res = bar_builder.build_bars(m, SlotTable(), FeatureConfig())
    return res


def test_live_builder_matches_the_training_clean_bars(live_bars):
    got = live_bars.bars
    ref = pd.read_parquet(CLEAN)
    ref["bar_start_utc"] = pd.to_datetime(ref["bar_start_utc"], utc=True)
    ref["bar_end_utc"] = pd.to_datetime(ref["bar_end_utc"], utc=True)

    assert len(got) == len(ref), (
        f"the live builder accepted {len(got)} bars, the training pipeline accepted "
        f"{len(ref)}")

    g = got.set_index("bar_start_utc").sort_index()
    r = ref.set_index("bar_start_utc").sort_index()
    assert list(g.index) == list(r.index), "the accepted bars are not the same bars"

    for col in ("open_mid", "high_mid", "low_mid", "close_mid", "volume"):
        a = g[col].to_numpy(dtype=float)
        b = r[col].to_numpy(dtype=float)
        err = float(np.max(np.abs(a - b)))
        assert err < 1e-9, f"{col} differs by up to {err:.3e}"

    for col in ("slot", "observed_minutes", "quote_coverage", "expected_minutes"):
        assert (g[col].to_numpy() == r[col].to_numpy()).all(), f"{col} differs"


def test_live_builder_matches_the_training_exclusion_reasons(live_bars):
    got = live_bars.rejected
    ref = pd.read_parquet(EXCL)
    ref["bar_start_utc"] = pd.to_datetime(ref["bar_start_utc"], utc=True)

    # the training pipeline trims the grid padding outside the vendor data span
    span = (ref["bar_end_utc"] > live_bars.minutes["ts"].min()) & \
           (ref["bar_start_utc"] <= live_bars.minutes["ts"].max())
    ref = ref.loc[span]

    g = got.set_index("bar_start_utc").sort_index()
    r = ref.set_index("bar_start_utc").sort_index()
    common = g.index.intersection(r.index)
    assert len(common) == len(r), (
        f"the live builder rejected {len(g)} bars; the training pipeline rejected "
        f"{len(r)}; {len(r) - len(common)} of the training rejections are missing "
        f"from the live rejections")

    a = g.loc[common, "rejection_reason"].to_numpy()
    b = r.loc[common, "rejection_reason"].to_numpy()
    bad = a != b
    assert not bad.any(), (
        f"{int(bad.sum())} bars were rejected for a different reason; e.g. "
        f"{list(zip(common[bad][:5], a[bad][:5], b[bad][:5]))}")


def test_live_builder_produces_the_same_accepted_bar_count(live_bars):
    """The accepted set and the rejected set partition the grid, and every grid row
    is a (weekday trade_date, slot) pair -- five slots per weekday trade date."""
    n_grid = len(live_bars.bars) + len(live_bars.rejected)
    assert n_grid % 5 == 0, f"{n_grid} grid rows is not a multiple of 5"
    keys = set()
    for frame in (live_bars.bars, live_bars.rejected):
        for td, slot in zip(frame["trade_date"], frame["slot"]):
            keys.add((pd.Timestamp(td), int(slot)))
    assert len(keys) == n_grid, "the grid rows are not unique (trade_date, slot) pairs"
    for td, slot in keys:
        assert pd.Timestamp(td).weekday() < 5, f"{td} is not a weekday"
