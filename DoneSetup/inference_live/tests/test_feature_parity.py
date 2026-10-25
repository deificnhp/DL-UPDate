"""Feature parity: the live feature engine vs the training project's feature file.

The training project's `DATA/PROCESSED/features_targets.parquet` is the ground truth:
it was produced by FE/nq4hq_features/pipeline.py from `nq_4h_clean.parquet`. The live
engine's `core/features.py` is an independent re-implementation, so the two must agree
bit-for-bit (or at worst to floating-point associativity) on every row.

The test needs the training project; it is skipped when it is not available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ..config import FeatureConfig
from ..core.features import FEATURE_ORDER, compute_features, feature_matrix

TRAINING_ROOT = Path(r"D:\DataScience\interm\RED SPRITE\DL-Layer")
CLEAN = TRAINING_ROOT / "DATA/PROCESSED/nq_4h_clean.parquet"
FEATS = TRAINING_ROOT / "DATA/PROCESSED/features_targets.parquet"

pytestmark = pytest.mark.skipif(
    not (CLEAN.exists() and FEATS.exists()),
    reason="the training project's processed files are not available")


def _clean_bars() -> pd.DataFrame:
    b = pd.read_parquet(CLEAN)
    b["bar_start_utc"] = pd.to_datetime(b["bar_start_utc"], utc=True)
    b["bar_end_utc"] = pd.to_datetime(b["bar_end_utc"], utc=True)
    b["trade_date"] = pd.to_datetime(b["trade_date"]).dt.date
    return b


def test_features_match_the_training_file():
    bars = _clean_bars()
    live = compute_features(bars, FeatureConfig())

    ref = pd.read_parquet(FEATS)
    ref["bar_start_utc"] = pd.to_datetime(ref["bar_start_utc"], utc=True)
    ref = ref.sort_values("bar_start_utc", kind="stable").reset_index(drop=True)

    # the training file is the filtered subset of the clean file; align on bar identity
    key = ["bar_start_utc"]
    merged = live.merge(ref, on=key, how="inner", suffixes=("", "_ref"))
    assert len(merged) > 1000, f"only {len(merged)} rows aligned"

    for col in FEATURE_ORDER:
        a = merged[col].to_numpy(dtype=float)
        b = merged[f"{col}_ref"].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        assert both_nan.all() or np.allclose(a[~both_nan], b[~both_nan], rtol=0, atol=1e-12), \
            f"feature {col} differs from the training file"

    # the calendar covariates describe the TARGET bar -- verify against the training file
    for col in ("slot_next", "dow_next"):
        a = merged[col].to_numpy()
        b = merged[f"{col}_ref"].to_numpy()
        assert (a == b).all(), f"covariate {col} differs from the training file"


def test_sigma_ewma_matches():
    bars = _clean_bars()
    live = compute_features(bars, FeatureConfig())
    ref = pd.read_parquet(FEATS)
    ref["bar_start_utc"] = pd.to_datetime(ref["bar_start_utc"], utc=True)
    merged = live.merge(ref, on=["bar_start_utc"], how="inner", suffixes=("", "_ref"))
    a = merged["sigma_ewma"].to_numpy(dtype=float)
    b = merged["sigma_ewma_ref"].to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    assert both_nan.all() or np.allclose(a[~both_nan], b[~both_nan], rtol=0, atol=1e-12)


def test_r_next_matches():
    """r_next is the label the model was trained on: ln(C_{t+1}/C_t)."""
    bars = _clean_bars()
    live = compute_features(bars, FeatureConfig())
    ref = pd.read_parquet(FEATS)
    ref["bar_start_utc"] = pd.to_datetime(ref["bar_start_utc"], utc=True)
    merged = live.merge(ref, on=["bar_start_utc"], how="inner", suffixes=("", "_ref"))
    a = merged["r_next"].to_numpy(dtype=float)
    b = merged["r_next_ref"].to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    assert both_nan.all() or np.allclose(a[~both_nan], b[~both_nan], rtol=0, atol=1e-12)


def test_features_are_strictly_trailing():
    """Recomputing from a prefix must reproduce the prefix of the full computation.

    This is the property that makes the live warm-up safe: the recursive state (EWMA,
    EMA120, same-slot volume median) can be carried from a historical prefix.
    """
    bars = _clean_bars()
    full = compute_features(bars, FeatureConfig())
    for cut in (500, 1500, 3000):
        prefix = compute_features(bars.iloc[:cut].copy(), FeatureConfig())
        a = feature_matrix(full.iloc[:cut])
        b = feature_matrix(prefix)
        both_nan = np.isnan(a) & np.isnan(b)
        assert both_nan.all() or np.allclose(a[~both_nan], b[~both_nan], rtol=0, atol=1e-12), \
            f"prefix of length {cut} is not reproducible"
