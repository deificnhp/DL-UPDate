"""The feature engine: 11 continuous features + 2 calendar covariates.

This is the live system's copy of the training project's feature layer:
  FE/nq4hq_features/core.py      (causal numerical primitives)
  FE/nq4hq_features/pipeline.py  (compute_all)
  FE/nq4hq_features/config.py    (Config constants)

Every function here is STRICTLY TRAILING: output[t] depends only on bars[0..t]. That is
what makes it safe to (a) seed the recursion with a historical prefix and continue with
live bars, and (b) verify parity against the training feature file.

The feature order is load-bearing -- it must match MODEL/config.py:FEATURES exactly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..config import FeatureConfig
from ..logging_setup import get_logger

log = get_logger("core.features")

FEATURE_ORDER = [
    "r1_z", "r5_z", "r30_z", "gap_z", "lgk_bar", "lrv_d", "lrv_w", "lrv_m",
    "clv", "ema_dist", "rvol",
]
CALENDAR_ORDER = ["slot_next", "dow_next"]

LN2 = np.log(2.0)
DAY_NS = 86_400_000_000_000


# ---------------------------------------------------------------------------
# causal primitives (mirrors FE/nq4hq_features/core.py)
# ---------------------------------------------------------------------------
def garman_klass(o, h, l, c):
    """Garman-Klass per-bar variance: 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2."""
    hl = np.log(h / l)
    co = np.log(c / o)
    return 0.5 * hl * hl - (2.0 * LN2 - 1.0) * co * co


def ewma_recursive(x, alpha, min_periods=1):
    """y_0 = x_0; y_t = (1-alpha)*y_{t-1} + alpha*x_t (pandas adjust=False).

    NaN inputs are skipped and the state carried forward. y is NaN until
    `min_periods` observations have been seen.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.full(x.shape, np.nan)
    state, n = np.nan, 0
    for t in range(x.size):
        xt = x[t]
        if not np.isnan(xt):
            state = xt if n == 0 else (1.0 - alpha) * state + alpha * xt
            n += 1
        if n >= min_periods:
            y[t] = state
    return y


def alpha_from_half_life(h: float) -> float:
    return 1.0 - 0.5 ** (1.0 / h)


def alpha_from_span(span: int) -> float:
    return 2.0 / (span + 1.0)


def trailing(x, w, fn):
    """fn over x[t-w+1..t]; NaN for t < w-1. A NaN inside a window makes it NaN."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    if x.size >= w:
        out[w - 1:] = fn(sliding_window_view(x, w), axis=1)
    return out


def trailing_mean(x, w):
    return trailing(x, w, np.mean)


def trailing_sum(x, w):
    return trailing(x, w, np.sum)


def shift(x, k=1):
    """Lag by k (k>0 = past). Fills with NaN."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def same_slot_prev_median(values, slot, n):
    """For each row t: median of `values` over the previous n rows with the same slot
    (excluding t itself). NaN until n prior same-slot observations exist."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    for k in np.unique(slot):
        idx = np.flatnonzero(slot == k)
        v = values[idx]
        if v.size > n:
            med = np.median(sliding_window_view(v, n), axis=1)   # med[j] = median v[j..j+n-1]
            out[idx[n:]] = med[:-1]                                # row j+n uses v[j..j+n-1]
    return out


# ---------------------------------------------------------------------------
# contiguity / return validity (mirrors pipeline._contiguity)
# ---------------------------------------------------------------------------
def contiguity(bars: pd.DataFrame, cfg: FeatureConfig) -> np.ndarray:
    """contiguous_prev[t] = True iff bar t-1 immediately precedes bar t in the session
    calendar (no missing bar in between).

    Rules (training project, pipeline._contiguity):
      same trade_date  -> slot must step by exactly +1
      new trade_date   -> prev slot == 4 (or prev session flagged) and this slot == 0
                          (or this session flagged), and the trade_date jump is
                          1..max_session_gap_days calendar days
      always           -> bar_start[t] >= bar_end[t-1] (no overlap)
    """
    n = len(bars)
    slot = bars["slot"].to_numpy().astype(np.int64)
    td = pd.to_datetime(bars["trade_date"]).to_numpy().astype("datetime64[D]").astype(np.int64)
    start = pd.to_datetime(bars["bar_start_utc"], utc=True)
    end = pd.to_datetime(bars["bar_end_utc"], utc=True)
    special = np.zeros(n, dtype=bool)
    for c in ("is_early_close", "is_holiday_session", "is_partial_calendar"):
        if c in bars.columns:
            special |= bars[c].to_numpy().astype(bool)
    last = 4  # session_bars - 1 (the 5-slot adaptation)

    cont = np.zeros(n, dtype=bool)
    if n < 2:
        return cont
    p, q = np.arange(n - 1), np.arange(1, n)
    same = td[q] == td[p]
    dd = td[q] - td[p]
    step_ok = slot[q] == slot[p] + 1
    cross_ok = ((slot[p] == last) | special[p]) & ((slot[q] == 0) | special[q]) \
        & (dd >= 1) & (dd <= cfg.max_session_gap_days)
    cont[q] = np.where(same, step_ok, cross_ok) & (start.iloc[q].to_numpy() >= end.iloc[p].to_numpy())
    return cont


# ---------------------------------------------------------------------------
def compute_features(bars: pd.DataFrame,
                     cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Compute every feature column for every row of `bars` (no rows dropped).

    `bars` must contain: bar_start_utc, bar_end_utc, trade_date, slot, contract_id,
    open_mid, high_mid, low_mid, close_mid, volume. Optional: is_early_close,
    is_holiday_session, is_partial_calendar.

    Returns a DataFrame with the SAME ROW ORDER as `bars` (sorted by bar_start_utc),
    carrying the bar identity columns through, plus the 11 continuous features, the 2
    calendar covariates, and the helper columns sigma_ewma / sigma_ewma_prev / r_bar /
    gk / next_valid / bar_seq / target_bar_seq.

    The identity columns are carried through so that a feature frame can be joined
    back to bars (and to the training project's feature file) on bar identity.
    """
    if bars.empty:
        raise ValueError("no bars to compute features for")
    df = bars.reset_index(drop=True)
    order = np.argsort(pd.to_datetime(df["bar_start_utc"], utc=True).to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    n = len(df)

    o = df["open_mid"].to_numpy().astype(np.float64)
    h = df["high_mid"].to_numpy().astype(np.float64)
    l = df["low_mid"].to_numpy().astype(np.float64)
    c = df["close_mid"].to_numpy().astype(np.float64)
    vol = df["volume"].to_numpy().astype(np.float64)
    slot = df["slot"].to_numpy().astype(np.int64)
    contract = df["contract_id"].to_numpy() if "contract_id" in df.columns \
        else np.zeros(n, dtype=object)
    td_days = pd.to_datetime(df["trade_date"]).to_numpy().astype("datetime64[D]").astype(np.int64)

    # --- contiguity & within-contract return validity --------------------------------
    cont = contiguity(df, cfg)
    same_contract = np.zeros(n, dtype=bool)
    same_contract[1:] = contract[1:] == contract[:-1]
    ret_valid = cont & same_contract

    # --- raw returns -------------------------------------------------------------------
    r = np.full(n, np.nan)
    r[1:] = np.log(c[1:] / c[:-1])
    r[~ret_valid] = np.nan
    gap = np.full(n, np.nan)
    gap[1:] = np.log(o[1:] / c[:-1])
    gap[~ret_valid] = np.nan
    r0 = np.nan_to_num(r, nan=0.0)   # breaks contribute zero to sums and to the price index

    # --- volatility state --------------------------------------------------------------
    gk_raw = garman_klass(o, h, l, c)
    gk = np.maximum(gk_raw, cfg.gk_floor)
    ewma = ewma_recursive(gk, alpha_from_half_life(cfg.ewma_half_life), cfg.ewma_min_periods)
    sigma = np.sqrt(ewma)
    sigma_prev = shift(sigma, 1)

    # --- A. return dynamics ------------------------------------------------------------
    r1_z = r0 / sigma_prev
    r5_z = trailing_sum(r0, cfg.win_r_d) / (sigma_prev * np.sqrt(cfg.win_r_d))
    r30_z = trailing_sum(r0, cfg.win_r_w) / (sigma_prev * np.sqrt(cfg.win_r_w))
    r5_z[: cfg.win_r_d] = np.nan
    r30_z[: cfg.win_r_w] = np.nan
    gap_z = np.nan_to_num(gap, nan=0.0) / sigma_prev
    r1_z[0] = gap_z[0] = np.nan

    # --- B. HAR-style log range variance ------------------------------------------------
    lgk_bar = np.log(gk)
    mgk_d = trailing_mean(gk, cfg.win_v_d)
    mgk_w = trailing_mean(gk, cfg.win_v_w)
    mgk_m = trailing_mean(gk, cfg.win_v_m)
    lrv_d, lrv_w, lrv_m = np.log(mgk_d), np.log(mgk_w), np.log(mgk_m)
    lgk_bar = lgk_bar + cfg.var_shift
    lrv_d = lrv_d + cfg.var_shift
    lrv_w = lrv_w + cfg.var_shift
    lrv_m = lrv_m + cfg.var_shift

    # --- C. close location value --------------------------------------------------------
    rng = h - l
    clv = np.where(rng > cfg.range_eps, (2.0 * c - h - l) / np.where(rng > cfg.range_eps, rng, 1.0), 0.0)

    # --- D. trend: distance from EMA120 on a roll-continuous price index ----------------
    log_px = np.cumsum(r0)
    px_idx = np.exp(log_px - log_px[0])
    ema = ewma_recursive(px_idx, alpha_from_span(cfg.ema_span), cfg.ema_span)
    sigma_m = np.sqrt(mgk_m)
    ema_dist = (np.log(px_idx) - np.log(ema)) / (sigma_m * np.sqrt(cfg.ema_span))

    # --- E. relative volume vs same-slot median of the previous 20 sessions -------------
    med_v = same_slot_prev_median(vol, slot, cfg.rvol_sessions)
    rvol = np.log(np.maximum(vol, cfg.vol_floor) / np.maximum(med_v, cfg.vol_floor))
    rvol[np.isnan(med_v)] = np.nan

    # --- F. calendar covariates of the TARGET bar ---------------------------------------
    # slot_next / dow_next describe the CHRONOLOGICAL SUCCESSOR of row t -- the bar whose
    # close-to-close return is the model's target. (The F-3 alignment invariant.)
    next_valid = np.zeros(n, dtype=bool)
    next_valid[:-1] = ret_valid[1:]
    slot_next = np.full(n, -1, dtype=np.int64)
    slot_next[:-1] = slot[1:]
    dow_all = ((td_days + 3) % 7).astype(np.int64)   # 1970-01-01 was a Thursday -> Monday = 0
    dow_next = np.full(n, -1, dtype=np.int64)
    dow_next[:-1] = dow_all[1:]

    r_next = np.full(n, np.nan)
    r_next[:-1] = np.log(c[1:] / c[:-1])
    r_next[~next_valid] = np.nan

    # --- clip ---------------------------------------------------------------------------
    r1_z, r5_z, r30_z, gap_z = [np.clip(x, -cfg.clip, cfg.clip) for x in (r1_z, r5_z, r30_z, gap_z)]
    clv, ema_dist, rvol = [np.clip(x, -cfg.clip, cfg.clip) for x in (clv, ema_dist, rvol)]
    lgk_bar = np.clip(lgk_bar, -cfg.clip, cfg.clip)
    lrv_d = np.clip(lrv_d, -cfg.clip, cfg.clip)
    lrv_w = np.clip(lrv_w, -cfg.clip, cfg.clip)
    lrv_m = np.clip(lrv_m, -cfg.clip, cfg.clip)

    out = pd.DataFrame({
        "interval_id": df["interval_id"].to_numpy() if "interval_id" in df.columns
        else [f"row_{i}" for i in range(n)],
        "bar_start_utc": df["bar_start_utc"].to_numpy(),
        "bar_end_utc": df["bar_end_utc"].to_numpy(),
        "trade_date": df["trade_date"].to_numpy(),
        "calendar_date": df["calendar_date"].to_numpy() if "calendar_date" in df.columns
        else df["trade_date"].to_numpy(),
        "slot": slot,
        "contract_id": contract,
        "bar_seq": np.arange(n, dtype=np.int64),
        "target_bar_seq": np.concatenate([np.arange(1, n, dtype=np.int64), [-1]]),
        "contiguous_prev": cont,
        "ret_valid": ret_valid,
        "gk": gk,
        "r_bar": r,
        "sigma_ewma": sigma,
        "sigma_ewma_prev": sigma_prev,
        "r1_z": r1_z, "r5_z": r5_z, "r30_z": r30_z, "gap_z": gap_z,
        "lgk_bar": lgk_bar, "lrv_d": lrv_d, "lrv_w": lrv_w, "lrv_m": lrv_m,
        "clv": clv, "ema_dist": ema_dist, "rvol": rvol,
        "slot_next": slot_next, "dow_next": dow_next,
        "next_valid": next_valid, "r_next": r_next,
    })
    return out


def training_mask(feat: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Rows usable for a decision: valid successor + all 11 features finite + sane
    calendar covariates. Mirrors pipeline.training_mask."""
    ok = feat["next_valid"].to_numpy().astype(bool)
    for k in FEATURE_ORDER + ["r_next"]:
        ok &= np.isfinite(feat[k].to_numpy().astype(np.float64))
    ok &= (feat["slot_next"].to_numpy() >= 0) & (feat["slot_next"].to_numpy() <= 4)
    ok &= (feat["dow_next"].to_numpy() >= 0) & (feat["dow_next"].to_numpy() <= 4)
    return ok


def feature_matrix(feat: pd.DataFrame) -> np.ndarray:
    """The (n, 11) feature matrix in the model's required column order."""
    missing = [c for c in FEATURE_ORDER if c not in feat.columns]
    if missing:
        raise ValueError(f"feature frame is missing columns: {missing}")
    return feat[FEATURE_ORDER].to_numpy().astype(np.float64)
