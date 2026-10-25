"""NQ4H-Q feature & target pipeline.

Timing convention (PLAN.md A3): a row t is "the decision at the close of bar t".
  * features[t] use bars <= t only
  * sigma_ewma[t]      = sqrt(EWMA_t(GK))      -> forecast of vol of bar t+1 (known at close t)
  * sigma_ewma_prev[t] = sigma_ewma[t-1]       -> forecast of vol of bar t (used to scale r_t)
  * r_next[t] = ln(C_{t+1}/C_t)  (label; the ONLY column that looks forward, plus the two
    known-future calendar covariates slot_next/dow_next which are deterministic schedule facts)

ALIGNMENT INVARIANT (F-3): the label `r_next[t]` and the two known-future calendar
covariates `slot_next[t]` / `dow_next[t]` all describe ONE AND THE SAME bar -- the
chronological successor of row t in the *source* (clean) frame, whose identity is recorded
in `target_bar_seq[t]`.  They must never be re-pointed at "the next row that happens to
survive filtering": a row whose successor was dropped from the output would then carry a
label from one bar and covariates from a different bar.  Filtering therefore never changes
the meaning of these columns; it only removes rows.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import core
from .config import CFG, Config, FEATURES_CONT, FEATURES_CAL, REQUIRED_COLUMNS
from .io import Frame

log = logging.getLogger("nq4hq.features")
DAY_NS = 86_400_000_000_000

# Engineered non-feature helper columns written alongside features (documented in dictionary)
HELPER_COLS = ["bar_seq", "target_bar_seq", "contiguous_prev", "ret_valid", "gk", "r_bar",
               "sigma_ewma", "sigma_ewma_prev"]


# ---------------------------------------------------------------------------
def _validate(f: Frame, cfg: Config):
    missing = [c for c in REQUIRED_COLUMNS if c not in f.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    slot = f.np("slot").astype(np.int64)
    if slot.min() < 0 or slot.max() > cfg.session_bars - 1:
        raise ValueError(f"slot outside 0..{cfg.session_bars-1}: [{slot.min()}, {slot.max()}] "
                         "(adaptation assumes slot 5 does not exist)")
    if f.np_bool("is_terminal_3h").any():
        raise ValueError("is_terminal_3h bars present: 5-bar adaptation invalid, raise a CR")
    o, h, l, c = (f.np(k).astype(np.float64) for k in ("open_mid", "high_mid", "low_mid", "close_mid"))
    bad = ~((l > 0) & (h >= l) & (h >= np.maximum(o, c)) & (l <= np.minimum(o, c)))
    if bad.any():
        raise ValueError(f"{bad.sum()} rows with inconsistent OHLC")
    start = f.np_time_ns("bar_start_utc")
    if np.unique(start).size != start.size:
        raise ValueError("duplicate bar_start_utc")


def _contiguity(f: Frame, cfg: Config):
    """contiguous_prev[t] = True iff row t-1 is the bar immediately preceding bar t in the
    session calendar (no missing bar in between). Rules:
      same trade_date      -> slot must step by exactly +1
      new trade_date       -> prev slot == 4 (or prev session flagged early-close/holiday/partial)
                              and this slot == 0 (or this session flagged), and the trade_date
                              jump is 1..max_session_gap_days calendar days
      always               -> bar_start[t] >= bar_end[t-1] (no overlap)"""
    n = len(f)
    slot = f.np("slot").astype(np.int64)
    td = f.np_time_ns("trade_date") // DAY_NS
    start, end = f.np_time_ns("bar_start_utc"), f.np_time_ns("bar_end_utc")
    special = f.np_bool("is_early_close") | f.np_bool("is_holiday_session") | f.np_bool("is_partial_calendar")
    last = cfg.session_bars - 1

    cont = np.zeros(n, dtype=bool)
    p, q = np.arange(n - 1), np.arange(1, n)
    same = td[q] == td[p]
    dd = td[q] - td[p]
    step_ok = slot[q] == slot[p] + 1
    cross_ok = ((slot[p] == last) | special[p]) & ((slot[q] == 0) | special[q]) & (dd >= 1) & (dd <= cfg.max_session_gap_days)
    cont[q] = np.where(same, step_ok, cross_ok) & (start[q] >= end[p])
    return cont


# ---------------------------------------------------------------------------
def compute_all(df, cfg: Config = CFG) -> Frame:
    """Compute every feature/target column for every row (no rows dropped).
    Exposed separately so leakage tests can compare un-dropped outputs."""
    f = Frame.from_any(df) if not isinstance(df, Frame) else df
    _validate(f, cfg)
    order = np.argsort(f.np_time_ns("bar_start_utc"), kind="stable")
    f = f.take(order)
    n = len(f)

    o, h, l, c = (f.np(k).astype(np.float64) for k in ("open_mid", "high_mid", "low_mid", "close_mid"))
    vol = f.np("volume").astype(np.float64)
    slot = f.np("slot").astype(np.int64)
    contract = f.np("contract_id")
    td_days = f.np_time_ns("trade_date") // DAY_NS

    # --- contiguity & within-contract return validity --------------------------------
    cont = _contiguity(f, cfg)
    same_contract = np.zeros(n, dtype=bool)
    same_contract[1:] = contract[1:] == contract[:-1]
    ret_valid = cont & same_contract          # r_t = ln(C_t/C_{t-1}) is a true 1-bar, 1-contract return

    # --- raw returns (NaN where invalid: roll boundary or missing bar) --------------
    r = np.full(n, np.nan)
    r[1:] = np.log(c[1:] / c[:-1])
    r[~ret_valid] = np.nan
    gap = np.full(n, np.nan)
    gap[1:] = np.log(o[1:] / c[:-1])
    gap[~ret_valid] = np.nan
    r0 = np.nan_to_num(r, nan=0.0)            # breaks contribute zero to sums/price index

    # --- volatility state -------------------------------------------------------------
    gk_raw = core.garman_klass(o, h, l, c)
    gk = np.maximum(gk_raw, cfg.gk_floor)
    ewma = core.ewma_recursive(gk, core.alpha_from_half_life(cfg.ewma_half_life), cfg.ewma_min_periods)
    sigma = np.sqrt(ewma)                     # sigma_hat_t (pre-seasonal), forecast for bar t+1
    sigma_prev = core.shift(sigma, 1)         # sigma_hat_{t-1}, forecast for bar t

    # --- A. return dynamics -------------------------------------------------------------
    r1_z = r0 / sigma_prev
    r5_z = core.trailing_sum(r0, cfg.win_r_d) / (sigma_prev * np.sqrt(cfg.win_r_d))
    r30_z = core.trailing_sum(r0, cfg.win_r_w) / (sigma_prev * np.sqrt(cfg.win_r_w))
    r5_z[: cfg.win_r_d] = np.nan              # need r_{t-w+1..t}, r_0 is undefined
    r30_z[: cfg.win_r_w] = np.nan
    gap_z = np.nan_to_num(gap, nan=0.0) / sigma_prev
    r1_z[0] = gap_z[0] = np.nan

    # --- B. HAR-style log range variance -------------------------------------------------
    lgk_bar = np.log(gk)
    mgk_d = core.trailing_mean(gk, cfg.win_v_d)
    mgk_w = core.trailing_mean(gk, cfg.win_v_w)
    mgk_m = core.trailing_mean(gk, cfg.win_v_m)
    lrv_d, lrv_w, lrv_m = np.log(mgk_d), np.log(mgk_w), np.log(mgk_m)

    # Shift log-variance features so their natural range fits within [-5, 5] after clipping.
    # lgk_bar (single bar) has wider range ~[-14, -5] due to outliers; lrv_d/w/m are narrower ~[-14, -8].
    # VAR_SHIFT=9.0 brings lgk_bar to ~[-5, 4] and lrv_d/w/m to ~[-5, 1] - all within ±5 after clip.
    VAR_SHIFT = 9.0
    lgk_bar = lgk_bar + VAR_SHIFT
    lrv_d = lrv_d + VAR_SHIFT
    lrv_w = lrv_w + VAR_SHIFT
    lrv_m = lrv_m + VAR_SHIFT

    # --- C. close location value ---------------------------------------------------------
    rng = h - l
    clv = np.where(rng > cfg.range_eps, (2.0 * c - h - l) / np.where(rng > cfg.range_eps, rng, 1.0), 0.0)

    # --- D. trend: distance from EMA120 on a roll-continuous price index ----------------
    # Ratio back-adjustment makes ln(adjusted price) = cumsum(within-contract returns) + const.
    # ln C - ln EMA(C) is invariant to that multiplicative constant, so building the index
    # causally from r0 is identical to using a ratio back-adjusted series, without the
    # back-adjustment's dependence on *future* roll ratios.
    log_px = np.cumsum(r0)
    px_idx = np.exp(log_px - log_px[0])
    ema = core.ewma_recursive(px_idx, core.alpha_from_span(cfg.ema_span), cfg.ema_span)
    sigma_m = np.sqrt(mgk_m)                  # monthly GK vol (PLAN: sigma_m)
    ema_dist = (np.log(px_idx) - np.log(ema)) / (sigma_m * np.sqrt(cfg.ema_span))

    # --- E. relative volume vs same-slot median of previous 20 sessions ------------------
    med_v = core.same_slot_prev_median(vol, slot, cfg.rvol_sessions)
    rvol = np.log(np.maximum(vol, cfg.vol_floor) / np.maximum(med_v, cfg.vol_floor))
    rvol[np.isnan(med_v)] = np.nan

    # --- F. chronological next-row calendar covariates + target -------------------------
    # These are schedule facts for the actual next record, not arithmetic increments.
    # This preserves slot/dow across weekends, holidays, missing bars and session gaps.
    # All three (r_next, slot_next, dow_next) refer to the SAME bar: the chronological
    # successor of row t in this (unfiltered) frame, identified by target_bar_seq[t].
    # The successor's bar_seq is bar_seq[t] + 1 because bar_seq enumerates this frame
    # in chronological order; the last row has no successor (-> -1).
    next_valid = np.zeros(n, dtype=bool)
    next_valid[:-1] = ret_valid[1:]           # last row remains invalid by construction
    slot_next = np.full(n, -1, dtype=np.int64)
    slot_next[:-1] = slot[1:]                 # actual chronological row t+1
    dow_all = ((td_days + 3) % 7).astype(np.int64)   # 1970-01-01 = Thursday -> Monday = 0
    dow_next = np.full(n, -1, dtype=np.int64)
    dow_next[:-1] = dow_all[1:]               # actual trade_date[t+1].weekday()

    r_next = np.full(n, np.nan)
    r_next[:-1] = np.log(c[1:] / c[:-1])
    r_next[~next_valid] = np.nan
    z_next = r_next / sigma

    # Identity of the target bar (the successor row) -- recorded explicitly so that every
    # downstream consumer can resolve the target by identity instead of by file adjacency.
    bar_seq = np.arange(n, dtype=np.int64)
    target_bar_seq = np.full(n, -1, dtype=np.int64)
    target_bar_seq[:-1] = bar_seq[1:]

    # Strict final guard: every continuous model input is bounded before persistence.
    # This is intentionally after all feature calculations and before Frame construction.
    # Return features (r1_z, r5_z, r30_z, gap_z) and other unbounded features (clv, ema_dist, rvol)
    # are clipped to [-5, 5]. Log-variance features (lgk_bar, lrv_d, lrv_w, lrv_m) are shifted
    # by VAR_SHIFT and also clipped to [-5, 5] to match audit requirements.
    r1_z, r5_z, r30_z, gap_z = [np.clip(x, -5.0, 5.0) for x in (r1_z, r5_z, r30_z, gap_z)]
    clv, ema_dist, rvol = [np.clip(x, -5.0, 5.0) for x in (clv, ema_dist, rvol)]
    lgk_bar = np.clip(lgk_bar, -5.0, 5.0)
    lrv_d = np.clip(lrv_d, -5.0, 5.0)
    lrv_w = np.clip(lrv_w, -5.0, 5.0)
    lrv_m = np.clip(lrv_m, -5.0, 5.0)

    cols = {
        "bar_seq": bar_seq,
        "target_bar_seq": target_bar_seq,
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
        "next_valid": next_valid,
        "r_next": r_next, "z_next_ewma": z_next,
    }
    return f.with_columns(cols)


def training_mask(f: Frame) -> np.ndarray:
    """Rows usable for training: valid next bar + all 11 continuous features finite
    + sane calendar covariates."""
    ok = f.np("next_valid").astype(bool)
    for k in FEATURES_CONT + ["z_next_ewma", "r_next"]:
        ok &= np.isfinite(f.np(k).astype(np.float64))
    ok &= (f.np("slot_next") >= 0) & (f.np("slot_next") <= CFG.session_bars - 1)
    ok &= (f.np("dow_next") >= 0) & (f.np("dow_next") <= 4)
    return ok


def build(src="DATA/PROCESSED/nq_4h_clean.parquet",
          dst="DATA/PROCESSED/features_targets.parquet",
          cfg: Config = CFG, prefer_polars=True) -> Frame:
    f = Frame.read(src, prefer_polars=prefer_polars)
    out = compute_all(f, cfg)
    n0 = len(out)
    nv = out.np("next_valid").astype(bool)
    m = training_mask(out)
    dow_bad = int(((out.np("dow_next") > 4) & nv).sum())
    if dow_bad:
        log.warning("%d valid rows have a weekend target trade_date; check trade_date convention", dow_bad)

    # Filtering must NOT change the meaning of r_next / slot_next / dow_next.
    # Those three columns are computed in compute_all() against the chronological successor
    # in the SOURCE frame and are carried through the filter untouched, so they always
    # describe the SAME bar (the one identified by target_bar_seq). Re-deriving them on the
    # filtered frame would silently re-point the covariates of a row whose successor was
    # dropped at a different bar than its label -- that is the F-3 misalignment.
    out = out.filter(m)

    # Post-filter self-check: for every retained row whose target bar is also retained,
    # the covariates must match that target bar exactly.
    n = len(out)
    bar_seq = out.np("bar_seq").astype(np.int64)
    target_bar_seq = out.np("target_bar_seq").astype(np.int64)
    has_target = np.zeros(n, dtype=bool)
    if n:
        valid_tgt = target_bar_seq >= 0
        if valid_tgt.any():
            tv = target_bar_seq[valid_tgt]
            pos = np.searchsorted(bar_seq, tv)      # bar_seq is strictly increasing
            pos_c = np.clip(pos, 0, n - 1)
            ok = (pos < n) & (bar_seq[pos_c] == tv)
            has_target[valid_tgt] = ok
            if ok.any():
                slot_next = out.np("slot_next").astype(np.int64)
                dow_next = out.np("dow_next").astype(np.int64)
                slot_all = out.np("slot").astype(np.int64)
                td_days = out.np_time_ns("trade_date") // DAY_NS
                dow_all = ((td_days + 3) % 7).astype(np.int64)
                sel = np.where(valid_tgt)[0][ok]
                t_idx = pos_c[ok]
                if not np.array_equal(slot_next[sel], slot_all[t_idx]):
                    raise ValueError("ALIGNMENT FAIL: slot_next != target bar slot")
                if not np.array_equal(dow_next[sel], dow_all[t_idx]):
                    raise ValueError("ALIGNMENT FAIL: dow_next != target bar weekday")
    n_missing_target = int((~has_target).sum())
    log.info("rows in=%d | no valid next bar=%d | warm-up/invalid features=%d | out=%d",
             n0, int((~nv).sum()), int((nv & ~m).sum()), len(out))
    log.info("rows whose target bar is not retained in the output: %d (labels remain valid; "
             "the target bar is identified by target_bar_seq)", n_missing_target)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out.write(dst)
    return out
