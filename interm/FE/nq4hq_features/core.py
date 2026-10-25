"""Causal numerical primitives (pure numpy).

Every function here is strictly *trailing*: output[t] depends only on input[0..t].
Recursive filters are written as explicit sequential loops and rolling statistics use
sliding windows (not cumsum differences), so a prefix of the input always yields a
bit-identical prefix of the output. That property is what the leakage tests check.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

LN2 = np.log(2.0)


def garman_klass(o, h, l, c):
    """GK (1980) per-bar variance: 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2.
    Always >= 0 for valid OHLC since |ln(C/O)| <= ln(H/L)."""
    hl = np.log(h / l)
    co = np.log(c / o)
    return 0.5 * hl * hl - (2.0 * LN2 - 1.0) * co * co


def ewma_recursive(x, alpha, min_periods=1):
    """y_0 = x_0; y_t = (1-alpha)*y_{t-1} + alpha*x_t  (pandas adjust=False).
    NaN inputs are skipped (state carried forward). y is NaN until min_periods obs seen."""
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


def alpha_from_half_life(h):
    return 1.0 - 0.5 ** (1.0 / h)


def alpha_from_span(span):
    return 2.0 / (span + 1.0)


def trailing(x, w, fn):
    """fn over the window x[t-w+1..t]; NaN for t < w-1. A NaN inside a window -> NaN."""
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
