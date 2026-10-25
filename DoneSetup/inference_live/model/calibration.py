"""Calibration: the trained CQR offsets and median bias, applied live.

This is the live system's copy of MODEL/calibration.py, reduced to what a live
inference needs. Deliberately EXCLUDED: `signals`, `theta`, `pup`, `score` and the
isotonic confidence -- those are trading logic (signal generation, confidence) and are
out of scope for this system. Only the quantile post-processing is reproduced.

What is reproduced exactly (the offline `Calibrator.apply(..., online=False)` path):
    out[..., 3] += median_bias                       # shift the median
    for (lo, hi, alpha) in [(0,6,0.10), (1,5,0.20), (2,4,0.50)]:
        off = cqr_offset(score_j, alpha)
        out[..., lo] -= off
        out[..., hi] += off
    out = sort(out, axis=-1)

where `cqr_offset` is the finite-sample conformal quantile of the calibration scores.

The ACI (adaptive conformal inference) path is available as an option: it advances the
three alpha values using realized outcomes, exactly as the training project did with
gamma = 0.005. The alpha vector can be seeded from the fold's persisted
`aci_alpha_after_observation`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# (lower index, upper index, nominal alpha) -- matches MODEL/calibration.py:PAIRS
PAIRS = [(0, 6, 0.10), (1, 5, 0.20), (2, 4, 0.50)]


def cqr_offset(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample conformal offset: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        return 0.0
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(s[np.clip(rank - 1, 0, n - 1)])


@dataclass
class StaticCalibrator:
    """The frozen calibration: a median bias and three CQR offsets.

    Built from the trained fold's provenance file:
        calibration.median_bias
        calibration.cqr_offsets   (3 values, in PAIRS order)
    """
    median_bias: float
    cqr_offsets: Sequence[float]
    gamma: float = 0.005

    @property
    def static(self) -> "StaticCalibrator":
        """Self-reference, so the static and ACI calibrators share one interface."""
        return self

    @staticmethod
    def from_metadata(calibration: dict) -> "StaticCalibrator":
        if "median_bias" not in calibration:
            raise ValueError("calibration metadata is missing 'median_bias'")
        offsets = calibration.get("cqr_offsets")
        if offsets is None or len(offsets) != len(PAIRS):
            raise ValueError(
                f"calibration metadata must carry {len(PAIRS)} cqr_offsets, got {offsets!r}")
        return StaticCalibrator(
            median_bias=float(calibration["median_bias"]),
            cqr_offsets=[float(v) for v in offsets],
            gamma=float(calibration.get("gamma", 0.005)),
        )

    def apply(self, member_q: np.ndarray) -> np.ndarray:
        """Calibrate per-member quantiles.

        `member_q` has shape (..., 7): the per-member quantile vector(s).
        Returns the calibrated, sorted quantiles in the same shape.
        """
        q = np.asarray(member_q, dtype=np.float64).copy()
        if q.shape[-1] != 7:
            raise ValueError(f"expected 7 quantiles on the last axis, got {q.shape[-1]}")
        q[..., 3] += self.median_bias
        for (lo, hi, _), off in zip(PAIRS, self.cqr_offsets):
            q[..., lo] -= off
            q[..., hi] += off
        return np.sort(q, axis=-1)


@dataclass
class ACICalibrator:
    """Static calibration plus an adaptive conformal alpha vector.

    The alpha vector advances by `gamma * (nominal - observed_error)` after each
    realized outcome, clipped to [1/(n+1), 0.999]. `n` is the number of calibration
    scores available for that pair, which the training project did not persist -- the
    clip bound is therefore approximated by 1/101 (the training calibration split had
    ~100-1000 rows; this only matters in the extreme tail of the alpha range).
    """
    static: StaticCalibrator
    alpha: np.ndarray
    score_pool: Optional[np.ndarray] = None   # (3, m) calibration scores, if available

    @staticmethod
    def from_metadata(calibration: dict,
                      aci_alpha: Optional[Sequence[float]] = None,
                      gamma: float = 0.005) -> "ACICalibrator":
        static = StaticCalibrator.from_metadata(calibration)
        alpha = np.asarray(aci_alpha if aci_alpha is not None
                           else [a for _, _, a in PAIRS], dtype=np.float64).copy()
        if alpha.shape != (len(PAIRS),):
            raise ValueError(f"aci alpha must have {len(PAIRS)} entries, got {alpha.shape}")
        return ACICalibrator(static=static, alpha=alpha)

    def apply(self, member_q: np.ndarray,
              realized: Optional[np.ndarray] = None) -> np.ndarray:
        """Calibrate and (optionally) advance the alpha state with a realized outcome.

        `realized` is the observed log return of the target bar for the SAME decision
        being calibrated, or None when the outcome is not yet known. Passing an
        outcome updates the alpha vector for SUBSEQUENT decisions.
        """
        q = np.asarray(member_q, dtype=np.float64).copy()
        if q.shape[-1] != 7:
            raise ValueError(f"expected 7 quantiles on the last axis, got {q.shape[-1]}")
        q[..., 3] += self.static.median_bias
        for j, ((lo, hi, nominal), off) in enumerate(zip(PAIRS, self.static.cqr_offsets)):
            # the offset is recomputed against the CURRENT alpha, using the frozen
            # offset as the base when no score pool is available
            o = off
            if self.score_pool is not None:
                o = cqr_offset(self.score_pool[j], self.alpha[j])
            q[..., lo] -= o
            q[..., hi] += o
        q = np.sort(q, axis=-1)

        if realized is not None:
            y = float(np.asarray(realized).reshape(-1)[0])
            pooled = q.mean(axis=tuple(range(q.ndim - 1))) if q.ndim > 1 else q
            for j, (lo, hi, nominal) in enumerate(PAIRS):
                err = float(y < pooled[lo] or y > pooled[hi])
                self.alpha[j] = float(np.clip(
                    self.alpha[j] + self.static.gamma * (nominal - err), 1.0 / 101.0, 0.999))
        return q
