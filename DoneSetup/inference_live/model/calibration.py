"""Calibration: the trained CQR offsets and median bias, applied live, plus Online ACI.

This is the live system's copy of MODEL/calibration.py, reduced to what live
inference needs. Only the quantile post-processing and ACI width tracking are reproduced.

What is reproduced exactly (the offline `Calibrator.apply(..., online=False)` path):
    out[..., 3] += median_bias                       # shift the median
    for (lo, hi, alpha) in [(0,6,0.10), (1,5,0.20), (2,4,0.50)]:
        off = cqr_offset(score_j, alpha)
        out[..., lo] -= off
        out[..., hi] += off
    out = sort(out, axis=-1)

where `cqr_offset` is the finite-sample conformal quantile of the calibration scores.

In addition:
    ACISymmetricWidthTracker (Mode A):
      - q50 (q[3]) untouched (preserves median, no directional bias).
      - For levels k in {50%, 80%, 90%}, pairs (2,4), (1,5), (0,6):
          \tilde{q}_\tau = q_50 + exp(phi_k) * (q_\tau - q_50)
      - Update rule after realized outcome:
          phi_k <- clip(phi_k + eta * (miss_k - (1 - level_k)), -ln(1.5), +ln(1.5))
      - Persists state in DoneSetup/ops_store/aci_state.json.
      - Supports 'shadow' and 'active' modes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

# (lower index, upper index, nominal alpha) -- matches MODEL/calibration.py:PAIRS
PAIRS = [(0, 6, 0.10), (1, 5, 0.20), (2, 4, 0.50)]

# Central interval definitions for ACISymmetricWidthTracker:
# level: (pair_index, (lo, hi))
# 50% interval: pair (2, 4)
# 80% interval: pair (1, 5)
# 90% interval: pair (0, 6)
INTERVAL_LEVELS = [
    {"level": 0.50, "lo": 2, "hi": 4, "key": "50"},
    {"level": 0.80, "lo": 1, "hi": 5, "key": "80"},
    {"level": 0.90, "lo": 0, "hi": 6, "key": "90"},
]

DEFAULT_ACI_STATE_PATH = Path(__file__).resolve().parents[2] / "ops_store" / "aci_state.json"


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
class ACISymmetricWidthTracker:
    """Online ACI Symmetric Width Tracker — Tier 2 (Mode A).

    - Leaves q50 (q[3]) untouched.
    - Tracks phi_k for intervals k in {50%, 80%, 90%}.
    - Scales intervals around q50:
        \tilde{q}_\tau = q_50 + exp(phi_k) * (q_\tau - q_50)
    - Updates phi_k:
        phi_k <- clip(phi_k + eta * (miss_k - (1 - level_k)), -ln(1.5), +ln(1.5))
    - Persists state across restarts.
    - Supports 'active' (modifies output) or 'shadow' (tracks phi, leaves output untouched).
    """
    eta: float = 0.005
    mode: str = "shadow"  # 'shadow' or 'active'
    state_file: Path | str | None = None
    phi: dict[str, float] = field(default_factory=lambda: {"50": 0.0, "80": 0.0, "90": 0.0})
    update_count: int = 0
    clip_bound: float = math.log(1.5)

    def __post_init__(self):
        if self.state_file is None:
            self.state_file = DEFAULT_ACI_STATE_PATH
        self.state_file = Path(self.state_file)
        self.load_state()

    def load_state(self) -> None:
        """Load state from JSON if present."""
        if self.state_file and self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.phi = {k: float(v) for k, v in data.get("phi", {}).items()}
                    self.update_count = int(data.get("update_count", 0))
                    for k in ("50", "80", "90"):
                        if k not in self.phi:
                            self.phi[k] = 0.0
            except Exception:
                self.phi = {"50": 0.0, "80": 0.0, "90": 0.0}
                self.update_count = 0

    def save_state(self) -> None:
        """Persist state to JSON file."""
        if self.state_file:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "phi": {k: float(v) for k, v in self.phi.items()},
                "update_count": self.update_count,
                "mode": self.mode,
                "eta": self.eta,
                "clip_bound": self.clip_bound,
            }
            tmp = self.state_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self.state_file)

    def apply(self, quantiles: np.ndarray) -> np.ndarray:
        """Apply width scaling to quantiles array of shape (..., 7).

        If mode is 'shadow', returns quantiles unchanged.
        If mode is 'active', adjusts (2,4) with phi_50, (1,5) with phi_80, (0,6) with phi_90,
        leaving q50 untouched.
        """
        if self.mode != "active":
            return np.asarray(quantiles, dtype=np.float64)

        q = np.asarray(quantiles, dtype=np.float64).copy()
        q50 = q[..., 3:4]

        for item in INTERVAL_LEVELS:
            lo, hi, key = item["lo"], item["hi"], item["key"]
            multiplier = math.exp(self.phi[key])
            q[..., lo] = q50[..., 0] + multiplier * (q[..., lo] - q50[..., 0])
            q[..., hi] = q50[..., 0] + multiplier * (q[..., hi] - q50[..., 0])

        return np.sort(q, axis=-1)

    def update(self, y: float, quantiles: np.ndarray) -> dict[str, float]:
        """Update phi_k based on realized target return y.

        miss_k = 1 if y outside interval, 0 if inside.
        phi_k <- clip(phi_k + eta * (miss_k - (1 - level_k)), -ln(1.5), +ln(1.5))
        """
        y_val = float(y)
        q = np.asarray(quantiles, dtype=np.float64)
        if q.ndim > 1:
            q = q.mean(axis=tuple(range(q.ndim - 1)))

        delta_phi = {}
        for item in INTERVAL_LEVELS:
            level = item["level"]
            lo, hi, key = item["lo"], item["hi"], item["key"]
            nominal_miss = 1.0 - level
            miss_k = 1.0 if (y_val < q[lo] or y_val > q[hi]) else 0.0

            step = self.eta * (miss_k - nominal_miss)
            new_phi = float(np.clip(self.phi[key] + step, -self.clip_bound, self.clip_bound))
            delta_phi[key] = new_phi - self.phi[key]
            self.phi[key] = new_phi

        self.update_count += 1
        self.save_state()
        return dict(self.phi)

    def get_state(self) -> dict[str, Any]:
        """Return state snapshot."""
        return {
            "phi": dict(self.phi),
            "mode": self.mode,
            "eta": self.eta,
            "update_count": self.update_count,
        }


@dataclass
class ACICalibrator:
    """Static calibration plus an adaptive conformal alpha vector."""
    static: StaticCalibrator
    alpha: np.ndarray
    score_pool: Optional[np.ndarray] = None   # (3, m) calibration scores, if available
    width_tracker: Optional[ACISymmetricWidthTracker] = None

    @staticmethod
    def from_metadata(calibration: dict,
                      aci_alpha: Optional[Sequence[float]] = None,
                      gamma: float = 0.005,
                      width_tracker: Optional[ACISymmetricWidthTracker] = None) -> "ACICalibrator":
        static = StaticCalibrator.from_metadata(calibration)
        alpha = np.asarray(aci_alpha if aci_alpha is not None
                           else [a for _, _, a in PAIRS], dtype=np.float64).copy()
        if alpha.shape != (len(PAIRS),):
            raise ValueError(f"aci alpha must have {len(PAIRS)} entries, got {alpha.shape}")
        return ACICalibrator(static=static, alpha=alpha, width_tracker=width_tracker)

    def apply(self, member_q: np.ndarray,
              realized: Optional[np.ndarray] = None) -> np.ndarray:
        """Calibrate and (optionally) advance alpha state with a realized outcome."""
        q = np.asarray(member_q, dtype=np.float64).copy()
        if q.shape[-1] != 7:
            raise ValueError(f"expected 7 quantiles on the last axis, got {q.shape[-1]}")
        q[..., 3] += self.static.median_bias
        for j, ((lo, hi, nominal), off) in enumerate(zip(PAIRS, self.static.cqr_offsets)):
            o = off
            if self.score_pool is not None:
                o = cqr_offset(self.score_pool[j], self.alpha[j])
            q[..., lo] -= o
            q[..., hi] += o
        q = np.sort(q, axis=-1)

        # Apply symmetric width tracker if present
        if self.width_tracker is not None:
            q = self.width_tracker.apply(q)

        if realized is not None:
            y = float(np.asarray(realized).reshape(-1)[0])
            pooled = q.mean(axis=tuple(range(q.ndim - 1))) if q.ndim > 1 else q
            for j, (lo, hi, nominal) in enumerate(PAIRS):
                err = float(y < pooled[lo] or y > pooled[hi])
                self.alpha[j] = float(np.clip(
                    self.alpha[j] + self.static.gamma * (nominal - err), 1.0 / 101.0, 0.999))
            if self.width_tracker is not None:
                self.width_tracker.update(y, pooled)
        return q
