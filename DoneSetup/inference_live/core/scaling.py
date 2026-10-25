"""The frozen feature transform: robust scaling + seasonal scale factors.

This is the live system's copy of MODEL/data.py:FoldTransform, but in APPLY-ONLY form.
The fitted parameters (median, IQR, slot_factors) come from the trained checkpoint's
`transform` block -- they are NEVER refitted here. That is the whole point: the live
system must place new data on exactly the same scale the model was trained on.

Training reference (MODEL/data.py):
    x     = clip(RobustScaler.transform(features), -5, 5)
    scale = sigma_ewma[t] * slot_factors[slot_next[t]]
    z     = r_next[t] / scale
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .features import FEATURE_ORDER

CLIP = 5.0


@dataclass(frozen=True)
class FrozenTransform:
    """The frozen per-fold transform.

    Attributes mirror the checkpoint's `transform` dict:
      median      : RobustScaler centre per feature (11,)
      iqr         : RobustScaler scale per feature (11,)
      slot_factors: 5 seasonal RMS factors, geometrically normalised to 1
    """
    median: np.ndarray
    iqr: np.ndarray
    slot_factors: np.ndarray
    train_indices: Optional[list] = None

    @staticmethod
    def from_provenance(transform: dict) -> "FrozenTransform":
        missing = [k for k in ("median", "iqr", "slot_factors") if k not in transform]
        if missing:
            raise ValueError(f"checkpoint transform is missing keys: {missing}")
        median = np.asarray(transform["median"], dtype=np.float64)
        iqr = np.asarray(transform["iqr"], dtype=np.float64)
        factors = np.asarray(transform["slot_factors"], dtype=np.float64)
        if median.shape != (len(FEATURE_ORDER),) or iqr.shape != (len(FEATURE_ORDER),):
            raise ValueError(
                f"transform shape mismatch: median {median.shape}, iqr {iqr.shape}, "
                f"expected ({len(FEATURE_ORDER)},)")
        if factors.shape != (5,):
            raise ValueError(f"slot_factors must have 5 entries, got {factors.shape}")
        if not np.all(iqr > 0):
            raise ValueError("transform IQR contains non-positive entries")
        if not np.all(factors > 0):
            raise ValueError("transform slot_factors contains non-positive entries")
        return FrozenTransform(median=median, iqr=iqr, slot_factors=factors,
                               train_indices=transform.get("train_indices"))

    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """Robust-scale and clip. `features` is (n, 11) in FEATURE_ORDER."""
        x = (np.asarray(features, dtype=np.float64) - self.median) / self.iqr
        return np.clip(x, -CLIP, CLIP).astype(np.float32)

    def target_scale(self, sigma_ewma: np.ndarray, slot_next: np.ndarray) -> np.ndarray:
        """The multiplicative scale from scaled units to raw log-return units.

        `scale[t] = sigma_ewma[t] * slot_factors[slot_next[t]]`
        """
        sigma = np.asarray(sigma_ewma, dtype=np.float64)
        slot = np.asarray(slot_next, dtype=np.int64)
        if np.any((slot < 0) | (slot >= len(self.slot_factors))):
            raise ValueError("slot_next out of range for the frozen transform")
        return sigma * self.slot_factors[slot]
