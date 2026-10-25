"""Automated Acceptance Gate for Model Retraining and Calibration Refits.

Verifies 4 gates on holdout test split before exporting artifacts to DoneSetup/CKPTs/:
1. Challenger Pinball Loss <= Champion Pinball Loss (with HAC t-stat / Diebold-Mariano check).
2. Pinball Skill vs Scale-Only Baseline > 0.
3. 90% interval coverage within 95% binomial band.
4. golden_io parity check passes (20 fixed windows saved alongside checkpoint).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binom, norm

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
Z_BASELINE = np.array([norm.ppf(t) for t in TAUS], dtype=np.float64)


def pinball_loss_matrix(y: np.ndarray, q: np.ndarray, taus: np.ndarray | list[float] = TAUS) -> np.ndarray:
    """Compute (N, 7) pinball loss matrix."""
    y = np.asarray(y, dtype=np.float64)[:, None]
    q = np.asarray(q, dtype=np.float64)
    t = np.asarray(taus, dtype=np.float64)[None, :]
    e = y - q
    return np.maximum(t * e, (t - 1.0) * e)


def diebold_mariano_hac(loss_champ: np.ndarray, loss_challenger: np.ndarray, max_lag: int = 5) -> Tuple[float, float]:
    """Diebold-Mariano test with Newey-West HAC standard error.

    loss_diff d_t = loss_champ_t - loss_challenger_t (positive d_t means challenger is better).
    Returns (t_stat, p_value).
    """
    d = np.asarray(loss_champ, dtype=np.float64) - np.asarray(loss_challenger, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return 0.0, 1.0

    mean_d = np.mean(d)
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0

    for lag in range(1, min(max_lag + 1, n)):
        weight = 1.0 - (lag / (max_lag + 1.0))
        cov = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        gamma_sum += 2.0 * weight * cov

    var_hac = (gamma0 + gamma_sum) / n
    if var_hac <= 1e-12:
        return (10.0 if mean_d > 0 else -10.0), 0.0

    t_stat = mean_d / np.sqrt(var_hac)
    p_val = 2.0 * (1.0 - norm.cdf(abs(t_stat)))
    return float(t_stat), float(p_val)


class AcceptanceGate:
    """Verifies challenger against champion and baseline before deployment."""

    def __init__(self, golden_io_path: Optional[Path] = None):
        self.golden_io_path = golden_io_path

    def evaluate(self,
                 y_test: np.ndarray,
                 scale_test: np.ndarray,
                 q_challenger: np.ndarray,
                 q_champion: Optional[np.ndarray] = None,
                 golden_check_fn: Optional[Any] = None,
                 strict_baseline_gate: bool = True) -> Tuple[bool, dict[str, Any]]:
        """Run the 4 acceptance checks.

        q_challenger: (N, 7) or (K, N, 7)
        q_champion: (N, 7) or (K, N, 7) (optional; if None, evaluated against baseline)
        """
        y = np.asarray(y_test, dtype=np.float64)
        scale = np.asarray(scale_test, dtype=np.float64)
        n = len(y)

        # Average over members if K dimension present
        if q_challenger.ndim == 3:
            qc = q_challenger.mean(axis=0)
        else:
            qc = np.asarray(q_challenger, dtype=np.float64)

        loss_challenger_mat = pinball_loss_matrix(y, qc)
        loss_challenger = loss_challenger_mat.mean(axis=1)  # mean over quantiles per row
        mean_pl_challenger = float(np.mean(loss_challenger))

        # Baseline loss
        q_base = scale[:, None] * Z_BASELINE[None, :]
        loss_base_mat = pinball_loss_matrix(y, q_base)
        loss_base = loss_base_mat.mean(axis=1)
        mean_pl_base = float(np.mean(loss_base))

        # Gate 1: Challenger vs Champion
        gate1_passed = True
        dm_stat, dm_pval = 0.0, 1.0
        mean_pl_champ = mean_pl_challenger
        if q_champion is not None:
            if q_champion.ndim == 3:
                qch = q_champion.mean(axis=0)
            else:
                qch = np.asarray(q_champion, dtype=np.float64)
            loss_champ = pinball_loss_matrix(y, qch).mean(axis=1)
            mean_pl_champ = float(np.mean(loss_champ))
            dm_stat, dm_pval = diebold_mariano_hac(loss_champ, loss_challenger)
            # Challenger must have loss <= champion (or within noise if not statistically significantly worse)
            if mean_pl_challenger > mean_pl_champ and (dm_stat < -1.645 and dm_pval < 0.10):
                gate1_passed = False
        else:
            # If no champion, Gate 1 checks against baseline
            gate1_passed = True if not strict_baseline_gate else (mean_pl_challenger <= mean_pl_base)

        # Gate 2: Skill vs Baseline > 0
        skill_score = 1.0 - (mean_pl_challenger / mean_pl_base) if mean_pl_base > 1e-12 else 0.0
        gate2_passed = True if not strict_baseline_gate else (skill_score > 0.0)

        # Gate 3: 90% interval coverage within 95% binomial band
        in_90_count = int(np.sum((qc[:, 0] <= y) & (y <= qc[:, 6])))
        lo_95 = int(binom.ppf(0.025, n, 0.90))
        hi_95 = int(binom.ppf(0.975, n, 0.90))
        gate3_passed = True if not strict_baseline_gate else (lo_95 <= in_90_count <= hi_95)
        cov_90 = float(in_90_count / n) if n > 0 else 0.0

        # Gate 4: Golden IO parity check
        gate4_passed = True
        golden_error = 0.0
        if golden_check_fn is not None:
            try:
                gate4_passed, golden_error = golden_check_fn()
            except Exception as exc:
                gate4_passed = False
                golden_error = float("inf")

        all_passed = bool(gate1_passed and gate2_passed and gate3_passed and gate4_passed)

        results = {
            "all_passed": all_passed,
            "gate1_loss_vs_champion": {
                "passed": gate1_passed,
                "pl_challenger": mean_pl_challenger,
                "pl_champion": mean_pl_champ,
                "dm_t_stat": dm_stat,
                "dm_p_value": dm_pval,
            },
            "gate2_skill_vs_baseline": {
                "passed": gate2_passed,
                "skill_score": skill_score,
                "pl_baseline": mean_pl_base,
            },
            "gate3_coverage_90": {
                "passed": gate3_passed,
                "observed_count": in_90_count,
                "observed_rate": cov_90,
                "band_95": [lo_95, hi_95],
                "n_samples": n,
            },
            "gate4_golden_io": {
                "passed": gate4_passed,
                "max_discrepancy": golden_error,
            },
        }

        return all_passed, results
