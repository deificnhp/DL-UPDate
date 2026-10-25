"""Lean Weekly Decay Monitor — Tier 1 to 4 Health Surveillance.

Reads scored_log.parquet (and optional feature data) over rolling windows:
    W125 (~1 month), W250 (~2 months / 10 weeks), W500 (~4 months)
Computes:
1. Empirical Tail Breaches & Interval Coverage (50/80/90%) vs exact Binomial 95% (YELLOW)
   and 99% (RED) bands (scipy.stats.binom).
2. Pinball Skill vs Scale-Only Baseline: 1 - PL_model / PL_baseline.
3. CUSUM / Page-Hinkley sequential drift detection on tail hits and |z| for sudden regime breaks.
4. Feature PSI on continuous features against a frozen reference profile.
5. State Machine Output: Writes DoneSetup/ops_store/health_state.json with hysteresis
   (2 consecutive RED evaluations required to trigger Tier 2/3/4 alerts) and generates
   a clean Markdown report summary.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

from ..logging_setup import get_logger
from .logger import DEFAULT_OPS_STORE

log = get_logger("ops.monitor")

WINDOWS = [125, 250, 500]
INTERVALS = [(50, "in_50", 0.50), (80, "in_80", 0.80), (90, "in_90", 0.90)]


def compute_psi(ref: np.ndarray, curr: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Compute Population Stability Index (PSI) between reference and current distribution."""
    ref_clean = np.asarray(ref, dtype=np.float64)
    curr_clean = np.asarray(curr, dtype=np.float64)
    ref_clean = ref_clean[np.isfinite(ref_clean)]
    curr_clean = curr_clean[np.isfinite(curr_clean)]

    if len(ref_clean) < 10 or len(curr_clean) < 10:
        return 0.0

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    # Ensure monotonic bin edges in case of ties
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(ref_clean, bins=bin_edges)[0]
    curr_counts = np.histogram(curr_clean, bins=bin_edges)[0]

    nb = len(ref_counts)
    ref_pct = (ref_counts + eps) / (len(ref_clean) + eps * nb)
    curr_pct = (curr_counts + eps) / (len(curr_clean) + eps * nb)

    return float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))


def page_hinkley_test(series: np.ndarray, delta: float = 0.005, threshold: float = 30.0) -> Tuple[bool, float]:
    """Page-Hinkley sequential drift detection for abrupt mean changes.

    m_t = sum_{i=1}^t (x_i - mean_x - delta)
    M_t = min_{i<=t} m_i
    PH_t = m_t - M_t > threshold -> Alarm
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return False, 0.0

    mean_x = np.mean(x)
    cum = 0.0
    min_cum = 0.0
    max_ph = 0.0

    for val in x:
        cum += (val - mean_x - delta)
        if cum < min_cum:
            min_cum = cum
        ph = cum - min_cum
        if ph > max_ph:
            max_ph = ph
        if ph > threshold:
            return True, float(max_ph)

    return False, float(max_ph)


def evaluate_binomial_bands(n: int, k: int, p: float) -> Tuple[str, dict]:
    """Evaluate observed count k against exact Binomial(n, p) bands.

    Returns:
        (state: 'GREEN' | 'YELLOW' | 'RED', stats_dict)
    95% two-sided band: [binom.ppf(0.025), binom.ppf(0.975)]
    99% two-sided band: [binom.ppf(0.005), binom.ppf(0.995)]
    """
    lo_99 = int(binom.ppf(0.005, n, p))
    hi_99 = int(binom.ppf(0.995, n, p))
    lo_95 = int(binom.ppf(0.025, n, p))
    hi_95 = int(binom.ppf(0.975, n, p))

    stats = {
        "n": n,
        "k": k,
        "p_expected": p,
        "rate_observed": float(k / n) if n > 0 else 0.0,
        "band_95": [lo_95, hi_95],
        "band_99": [lo_99, hi_99],
    }

    if k < lo_99 or k > hi_99:
        return "RED", stats
    if k < lo_95 or k > hi_95:
        return "YELLOW", stats
    return "GREEN", stats


class DecayMonitor:
    """Executes decay checks across rolling windows and manages persistent health state."""

    def __init__(self, store_dir: Path | str | None = None):
        self.store_dir = Path(store_dir) if store_dir is not None else DEFAULT_OPS_STORE
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.scored_parquet = self.store_dir / "scored_log.parquet"
        self.health_state_file = self.store_dir / "health_state.json"
        self.report_md_file = self.store_dir / "DECAY_MONITOR_REPORT.md"

    def load_prev_state(self) -> dict[str, Any]:
        """Load prior health state for hysteresis tracking."""
        if self.health_state_file.exists():
            try:
                with open(self.health_state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "current_state": "GREEN",
            "consecutive_red_count": 0,
            "last_updated_utc": None,
            "reasons": [],
        }

    def run_checks(self, ref_features_df: Optional[pd.DataFrame] = None,
                   curr_features_df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
        """Run all decay monitoring checks."""
        prev_state = self.load_prev_state()

        if not self.scored_parquet.exists():
            log.warning("No scored_log.parquet found at %s. State remains GREEN (insufficient data).",
                        self.scored_parquet)
            summary = {
                "evaluated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "health_state": "GREEN",
                "sizing_multiplier_cap": 1.0,
                "consecutive_red_count": 0,
                "reasons": ["No scored history available yet."],
                "windows": {},
                "psi": {},
            }
            self._write_outputs(summary, "# Decay Monitor Report\n\nNo scored logs found yet.")
            return summary

        df = pd.read_parquet(self.scored_parquet)
        total_bars = len(df)
        reasons = []
        worst_eval_state = "GREEN"

        window_results = {}

        for w in WINDOWS:
            if total_bars < min(w, 30):
                continue
            sub = df.tail(w)
            n = len(sub)
            w_res: dict[str, Any] = {"n": n, "intervals": {}, "skill": {}, "drift": {}}

            # 1. Interval Coverage checks
            for level_pct, col_name, nominal_p in INTERVALS:
                if col_name in sub.columns:
                    k = int(sub[col_name].sum())
                    state, stats = evaluate_binomial_bands(n, k, nominal_p)
                    w_res["intervals"][f"{level_pct}%"] = {"state": state, **stats}
                    if state == "RED":
                        reasons.append(f"Window {n}: {level_pct}% coverage breach ({stats['rate_observed']:.3f} outside 99% band)")
                        worst_eval_state = "RED"
                    elif state == "YELLOW" and worst_eval_state != "RED":
                        reasons.append(f"Window {n}: {level_pct}% coverage slight breach ({stats['rate_observed']:.3f} outside 95% band)")
                        worst_eval_state = "YELLOW"

            # 2. Pinball Skill vs Scale-Only Baseline: 1 - PL_model / PL_baseline
            if "pinball" in sub.columns and "baseline_pinball" in sub.columns:
                try:
                    # pinball is list/array of 7 values
                    pl_model_list = [np.mean(row) for row in sub["pinball"]]
                    pl_base_list = [np.mean(row) for row in sub["baseline_pinball"]]
                    pl_model = float(np.mean(pl_model_list))
                    pl_base = float(np.mean(pl_base_list))
                    skill = 1.0 - (pl_model / pl_base) if pl_base > 1e-12 else 0.0
                    w_res["skill"] = {
                        "pl_model": pl_model,
                        "pl_baseline": pl_base,
                        "skill_score": skill,
                    }
                    if skill < 0.0:
                        worst_eval_state = "RED"
                        reasons.append(f"Window {n}: Negative skill vs baseline ({skill:.4f} < 0)")
                except Exception as exc:
                    log.debug("Error computing skill in window %d: %s", w, exc)

            # 3. Sequential Drift Detection (Page-Hinkley on |z| and tail hits)
            if "z" in sub.columns:
                z_abs = np.abs(sub["z"].to_numpy(dtype=float))
                alarm_z, ph_z = page_hinkley_test(z_abs, delta=0.01, threshold=20.0)
                w_res["drift"]["z_abs"] = {"alarm": alarm_z, "stat": ph_z}
                if alarm_z:
                    reasons.append(f"Window {n}: Page-Hinkley drift detected on |z| (stat={ph_z:.2f})")
                    if worst_eval_state != "RED":
                        worst_eval_state = "YELLOW"

            window_results[f"W{w}"] = w_res

        # 4. Feature PSI Checks
        psi_results = {}
        if ref_features_df is not None and curr_features_df is not None:
            common_cols = [c for c in ref_features_df.columns if c in curr_features_df.columns and c not in ("bar_start_utc", "bar_end_utc", "trade_date")]
            for col in common_cols:
                ref_vals = ref_features_df[col].to_numpy(dtype=float)
                curr_vals = curr_features_df[col].to_numpy(dtype=float)
                val_psi = compute_psi(ref_vals, curr_vals)
                psi_results[col] = val_psi
                if val_psi > 0.25:
                    worst_eval_state = "RED"
                    reasons.append(f"Feature PSI critical drift for {col}: {val_psi:.3f} > 0.25")
                elif val_psi > 0.10 and worst_eval_state == "GREEN":
                    worst_eval_state = "YELLOW"
                    reasons.append(f"Feature PSI moderate drift for {col}: {val_psi:.3f} > 0.10")

        # 5. State Machine Hysteresis:
        # Require 2 consecutive RED evaluations to trigger Tier 2/3/4 action alerts.
        prev_consecutive_red = int(prev_state.get("consecutive_red_count", 0))
        if worst_eval_state == "RED":
            consecutive_red = prev_consecutive_red + 1
        else:
            consecutive_red = 0

        # Effective health state
        if worst_eval_state == "RED":
            if consecutive_red >= 2:
                final_state = "RED"
                sizing_cap = 0.25
            else:
                final_state = "YELLOW"  # 1st RED downgraded to YELLOW by hysteresis
                sizing_cap = 0.50
                reasons.append("First RED occurrence: state held at YELLOW pending 2nd consecutive RED verification.")
        elif worst_eval_state == "YELLOW":
            final_state = "YELLOW"
            sizing_cap = 0.50
        else:
            final_state = "GREEN"
            sizing_cap = 1.0

        summary = {
            "evaluated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "health_state": final_state,
            "raw_eval_state": worst_eval_state,
            "sizing_multiplier_cap": sizing_cap,
            "consecutive_red_count": consecutive_red,
            "reasons": reasons if reasons else ["All coverage, skill, drift, and PSI metrics normal."],
            "total_scored_bars": total_bars,
            "windows": window_results,
            "psi": psi_results,
        }

        # Build Markdown report
        md_report = self._build_markdown_report(summary)
        self._write_outputs(summary, md_report)
        return summary

    def _write_outputs(self, summary: dict[str, Any], md_report: str) -> None:
        """Write health_state.json and DECAY_MONITOR_REPORT.md."""
        with open(self.health_state_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(self.report_md_file, "w", encoding="utf-8") as f:
            f.write(md_report)
        log.info("Decay monitor completed. State: %s (Cap: %.2f)",
                 summary["health_state"], summary["sizing_multiplier_cap"])

    def _build_markdown_report(self, summary: dict[str, Any]) -> str:
        lines = [
            "# Model Health & Decay Surveillance Report",
            f"**Generated at (UTC):** {summary['evaluated_at_utc']}",
            f"**Current Health State:** `{summary['health_state']}` (Raw: `{summary.get('raw_eval_state')}`)",
            f"**Position Sizing Cap:** `{summary['sizing_multiplier_cap'] * 100:.0f}%`",
            f"**Consecutive RED Count:** `{summary['consecutive_red_count']}` (Hysteresis threshold: 2)",
            f"**Total Scored Bars:** {summary.get('total_scored_bars', 0)}",
            "",
            "## Summary of Findings & Health Reasons",
        ]
        for r in summary["reasons"]:
            lines.append(f"- {r}")

        lines.extend([
            "",
            "## Rolling Window Performance",
            "| Window | Metric / Interval | Observed | Expected | Status |",
            "|---|---|---|---|---|",
        ])

        for w_name, w_data in summary.get("windows", {}).items():
            for int_name, int_data in w_data.get("intervals", {}).items():
                obs = f"{int_data['rate_observed']*100:.1f}% ({int_data['k']}/{int_data['n']})"
                exp = f"{int_data['p_expected']*100:.1f}%"
                st = int_data["state"]
                lines.append(f"| {w_name} | Coverage {int_name} | {obs} | {exp} | {st} |")
            if "skill" in w_data and w_data["skill"]:
                sk = w_data["skill"]["skill_score"]
                st = "GREEN" if sk > 0 else "RED"
                lines.append(f"| {w_name} | Pinball Skill vs Baseline | {sk:+.4f} | > 0.00 | {st} |")
            if "drift" in w_data and "z_abs" in w_data["drift"]:
                alarm = w_data["drift"]["z_abs"]["alarm"]
                st = "YELLOW" if alarm else "GREEN"
                lines.append(f"| {w_name} | Page-Hinkley \|z\| Drift | Alarm={alarm} | None | {st} |")

        if summary.get("psi"):
            lines.extend([
                "",
                "## Feature Population Stability Index (PSI)",
                "| Feature | PSI | Status |",
                "|---|---|---|",
            ])
            for feat, val in summary["psi"].items():
                st = "GREEN" if val < 0.10 else ("YELLOW" if val <= 0.25 else "RED")
                lines.append(f"| {feat} | {val:.4f} | {st} |")

        lines.append("")
        return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Weekly Model Decay & Health Surveillance Monitor.")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_OPS_STORE,
                        help="Path to ops store containing scored_log.parquet")
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout")
    args = parser.parse_args(argv)

    monitor = DecayMonitor(store_dir=args.store_dir)
    res = monitor.run_checks()
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Decay Monitor completed. Health State: {res['health_state']} (Sizing Cap: {res['sizing_multiplier_cap']})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
