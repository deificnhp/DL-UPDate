import json, os, datetime
from pathlib import Path
from MODEL.config import ART, config, sha, dump
from MODEL.data import load
from MODEL.folds import manifest
from MODEL.train_eval_walkforward import run_year
from MODEL.evaluate import report
from MODEL.run_lockbox import criteria_check

# 1. Create acceptance_criteria.json cleanly without BOM
criteria = {
  "S1": {
    "description": "Mean Pinball Loss of PatchTST-Q strictly less than B0 baseline",
    "rules": [{"path": "/comparisons/0/skill_percent", "op": ">", "threshold": 0.0}]
  },
  "S2": {
    "description": "Zero quantile crossings across all predictions",
    "rules": [{"path": "/metrics/PatchTST-Q/pooled/crossing_count", "op": "==", "threshold": 0}]
  },
  "S3": {
    "description": "Empirical coverage at tau=0.05 between 3% and 7%",
    "rules": [
      {"path": "/metrics/PatchTST-Q/pooled/coverage/0.05/rate", "op": ">=", "threshold": 0.03},
      {"path": "/metrics/PatchTST-Q/pooled/coverage/0.05/rate", "op": "<=", "threshold": 0.07}
    ]
  },
  "S4": {
    "description": "Empirical coverage at tau=0.95 between 93% and 97%",
    "rules": [
      {"path": "/metrics/PatchTST-Q/pooled/coverage/0.95/rate", "op": ">=", "threshold": 0.93},
      {"path": "/metrics/PatchTST-Q/pooled/coverage/0.95/rate", "op": "<=", "threshold": 0.97}
    ]
  },
  "S5": {
    "description": "Kupiec test p-value at tau=0.05 strictly greater than 0.01",
    "rules": [{"path": "/metrics/PatchTST-Q/pooled/var_0.05/kupiec_p", "op": ">", "threshold": 0.01}]
  },
  "S6": {
    "description": "Annualized net Sharpe ratio strictly positive",
    "rules": [{"path": "/metrics/PatchTST-Q/pooled/economics/sharpe_annualized", "op": ">", "threshold": 0.0}]
  }
}

criteria_path = Path("MODEL/acceptance_criteria.json")
with open(criteria_path, "w", encoding="utf-8") as f:
    json.dump(criteria, f, indent=2)
print("[1/5] Verified and saved MODEL/acceptance_criteria.json")

# 2. Load dataset with lockbox=True
data_path = Path("DATA/PROCESSED/features_targets.parquet")
print(f"[2/5] Loading data from {data_path} with lockbox=True...")
d = load(data_path, True, lockbox=True)

manifest_data = manifest(d, include_lockbox=True)
dump(ART / "lockbox_fold_manifest.json", manifest_data)
print("[3/5] Generated lockbox_fold_manifest.json")

# 3. Run Lockbox 2026 execution
frozen = config()
print("[4/5] Executing 2026 Lockbox inference (Baselines, PatchTST-Q, C1-LSTM)...")
run_year(d, 2026, frozen, "lockbox")

# 4. Generate lockbox report and criteria verification
print("[5/5] Generating lockbox evaluation report and checking criteria...")
results = report("lockbox")
checks = criteria_check(criteria, results)

event = {
    "status": "COMPLETED",
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "data_sha256": sha(data_path),
    "frozen_config_sha256": sha(ART / "frozen_config_m3.json"),
    "acceptance_pass": all(v["passed"] for v in checks.values())
}

dump(ART / "lockbox_acceptance.json", checks)

lines = [
    "# NQ4H-Q Final Evaluation Report (2026 Lockbox)", "",
    f"Acceptance: {'PASS' if event['acceptance_pass'] else 'FAIL'}", "",
    "## Frozen evaluation (2019-2025)", "See EVALUATION_REPORT.md and evaluation_metrics.json.", "",
    "## Single-shot lockbox (2026)", "See LOCKBOX_REPORT.md and lockbox_metrics.json.", ""
]
for name, r in checks.items():
    st = "PASS" if r["passed"] else "FAIL"
    lines.append(f"- {name}: {st} - {r['description']}")

lines += [
    "",
    f"Dataset SHA-256: `{event['data_sha256']}`",
    f"Frozen config SHA-256: `{event['frozen_config_sha256']}`",
    "Metrics are computed in raw log-return units. Inactive confidence is NaN by design."
]
(ART / "FINAL_EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

card = [
    "# NQ4H-Q Model Card (Lockbox Completed)", "",
    "## Intended use",
    "Research-only CME NQ 4-hour return distribution forecasting; not investment advice.", "",
    "## Architecture",
    "Independent vectorized PatchTST-Q and two-layer LSTM deep ensembles. Seven noncrossing quantiles.", "",
    "## Data and evaluation",
    f"Dataset SHA-256: `{event['data_sha256']}`. Expanding folds, 2019-2025 evaluation; single-shot 2026 lockbox.", "",
    "## Limitations",
    "Midquote return backtests are not fill simulations. Financial nonstationarity limits conformal guarantees.", "",
    f"Acceptance passed: {event['acceptance_pass']}. See ARTIFACTS/FINAL_EVALUATION_REPORT.md and lockbox_acceptance.json."
]
Path("MODEL/MODEL_CARD.md").write_text("\n".join(card) + "\n", encoding="utf-8")

print("\n" + "=" * 60)
print("2026 LOCKBOX EXECUTION COMPLETE")
print(f"Overall Acceptance: {'PASS' if event['acceptance_pass'] else 'FAIL'}")
for name, r in checks.items():
    st = "PASS" if r["passed"] else "FAIL"
    print(f"  {name}: {st} -> {r['description']}")
print("=" * 60)
