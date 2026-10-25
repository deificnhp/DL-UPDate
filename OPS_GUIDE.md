# NQ4H-Q Operational Guide (Windows PowerShell & Linux Bash)

This guide documents the operational procedures, CLI commands, and automated pipelines for the **NQ4H-Q** (`PatchTST` K=16, L=300, 7 quantiles) system:
- Runtime Inference with Fail-Closed Health Guards & DST Clock Monitoring
- Online Adaptive Conformal Inference (ACI) Width Tracking
- Weekly Model Decay Surveillance & State Machine
- Controlled Retraining, Staggered Refresh & Automated Acceptance
- Instant Rollback via Pointer-Based Checkpoint Registry

---

## 1. Runtime Inference with Health Guards

The live inference engine continuously queries MT5 1-minute bars, aggregates 4-hour session-anchored bars (slots 0–4 starting at 18, 22, 02, 06, 10 ET), computes the causal 11 continuous features + calendar covariates, runs BF16/FP32 ensemble forward passes, applies static CQR + online symmetric width adjustments, and verifies health state.

### PowerShell Command:
```powershell
python -m inference_live.cli predict `
    --symbol NDXUSD `
    --ckpt-dir DoneSetup\CKPTs `
    --arch patchtst `
    --server-utc-offset 3 `
    --json
```

### Health Contract:
- **`GREEN` (Cap: 100%):** All inputs finite, broker clock offset verified (`broker_offset_detected == broker_offset_configured`), empirical coverage normal.
- **`YELLOW` (Cap: 50%):** Moderate drift or single binomial tail breach detected.
- **`RED` (Cap: 25%):** Persistent decay confirmed by 2 consecutive weekly evaluations or negative pinball skill vs baseline.
- **`BLACK` (Cap: 0% / `forecast_valid=False`):** Hard failure (e.g. DST offset mismatch, NaN/Inf in 300-bar lookback window). Sizing multiplier capped to 0.0 to prevent executing trades on corrupt data.

---

## 2. Weekly Model Decay & Surveillance Monitor

Run weekly (or after every 25–50 bars) to evaluate empirical tail coverage vs exact Binomial 95% (YELLOW) and 99% (RED) bands across rolling windows `W125`, `W250`, `W500`, monitor pinball skill against scale-only baseline, detect Page-Hinkley drift on $|z|$, and compute continuous feature PSI.

### PowerShell Command:
```powershell
python -m inference_live.ops.monitor `
    --store-dir DoneSetup\ops_store `
    --json
```

### Outputs Generated:
- `DoneSetup/ops_store/health_state.json`: Machine-readable health state with 2-consecutive-evaluation hysteresis.
- `DoneSetup/ops_store/DECAY_MONITOR_REPORT.md`: Comprehensive Markdown surveillance report.

---

## 3. Controlled Retraining & Staggered Refresh

Retraining orchestrator bridging `interm/` to `DoneSetup/CKPTs/` with 4 operational tiers:

### Tier 2: Static CQR Offset Refit (No weights changed)
Refits conformal quantile regression offsets on the latest calibration window without modifying `.pt` neural network weights.
```powershell
python -m interm.MODEL.retrain_ops `
    --mode tier2_cqr `
    --data DATA\PROCESSED\features_targets.parquet `
    --ckpt-dir DoneSetup\CKPTs
```

### Tier 3a: Seasonal Scale Factor Refit (50/50 Shrinkage + CQR)
Re-estimates slot factors with 50% shrinkage toward current values and refits CQR offsets.
```powershell
python -m interm.MODEL.retrain_ops `
    --mode tier3a_scale `
    --data DATA\PROCESSED\features_targets.parquet `
    --ckpt-dir DoneSetup\CKPTs
```

### Tier 3b: Staggered Member Refresh (Partial Retraining)
Freezes 12 of the 16 ensemble members and trains only 4 members on expanding data with exponential recency weighting (half-life = 3 years = 4500 bars), applying the exact 25-bar embargo and 1-bar purge.
```powershell
python -m interm.MODEL.retrain_ops `
    --mode tier3b_partial `
    --data DATA\PROCESSED\features_targets.parquet `
    --base-ckpt DoneSetup\CKPTs\checkpoints\2026_patchtst.pt `
    --ckpt-dir DoneSetup\CKPTs `
    --epochs 20
```

### Tier 4: Full Ensemble Retrain
Full 16-member expanding-window retrain with recency weighting and stationary block bootstrap.
```powershell
python -m interm.MODEL.retrain_ops `
    --mode tier4_full `
    --data DATA\PROCESSED\features_targets.parquet `
    --ckpt-dir DoneSetup\CKPTs `
    --epochs 30
```

### Automated Acceptance Gate:
Before exporting any checkpoint to `DoneSetup/CKPTs/`, candidate models must pass 4 gates on holdout data:
1. Challenger Pinball Loss $\le$ Champion Pinball Loss (HAC Diebold-Mariano test).
2. Skill vs Scale-Only Baseline $> 0$.
3. 90% interval coverage within 95% binomial band.
4. Golden I/O parity check matches fixed test windows.

---

## 4. Instant Rollback via Checkpoint Registry

Model deployments use pointer-based references in `DoneSetup/CKPTs/checkpoint_registry.yaml`. If an anomaly is observed in production with the champion model, roll back to standby instantly (< 1 minute) without recompilation or server restarts.

### Registry Structure (`checkpoint_registry.yaml`):
```yaml
active: champion
pointers:
  champion:
    model_version: 20261025_tier3b_partial
    checkpoint: checkpoints/20261025_tier3b_partial_patchtst.pt
    provenance: provenance/20261025_tier3b_partial_patchtst.json
  standby:
    model_version: 2026_patchtst
    checkpoint: checkpoints/2026_patchtst.pt
    provenance: provenance/2026_patchtst.json
  shadow: null
```

### Instant Rollback Command:
In PowerShell, switch `active: champion` to `active: standby`:
```powershell
(Get-Content DoneSetup\CKPTs\checkpoint_registry.yaml) -replace 'active: champion', 'active: standby' | Set-Content DoneSetup\CKPTs\checkpoint_registry.yaml
```
To verify the active model pointer:
```powershell
python -m inference_live.cli check --ckpt-dir DoneSetup\CKPTs
```
EOF
