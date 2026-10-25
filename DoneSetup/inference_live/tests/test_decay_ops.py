"""Comprehensive unit and integration tests for Model Decay, Online ACI, and Retrain Pipeline.

Verifies:
1. Urgent Runtime Data-Contract & DST Clock Guard
2. Operational Logging & Realization Join
3. Online ACI Symmetric Width Tracker (Mode A)
4. Lean Weekly Decay Monitor & State Machine
5. Controlled Retraining (Partial/Full/CQR) & Pointer-Based Checkpoint Registry Rollback
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from DoneSetup.inference_live.config import Config, FeatureConfig, ModelConfig, MT5Config, SlotTable
from DoneSetup.inference_live.core import bar_builder
from DoneSetup.inference_live.core.registry import load_registry, resolve_registry_pointer
from DoneSetup.inference_live.data.mt5_source import detect_broker_utc_offset
from DoneSetup.inference_live.engine import LiveEngine, QuantilePrediction
from DoneSetup.inference_live.model.calibration import (
    ACICalibrator,
    ACISymmetricWidthTracker,
    StaticCalibrator,
)
from DoneSetup.inference_live.ops.acceptance import (
    AcceptanceGate,
    diebold_mariano_hac,
    pinball_loss_matrix,
)
from DoneSetup.inference_live.ops.logger import OperationalLogger
from DoneSetup.inference_live.ops.monitor import (
    DecayMonitor,
    compute_psi,
    evaluate_binomial_bands,
    page_hinkley_test,
)
from DoneSetup.inference_live.ops.realization_join import RealizationJoiner, compute_pinball
from interm.MODEL.model_patchtst import Ensemble, PatchTST
from interm.MODEL.retrain_ops import RetrainPipeline, load_or_init_registry, save_registry


@pytest.fixture
def tmp_ops_dir(tmp_path):
    d = tmp_path / "ops_store"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    d = tmp_path / "CKPTs"
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    (d / "provenance").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1. DST Clock Guard and Data-Contract Checks
# ---------------------------------------------------------------------------
def test_detect_broker_utc_offset():
    """Verify offset detection rounds correctly to nearest half-hour or hour."""
    # When server tick is 3h (10800s) ahead of system UTC:
    offset = detect_broker_utc_offset(last_tick_epoch=10800.0, system_utc_epoch=0.0)
    assert offset == 3.0

    # With minor network jitter (e.g., 10815s)
    offset_jitter = detect_broker_utc_offset(last_tick_epoch=10815.0, system_utc_epoch=0.0)
    assert offset_jitter == 3.0

    # Half hour broker (e.g., UTC+5:30 -> 19800s)
    offset_half = detect_broker_utc_offset(last_tick_epoch=19800.0, system_utc_epoch=0.0)
    assert offset_half == 5.5

    # EU DST shift from UTC+3 to UTC+2 (7200s)
    offset_dst = detect_broker_utc_offset(last_tick_epoch=7200.0, system_utc_epoch=0.0)
    assert offset_dst == 2.0


def test_dst_clock_guard_fail_closed_in_engine(tmp_path, tmp_ckpt_dir):
    """If broker_offset_detected != broker_offset_configured, engine must set health_state='BLACK',
    forecast_valid=False, and sizing_multiplier_cap=0.0."""
    # Create synthetic checkpoint and provenance
    cfg_pt = {"lookback": 10, "K": 2, "d_model": 16, "n_heads": 2, "n_layers": 1, "dropout": 0.1, "seed": 42}
    ens = Ensemble(lambda c: PatchTST(c, Fdim=11), cfg_pt, device="cpu")
    pt_path = tmp_ckpt_dir / "checkpoints" / "2026_patchtst.pt"
    prov_path = tmp_ckpt_dir / "provenance" / "2026_patchtst.json"

    torch.save({
        "params": {k: v.cpu() for k, v in ens.params.items()},
        "config": cfg_pt,
        "architecture": "patchtst",
        "transform": {
            "train_indices": [0],
            "median": [0.0] * 11,
            "iqr": [1.0] * 11,
            "slot_factors": [1.0] * 5,
        },
        "logs": [],
    }, pt_path)

    prov_path.write_text(json.dumps({
        "fold": "2026",
        "calibration": {
            "median_bias": 0.0,
            "cqr_offsets": [0.01, 0.02, 0.03],
        },
    }))

    cfg = Config(
        checkpoint_dir=tmp_ckpt_dir,
        fold="2026",
        architectures=("patchtst",),
        model=ModelConfig(lookback=10, K=2),
        mt5=MT5Config(server_utc_offset_hours=3),
        warmup_offset=0,
    )
    engine = LiveEngine(cfg)
    engine.load_models()

    # Create synthetic bars (needs >= 135 bars for warm-up features to be finite)
    n_bars = 160
    start_ts = pd.Timestamp("2026-01-01 18:00:00", tz="UTC")
    bar_records = []
    for i in range(n_bars):
        b_start = start_ts + pd.Timedelta(hours=4 * i)
        b_end = b_start + pd.Timedelta(hours=4)
        slot = i % 5
        td = (b_start + pd.Timedelta(days=1 if slot == 0 else 0)).date()
        bar_records.append({
            "bar_start_utc": b_start,
            "bar_end_utc": b_end,
            "trade_date": td,
            "slot": slot,
            "open_mid": 100.0 + i,
            "high_mid": 101.0 + i,
            "low_mid": 99.0 + i,
            "close_mid": 100.5 + i,
            "volume": 1000.0,
            "expected_minutes": 240,
            "observed_minutes": 240,
            "quote_coverage": 1.0,
            "longest_missing_run": 0,
        })
    df_bars = pd.DataFrame(bar_records)
    engine.seed_history_from_bars(df_bars)

    # Normal prediction without DST mismatch (use row 145 where lookback window is completely finite)
    pred_normal = engine.predict("patchtst", decision_index=145, live=False)
    assert pred_normal is not None
    assert pred_normal.forecast_valid is True
    assert pred_normal.health_state == "GREEN"
    assert pred_normal.sizing_multiplier_cap == 1.0

    # Now simulate DST shift: server tick indicates UTC+2, but configured is UTC+3
    engine.state.detected_broker_offset = 2.0
    pred_dst_mismatch = engine.predict("patchtst", decision_index=145, live=False)
    assert pred_dst_mismatch is not None
    assert pred_dst_mismatch.forecast_valid is False
    assert pred_dst_mismatch.health_state == "BLACK"
    assert pred_dst_mismatch.sizing_multiplier_cap == 0.0
    assert any("offset mismatch" in r.lower() for r in pred_dst_mismatch.health_reasons)
    assert np.isnan(pred_dst_mismatch.quantiles).all()


# ---------------------------------------------------------------------------
# 2. Operational Logging & Realization Join
# ---------------------------------------------------------------------------
def test_operational_logger_and_realization_join(tmp_ops_dir):
    """Test forecast logging, pending registration, bar closing join, and scored log output."""
    logger = OperationalLogger(store_dir=tmp_ops_dir)
    joiner = RealizationJoiner(store_dir=tmp_ops_dir)

    target_start = pd.Timestamp("2026-06-02 02:00:00", tz="UTC")
    target_end = pd.Timestamp("2026-06-02 06:00:00", tz="UTC")

    forecast_row = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "broker_offset_detected": 3.0,
        "trade_date": "2026-06-02",
        "slot_current": 1,
        "slot_next": 2,
        "last_close": 10000.0,
        "sigma_ewma": 0.005,
        "slot_factor": 1.0,
        "scale": 0.005,
        "q_raw": [-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015],
        "q_final": [-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015],
        "ens_dispersion": [0.001] * 7,
        "aci_state": {"phi": {"50": 0.0, "80": 0.0, "90": 0.0}},
        "health_state": "GREEN",
        "model_version": "champion",
        "target_bar_start_utc": target_start.isoformat(),
        "target_trade_date": "2026-06-02",
        "target_slot": 2,
        "bar_end_utc": "2026-06-02T02:00:00+00:00",
    }

    logger.log_forecast(forecast_row)
    joiner.register_forecast(forecast_row)

    assert (tmp_ops_dir / "forecast_log.parquet").exists()
    assert (tmp_ops_dir / "pending_forecasts.parquet").exists()

    df_fc = logger.read_forecasts()
    assert len(df_fc) == 1
    assert df_fc.iloc[0]["last_close"] == 10000.0

    # Simulate bar t+1 closing with close price = 10050.0
    closed_bar = {
        "close_mid": 10050.0,
        "bar_start_utc": target_start,
        "bar_end_utc": target_end,
        "trade_date": "2026-06-02",
        "slot": 2,
    }
    scored = joiner.process_closed_bar(closed_bar)
    assert scored is not None
    assert scored["c_t"] == 10000.0
    assert scored["c_next"] == 10050.0
    expected_ret = math.log(10050.0 / 10000.0)
    assert math.isclose(scored["realized_logret"], expected_ret, rel_tol=1e-5)
    assert len(scored["pinball"]) == 7
    assert len(scored["baseline_pinball"]) == 7
    assert scored["in_90"] in (0, 1)

    # Scored parquet exists and pending is emptied
    assert (tmp_ops_dir / "scored_log.parquet").exists()
    scored_df = pd.read_parquet(tmp_ops_dir / "scored_log.parquet")
    assert len(scored_df) == 1


# ---------------------------------------------------------------------------
# 3. Online ACI Symmetric Width Tracker (Mode A)
# ---------------------------------------------------------------------------
def test_aci_symmetric_width_tracker(tmp_ops_dir):
    """Verify Mode A ACI: q50 untouched, phi update formula, clipping, and persistence."""
    state_file = tmp_ops_dir / "aci_state.json"
    tracker = ACISymmetricWidthTracker(eta=0.005, mode="active", state_file=state_file)

    q_init = np.array([-0.03, -0.02, -0.01, 0.001, 0.01, 0.02, 0.03])
    q_out = tracker.apply(q_init)
    # When phi is 0, output is unchanged
    assert np.allclose(q_out, q_init)
    assert q_out[3] == q_init[3]

    # Extreme positive miss: y = 0.10 is greater than q[6], so misses 50%, 80%, 90%
    # Update step: phi_k += eta * (1 - (1 - level_k)) = eta * level_k > 0
    tracker.update(y=0.10, quantiles=q_init)
    assert tracker.phi["50"] > 0.0
    assert tracker.phi["80"] > 0.0
    assert tracker.phi["90"] > 0.0
    assert tracker.update_count == 1

    # Apply width widening: intervals should expand around q50, q50 remains unchanged!
    q_widened = tracker.apply(q_init)
    assert q_widened[3] == q_init[3]
    assert q_widened[0] < q_init[0]  # lower bound expands downwards
    assert q_widened[6] > q_init[6]  # upper bound expands upwards
    assert q_widened[2] < q_init[2]
    assert q_widened[4] > q_init[4]

    # Check persistence: new instance loads state
    tracker_reloaded = ACISymmetricWidthTracker(mode="active", state_file=state_file)
    assert tracker_reloaded.update_count == 1
    assert math.isclose(tracker_reloaded.phi["90"], tracker.phi["90"])

    # Test clipping: many misses cannot exceed ln(1.5)
    for _ in range(500):
        tracker.update(y=0.50, quantiles=q_init)
    assert tracker.phi["90"] <= math.log(1.5) + 1e-9
    assert tracker.phi["50"] <= math.log(1.5) + 1e-9


# ---------------------------------------------------------------------------
# 4. Lean Weekly Decay Monitor & State Machine
# ---------------------------------------------------------------------------
def test_decay_monitor_state_machine(tmp_ops_dir):
    """Test Binomial bands, Page-Hinkley drift, PSI, and hysteresis."""
    scored_path = tmp_ops_dir / "scored_log.parquet"
    monitor = DecayMonitor(store_dir=tmp_ops_dir)

    # Synthetic scored logs with 150 bars of normal performance
    rng = np.random.default_rng(123)
    n = 150
    records = []
    for i in range(n):
        y = rng.normal(0, 0.005)
        # 90% interval [q0, q6] covers ~90%
        in_90 = 1 if abs(y) < 0.008 else 0
        in_80 = 1 if abs(y) < 0.0065 else 0
        in_50 = 1 if abs(y) < 0.0035 else 0
        q = np.array([-0.008, -0.0065, -0.0035, 0.0, 0.0035, 0.0065, 0.008])
        pl = compute_pinball(y, q).tolist()
        pl_base = compute_pinball(y, q * 1.1).tolist()  # baseline slightly worse -> positive skill

        records.append({
            "evaluated_at_utc": "2026-06-01T00:00:00+00:00",
            "realized_logret": y,
            "scale": 0.005,
            "z": y / 0.005,
            "q_final": q.tolist(),
            "hit": [int(y < val) for val in q],
            "pinball": pl,
            "in_50": in_50,
            "in_80": in_80,
            "in_90": in_90,
            "baseline_pinball": pl_base,
            "model_version": "champion",
            "health_state": "GREEN",
        })
    pd.DataFrame(records).to_parquet(scored_path, index=False)

    summary = monitor.run_checks()
    assert summary["health_state"] == "GREEN"
    assert summary["sizing_multiplier_cap"] == 1.0
    assert (tmp_ops_dir / "health_state.json").exists()
    assert (tmp_ops_dir / "DECAY_MONITOR_REPORT.md").exists()

    # Now introduce severe decay (negative skill and tail miscoverage)
    decayed_records = []
    for i in range(150):
        y = rng.normal(0.02, 0.02)  # severe regime break
        q = np.array([-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005])
        pl = compute_pinball(y, q).tolist()  # high loss
        pl_base = compute_pinball(y, q * 0.5).tolist()  # baseline better -> negative skill
        decayed_records.append({
            "evaluated_at_utc": "2026-06-02T00:00:00+00:00",
            "realized_logret": y,
            "scale": 0.005,
            "z": y / 0.005,
            "q_final": q.tolist(),
            "hit": [int(y < val) for val in q],
            "pinball": pl,
            "in_50": 0,
            "in_80": 0,
            "in_90": 0,
            "baseline_pinball": pl_base,
            "model_version": "champion",
            "health_state": "GREEN",
        })
    pd.DataFrame(decayed_records).to_parquet(scored_path, index=False)

    # 1st evaluation: raw RED downgraded to YELLOW by hysteresis
    summary_eval1 = monitor.run_checks()
    assert summary_eval1["raw_eval_state"] == "RED"
    assert summary_eval1["health_state"] == "YELLOW"
    assert summary_eval1["consecutive_red_count"] == 1
    assert summary_eval1["sizing_multiplier_cap"] == 0.50

    # 2nd evaluation on persisted RED: triggers confirmed RED state and 0.25 cap
    summary_eval2 = monitor.run_checks()
    assert summary_eval2["health_state"] == "RED"
    assert summary_eval2["consecutive_red_count"] == 2
    assert summary_eval2["sizing_multiplier_cap"] == 0.25


# ---------------------------------------------------------------------------
# 5. Controlled Retrain, Staggered Refresh & Registry Rollback
# ---------------------------------------------------------------------------
def test_retrain_pipeline_modes_and_registry_rollback(tmp_ckpt_dir):
    """Test retrain execution modes (tier2_cqr, tier3b_partial) and instant rollback."""
    pipeline = RetrainPipeline(ckpt_dir=tmp_ckpt_dir, device="cpu")

    # Create synthetic dataset DataFrame
    n = 1200
    rng = np.random.default_rng(42)
    trade_dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    feats = {f"f{i}": rng.normal(0, 1, n) for i in range(11)}
    feats_named = {
        "r1_z": rng.normal(0, 1, n),
        "r5_z": rng.normal(0, 1, n),
        "r30_z": rng.normal(0, 1, n),
        "gap_z": rng.normal(0, 1, n),
        "lgk_bar": rng.normal(0, 1, n),
        "lrv_d": rng.normal(0, 1, n),
        "lrv_w": rng.normal(0, 1, n),
        "lrv_m": rng.normal(0, 1, n),
        "clv": rng.normal(0, 1, n),
        "ema_dist": rng.normal(0, 1, n),
        "rvol": rng.normal(0, 1, n),
    }
    df_data = pd.DataFrame({
        **feats_named,
        "bar_start_utc": trade_dates,
        "bar_end_utc": trade_dates + pd.Timedelta(hours=4),
        "trade_date": trade_dates.date,
        "slot": [(i % 5) for i in range(n)],
        "slot_next": [((i + 1) % 5) for i in range(n)],
        "dow_next": [((trade_dates[i].weekday()) % 5) for i in range(n)],
        "sigma_ewma": np.full(n, 0.005),
        "r_next": rng.normal(0, 0.005, n),
        "close_mid": 10000.0 + np.cumsum(rng.normal(0, 10, n)),
        "bar_seq": np.arange(n),
        "target_bar_seq": np.arange(1, n + 1),
        "next_valid": np.ones(n, dtype=bool),
        "expected_minutes": np.full(n, 240),
    })

    # Train/val/cal/test splits
    parts = {
        "train": np.arange(300, 800),
        "validation": np.arange(800, 950),
        "calibration": np.arange(950, 1050),
        "test": np.arange(1050, 1150),
    }

    # 1. First run tier2_cqr (generates champion)
    res_cqr = pipeline.execute_pipeline(
        mode="tier2_cqr",
        data_df=df_data,
        parts=parts,
        epochs=1,
        version_tag="v1_champ",
        strict_baseline_gate=False,  # On random synthetic weights without training, relax baseline gate
    )
    assert res_cqr["passed_acceptance"] is True
    reg1 = load_registry(tmp_ckpt_dir)
    assert reg1["active"] == "champion"
    assert reg1["pointers"]["champion"]["model_version"] == "v1_champ"

    # 2. Next run tier3b_partial (staggered refresh)
    champ_pt_file = tmp_ckpt_dir / "checkpoints" / "v1_champ_patchtst.pt"
    res_partial = pipeline.execute_pipeline(
        mode="tier3b_partial",
        data_df=df_data,
        parts=parts,
        base_ckpt_path=champ_pt_file,
        epochs=1,
        version_tag="v2_partial",
        strict_baseline_gate=False,
    )
    assert res_partial["passed_acceptance"] is True
    reg2 = load_registry(tmp_ckpt_dir)
    assert reg2["pointers"]["champion"]["model_version"] == "v2_partial"
    assert reg2["pointers"]["standby"]["model_version"] == "v1_champ"

    # 3. Test instant rollback (<1 second by changing pointer)
    ptr_current = resolve_registry_pointer(tmp_ckpt_dir, role="active")
    assert ptr_current["model_version"] == "v2_partial"

    # Roll back by setting active to standby
    reg2["active"] = "standby"
    save_registry(reg2, tmp_ckpt_dir / "checkpoint_registry.yaml")

    ptr_rolled_back = resolve_registry_pointer(tmp_ckpt_dir, role="active")
    assert ptr_rolled_back["model_version"] == "v1_champ"
    assert ptr_rolled_back["checkpoint_path"].exists()
    assert ptr_rolled_back["provenance_path"].exists()
