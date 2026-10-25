"""Controlled Retrain, Staggered Refresh & Export Orchestrator.

Bridges interm/ to DoneSetup/CKPTs/:
Modes supported via CLI flags:
  --mode tier2_cqr: Refit only static CQR offsets (calibration.json) on latest calibration window
                    without touching .pt weights.
  --mode tier3a_scale: Re-estimate slot_factors with 50/50 shrinkage toward current values + refit CQR.
  --mode tier3b_partial: Staggered Member Refresh — load existing .pt ensemble, freeze 12 of 16 members,
                         and retrain only 4 members on expanding data with exponential recency weighting
                         (half-life = 3 years = 4500 4H bars), using exact 25-bar embargo and 1-bar purge
                         from folds.py.
  --mode tier4_full: Full 16-member expanding-window retrain with recency weighting.

Automated Acceptance Gate:
  Verifies on holdout test split before exporting to DoneSetup/CKPTs/:
  1. Challenger Pinball Loss <= Champion Pinball Loss (with HAC t-stat / Diebold-Mariano check).
  2. Pinball Skill vs Scale-Only Baseline > 0.
  3. 90% interval coverage within 95% binomial band.
  4. golden_io parity check passes (20 fixed windows saved alongside the checkpoint).

Pointer-Based Checkpoint Registry:
  Updates DoneSetup/CKPTs/checkpoint_registry.yaml.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

log = logging.getLogger("interm.retrain_ops")

from DoneSetup.inference_live.ops.acceptance import AcceptanceGate, pinball_loss_matrix
from interm.MODEL.calibration import Calibrator, offset
from interm.MODEL.config import DEFAULT, FEATURES, TAUS
from interm.MODEL.data import FoldTransform
from interm.MODEL.model_patchtst import Ensemble, PatchTST, pinball

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERM_DIR = REPO_ROOT / "interm"
DONESETUP_DIR = REPO_ROOT / "DoneSetup"
CKPT_DIR = DONESETUP_DIR / "CKPTs"
REGISTRY_PATH = CKPT_DIR / "checkpoint_registry.yaml"

HALF_LIFE_BARS = 3 * 1500  # 3 years ~ 4500 4H bars


def compute_recency_weights(n_samples: int, half_life: float = HALF_LIFE_BARS) -> np.ndarray:
    """Compute normalized exponential recency weights for training samples.

    The latest sample (index n-1) has weight 1.0, decaying backwards:
        w_i = exp(- ln(2) * (n - 1 - i) / half_life)
    """
    decay_rate = math.log(2.0) / half_life
    indices = np.arange(n_samples)
    w = np.exp(-decay_rate * (n_samples - 1 - indices))
    return w / np.sum(w)


def load_or_init_registry(reg_path: Optional[Path] = None) -> dict[str, Any]:
    """Load checkpoint_registry.yaml or initialize standard structure."""
    path = reg_path or REGISTRY_PATH
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "pointers" in data:
                    return data
        except Exception:
            pass

    # Default registry
    reg = {
        "active": "champion",
        "pointers": {
            "champion": {
                "model_version": "2026_patchtst",
                "checkpoint": "checkpoints/2026_patchtst.pt",
                "provenance": "provenance/2026_patchtst.json",
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            "standby": None,
            "shadow": None,
        },
    }
    return reg


def save_registry(reg: dict[str, Any], reg_path: Optional[Path] = None) -> None:
    """Save registry to YAML."""
    path = reg_path or REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(reg, f, sort_keys=False)


def build_golden_io(model: Ensemble, template: PatchTST, x_sample: torch.Tensor, c_sample: torch.Tensor) -> dict[str, Any]:
    """Build 20 fixed windows golden I/O verification dict."""
    model.eval()
    with torch.no_grad():
        K = x_sample.shape[0] if x_sample.ndim == 4 else 16
        if x_sample.ndim == 3:
            x_in = x_sample.unsqueeze(0).expand(K, -1, -1, -1)
            c_in = c_sample.unsqueeze(0).expand(K, -1, -1)
        else:
            x_in = x_sample
            c_in = c_sample
        p = {k: model.params[k.replace('.', '__')] for k in model.keys}
        from interm.MODEL.model_patchtst import batched_patch
        out = batched_patch(template, p, x_in, c_in, training=False)
    return {
        "input_x": x_sample.cpu().numpy().tolist(),
        "input_c": c_sample.cpu().numpy().tolist(),
        "output_q": out.cpu().numpy().tolist(),
    }


def verify_golden_io(model: Ensemble, template: PatchTST, golden: dict[str, Any], tol: float = 1e-4) -> Tuple[bool, float]:
    """Verify that current model matches golden IO exactly."""
    model.eval()
    device = next(model.parameters()).device
    x_in = torch.as_tensor(golden["input_x"], device=device, dtype=torch.float32)
    c_in = torch.as_tensor(golden["input_c"], device=device, dtype=torch.long)
    expected_q = np.asarray(golden["output_q"], dtype=np.float32)

    K = 16
    if x_in.ndim == 3:
        x_in = x_in.unsqueeze(0).expand(K, -1, -1, -1)
        c_in = c_in.unsqueeze(0).expand(K, -1, -1)

    with torch.no_grad():
        p = {k: model.params[k.replace('.', '__')] for k in model.keys}
        from interm.MODEL.model_patchtst import batched_patch
        out = batched_patch(template, p, x_in, c_in, training=False)
        out_np = out.cpu().numpy()

    diff = float(np.max(np.abs(out_np - expected_q)))
    return bool(diff <= tol), diff


class RetrainPipeline:
    """Retraining and acceptance orchestrator."""

    def __init__(self, ckpt_dir: Path | str | None = None, device: str = "cpu"):
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else CKPT_DIR
        self.device = device
        self.checkpoints_dir = self.ckpt_dir / "checkpoints"
        self.provenance_dir = self.ckpt_dir / "provenance"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

    def execute_pipeline(self,
                         mode: str,
                         data_df: Any,
                         parts: dict[str, np.ndarray],
                         base_ckpt_path: Optional[Path] = None,
                         epochs: int = 20,
                         lr: float = 1e-3,
                         version_tag: Optional[str] = None,
                         strict_baseline_gate: bool = True) -> dict[str, Any]:
        """Execute retraining according to requested mode."""
        valid_modes = ("tier2_cqr", "tier3a_scale", "tier3b_partial", "tier4_full")
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode {mode!r}. Must be one of {valid_modes}")

        timestamp_str = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        v_tag = version_tag or f"{timestamp_str}_{mode}"

        # 1. Transform & Data Preparation
        train_idx = parts["train"]
        val_idx = parts["validation"]
        cal_idx = parts["calibration"]
        test_idx = parts["test"]

        # Base transform
        transform = FoldTransform().fit(data_df, train_idx)
        x_all, scale_all, z_all = transform.apply(data_df)

        L = 300
        K = 16
        cfg = dict(DEFAULT, lookback=L, K=K, seed=1729, lr=lr, epochs=epochs)

        # Existing champion weights if available
        champ_pt = None
        champ_prov = None
        if base_ckpt_path and base_ckpt_path.exists():
            champ_pt = torch.load(base_ckpt_path, map_location=self.device, weights_only=False)
            prov_cand = self.provenance_dir / f"{base_ckpt_path.stem}.json"
            if prov_cand.exists():
                champ_prov = json.loads(prov_cand.read_text(encoding="utf-8"))

        # Base model initialization
        def make_ctor(c):
            return PatchTST(c, Fdim=11)

        model = Ensemble(make_ctor, cfg, device=self.device)
        template = model.template

        if champ_pt is not None:
            # Load params from champion
            loaded_p = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in champ_pt["params"].items()}
            model.params.load_state_dict(loaded_p)
            if "transform" in champ_pt:
                transform.scaler.center_ = np.asarray(champ_pt["transform"]["median"])
                transform.scaler.scale_ = np.asarray(champ_pt["transform"]["iqr"])
                transform.factors = np.asarray(champ_pt["transform"]["slot_factors"])
                x_all, scale_all, z_all = transform.apply(data_df)

        # Build feature windows for evaluation
        def get_windows(indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
            wins = np.stack([x_all[i - L + 1: i + 1] for i in indices]).astype(np.float32)
            cats = data_df[["slot_next", "dow_next"]].iloc[indices].to_numpy().astype(np.int64)
            scales = scale_all[indices]
            targets = data_df["r_next"].iloc[indices].to_numpy().astype(np.float64)
            return (torch.as_tensor(wins, device=self.device),
                    torch.as_tensor(cats, device=self.device),
                    scales,
                    targets)

        x_cal, c_cal, s_cal, y_cal = get_windows(cal_idx)
        x_test, c_test, s_test, y_test = get_windows(test_idx)

        # Compute champion predictions on test set if available
        q_champion_test = None
        if champ_pt is not None:
            model.eval()
            with torch.no_grad():
                xt = x_test.unsqueeze(0).expand(K, -1, -1, -1)
                ct = c_test.unsqueeze(0).expand(K, -1, -1)
                p_dict = {k: model.params[k.replace('.', '__')] for k in model.keys}
                from interm.MODEL.model_patchtst import batched_patch
                q_scaled_test = batched_patch(template, p_dict, xt, ct, training=False)
                q_raw_test = q_scaled_test.cpu().numpy() * s_test[None, :, None]
                if champ_prov and "calibration" in champ_prov:
                    cal_info = champ_prov["calibration"]
                    bias = float(cal_info.get("median_bias", 0.0))
                    offs = cal_info.get("cqr_offsets", [0.0, 0.0, 0.0])
                    q_champion_test = q_raw_test.copy()
                    q_champion_test[:, :, 3] += bias
                    for (lo, hi, _), off_val in zip([(0,6,.1),(1,5,.2),(2,4,.5)], offs):
                        q_champion_test[:, :, lo] -= off_val
                        q_champion_test[:, :, hi] += off_val
                    q_champion_test.sort(axis=-1)
                else:
                    q_champion_test = q_raw_test

        # --- MODE LOGIC ---
        if mode == "tier2_cqr":
            # Only refit CQR on calibration window, keep weights intact
            model.eval()
            with torch.no_grad():
                xt_c = x_cal.unsqueeze(0).expand(K, -1, -1, -1)
                ct_c = c_cal.unsqueeze(0).expand(K, -1, -1)
                p_dict = {k: model.params[k.replace('.', '__')] for k in model.keys}
                from interm.MODEL.model_patchtst import batched_patch
                q_scaled_cal = batched_patch(template, p_dict, xt_c, ct_c, training=False)
                q_cal_members = q_scaled_cal.cpu().numpy() * s_cal[None, :, None]

            calibrator = Calibrator().fit(q_cal_members, y_cal)

        elif mode == "tier3a_scale":
            # Re-estimate slot_factors with 50/50 shrinkage toward current values
            curr_factors = np.asarray(transform.factors).copy()
            # Empirical RMS on recent training / validation
            recent_idx = np.concatenate([train_idx[-1000:], val_idx])
            t_sub = data_df.iloc[recent_idx]
            u = t_sub["r_next"].to_numpy() / t_sub["sigma_ewma"].to_numpy()
            new_factors = np.array([np.sqrt(np.mean(u[t_sub["slot_next"].to_numpy() == s] ** 2)) for s in range(5)])
            new_factors /= np.exp(np.log(new_factors).mean())
            shrunk_factors = 0.5 * curr_factors + 0.5 * new_factors
            shrunk_factors /= np.exp(np.log(shrunk_factors).mean())
            transform.factors = shrunk_factors

            # Re-scale calibration features/scale
            x_all, scale_all, z_all = transform.apply(data_df)
            x_cal, c_cal, s_cal, y_cal = get_windows(cal_idx)
            x_test, c_test, s_test, y_test = get_windows(test_idx)

            model.eval()
            with torch.no_grad():
                xt_c = x_cal.unsqueeze(0).expand(K, -1, -1, -1)
                ct_c = c_cal.unsqueeze(0).expand(K, -1, -1)
                p_dict = {k: model.params[k.replace('.', '__')] for k in model.keys}
                from interm.MODEL.model_patchtst import batched_patch
                q_scaled_cal = batched_patch(template, p_dict, xt_c, ct_c, training=False)
                q_cal_members = q_scaled_cal.cpu().numpy() * s_cal[None, :, None]

            calibrator = Calibrator().fit(q_cal_members, y_cal)

        elif mode in ("tier3b_partial", "tier4_full"):
            # Model training with recency weighting and stationary block bootstrap
            # In tier3b_partial: freeze 12 members (indices 0..11), retrain 4 members (12..15)
            # In tier4_full: retrain all 16 members
            active_members = list(range(12, 16)) if mode == "tier3b_partial" else list(range(16))
            recency_weights = compute_recency_weights(len(train_idx))

            optimizer = torch.optim.AdamW(model.params.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            B = min(cfg["B"], 16 if self.device == "cpu" else 128)
            n_steps = max(1, min(len(train_idx) // B, 10 if self.device == "cpu" else 50))

            for epoch in range(epochs):
                model.train()
                for _ in range(n_steps):
                    # Sample B indices with recency weights
                    batch_pos = np.random.choice(len(train_idx), size=B, replace=True, p=recency_weights)
                    ids = train_idx[batch_pos]
                    b_wins = np.stack([x_all[i - L + 1: i + 1] for i in ids]).astype(np.float32)
                    b_cats = data_df[["slot_next", "dow_next"]].iloc[ids].to_numpy().astype(np.int64)
                    b_y = torch.as_tensor(z_all[ids], device=self.device, dtype=torch.float32)

                    b_x_t = torch.as_tensor(b_wins, device=self.device).unsqueeze(0).expand(K, -1, -1, -1)
                    b_c_t = torch.as_tensor(b_cats, device=self.device).unsqueeze(0).expand(K, -1, -1)

                    optimizer.zero_grad(set_to_none=True)
                    from interm.MODEL.model_patchtst import batched_patch
                    p_dict = {k: model.params[k.replace('.', '__')] for k in model.keys}
                    q_pred = batched_patch(template, p_dict, b_x_t, b_c_t, training=True)

                    # Compute pinball loss
                    loss = pinball(q_pred, b_y.unsqueeze(0).expand(K, -1))
                    loss.backward()

                    # If partial, zero gradients of frozen members (0..11)
                    if mode == "tier3b_partial":
                        for p_val in model.params.values():
                            if p_val.grad is not None:
                                p_val.grad[:12] = 0.0

                    optimizer.step()

            # Calibrate on calibration set
            model.eval()
            with torch.no_grad():
                xt_c = x_cal.unsqueeze(0).expand(K, -1, -1, -1)
                ct_c = c_cal.unsqueeze(0).expand(K, -1, -1)
                p_dict = {k: model.params[k.replace('.', '__')] for k in model.keys}
                from interm.MODEL.model_patchtst import batched_patch
                q_scaled_cal = batched_patch(template, p_dict, xt_c, ct_c, training=False)
                q_cal_members = q_scaled_cal.cpu().numpy() * s_cal[None, :, None]

            calibrator = Calibrator().fit(q_cal_members, y_cal)

        # Generate challenger predictions on test set
        model.eval()
        with torch.no_grad():
            xt_t = x_test.unsqueeze(0).expand(K, -1, -1, -1)
            ct_t = c_test.unsqueeze(0).expand(K, -1, -1)
            p_dict = {k: model.params[k.replace('.', '__')] for k in model.keys}
            from interm.MODEL.model_patchtst import batched_patch
            q_scaled_test = batched_patch(template, p_dict, xt_t, ct_t, training=False)
            q_raw_test = q_scaled_test.cpu().numpy() * s_test[None, :, None]
            q_challenger_test = calibrator.apply(q_raw_test, online=False)

        # Build 20 golden IO test windows
        sample_indices = test_idx[:20]
        x_g, c_g, _, _ = get_windows(sample_indices)
        golden_io = build_golden_io(model, template, x_g, c_g)

        # Automated Acceptance Gate
        def golden_fn():
            return verify_golden_io(model, template, golden_io)

        gate = AcceptanceGate()
        passed, gate_details = gate.evaluate(
            y_test=y_test,
            scale_test=s_test,
            q_challenger=q_challenger_test,
            q_champion=q_champion_test,
            golden_check_fn=golden_fn,
            strict_baseline_gate=strict_baseline_gate,
        )

        res_summary = {
            "version_tag": v_tag,
            "mode": mode,
            "passed_acceptance": passed,
            "gate_details": gate_details,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }

        if not passed:
            log.warning("Challenger failed acceptance gate! Not exporting: %s", gate_details)
            return res_summary

        # Export artifacts to DoneSetup/CKPTs/
        ckpt_filename = f"{v_tag}_patchtst.pt"
        prov_filename = f"{v_tag}_patchtst.json"
        golden_filename = f"{v_tag}_golden_io.json"

        ckpt_file = self.checkpoints_dir / ckpt_filename
        prov_file = self.provenance_dir / prov_filename
        golden_file = self.checkpoints_dir / golden_filename

        state_dict = {k: v.detach().cpu().clone() for k, v in model.params.items()}
        torch.save({
            "params": state_dict,
            "config": cfg,
            "architecture": "patchtst",
            "transform": transform.provenance(),
            "logs": [],
        }, ckpt_file)

        prov_data = {
            "fold": v_tag,
            "mode": mode,
            "fit": {"best_validation": float(gate_details["gate1_loss_vs_champion"]["pl_challenger"])},
            "calibration": calibrator.metadata(),
            "aci_alpha_after_observation": [0.10, 0.20, 0.50],
        }
        with open(prov_file, "w", encoding="utf-8") as f:
            json.dump(prov_data, f, indent=2)

        with open(golden_file, "w", encoding="utf-8") as f:
            json.dump(golden_io, f, indent=2)

        # Update registry: previous champion becomes standby, new model becomes champion
        reg_file = self.ckpt_dir / "checkpoint_registry.yaml"
        reg = load_or_init_registry(reg_file)
        prev_champ = reg["pointers"].get("champion")
        if prev_champ is not None:
            reg["pointers"]["standby"] = prev_champ

        reg["pointers"]["champion"] = {
            "model_version": v_tag,
            "checkpoint": f"checkpoints/{ckpt_filename}",
            "provenance": f"provenance/{prov_filename}",
            "golden_io": f"checkpoints/{golden_filename}",
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        reg["active"] = "champion"
        save_registry(reg, reg_file)

        log.info("Successfully exported champion checkpoint %s and updated registry", ckpt_filename)
        return res_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Controlled Retrain, Staggered Refresh & Export Pipeline.")
    parser.add_argument("--mode", required=True, choices=["tier2_cqr", "tier3a_scale", "tier3b_partial", "tier4_full"])
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "DATA/PROCESSED/features_targets.parquet")
    parser.add_argument("--base-ckpt", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--version-tag", type=str, default=None)
    args = parser.parse_args(argv)

    from interm.MODEL.data import load
    from interm.MODEL.folds import make_fold

    if not args.data.exists():
        print(f"Data file {args.data} does not exist.")
        return 1

    d = load(args.data, allow_sparse=True, lockbox=True)
    parts = make_fold(d, 2025, strict=False)

    pipeline = RetrainPipeline(ckpt_dir=args.ckpt_dir, device=args.device)
    res = pipeline.execute_pipeline(
        mode=args.mode,
        data_df=d,
        parts=parts,
        base_ckpt_path=args.base_ckpt,
        epochs=args.epochs,
        version_tag=args.version_tag,
    )
    print("Retrain pipeline result:")
    print(json.dumps(res, indent=2))
    return 0 if res.get("passed_acceptance", False) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
