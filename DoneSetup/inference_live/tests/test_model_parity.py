"""Model parity: the live model runtime vs the training project's saved predictions.

The training project saved the per-member raw quantiles of every fold's test set:
    MODEL/ARTIFACTS/members/lockbox/2026_PatchTST-Q.npz   -> raw (K, n, 7), row_ids
    MODEL/ARTIFACTS/members/lockbox/2026_C1-LSTM.npz

The live runtime must reproduce those numbers from the same checkpoint, the same
features and the same transform. Under bf16 autocast on CUDA the reproduction is
bit-exact (max abs error 0.0); in fp32 it differs by ~2e-5 per member.

The test needs the training project; it is skipped when it is not available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ..config import ModelConfig
from ..core.features import FEATURE_ORDER, feature_matrix
from ..model.runtime import load_model, predict_members

TRAINING_ROOT = Path(r"D:\DataScience\interm\RED SPRITE\DL-Layer")
CLEAN = TRAINING_ROOT / "DATA/PROCESSED/nq_4h_clean.parquet"
FEATS = TRAINING_ROOT / "DATA/PROCESSED/features_targets.parquet"
CKPT = TRAINING_ROOT / "MODEL/ARTIFACTS/checkpoints"
MEMBERS = TRAINING_ROOT / "MODEL/ARTIFACTS/members/lockbox"

pytestmark = pytest.mark.skipif(
    not (CLEAN.exists() and FEATS.exists() and CKPT.exists()),
    reason="the training project's artifacts are not available")

# The member npz files were produced under bf16 autocast on CUDA.
BF16_TOL = 0.0
FP32_TOL = 5e-5


def _load_frame() -> pd.DataFrame:
    """The training feature file, in the same order the training project used."""
    d = pd.read_parquet(FEATS)
    d["bar_start_utc"] = pd.to_datetime(d["bar_start_utc"], utc=True)
    d["bar_end_utc"] = pd.to_datetime(d["bar_end_utc"], utc=True)
    d = d.sort_values("bar_end_utc", kind="stable").reset_index(drop=True)
    return d


def _windows(d: pd.DataFrame, row_ids: np.ndarray, L: int, transform) -> np.ndarray:
    x = transform.scale_features(feature_matrix(d))
    out = np.empty((len(row_ids), L, len(FEATURE_ORDER)), dtype=np.float32)
    for i, t in enumerate(row_ids):
        out[i] = x[t - L + 1: t + 1]
    return out


@pytest.mark.parametrize("arch,npz_name", [("patchtst", "2026_PatchTST-Q.npz"),
                                            ("lstm", "2026_C1-LSTM.npz")])
def test_model_reproduces_saved_member_predictions(arch, npz_name):
    npz_path = MEMBERS / npz_name
    if not npz_path.exists():
        pytest.skip(f"missing member predictions: {npz_path}")

    d = _load_frame()
    mem = np.load(npz_path)
    raw_saved = mem["raw"]          # (K, n, 7)
    row_ids = mem["row_ids"]        # positions in the full frame

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda"
    mcfg = ModelConfig(device=device, use_bf16_autocast=use_bf16)
    model = load_model(CKPT / f"2026_{arch}.pt", mcfg)
    L = model.lookback

    # a spread of rows: the first, the last, and some in the middle
    sel = np.array(sorted(set(list(row_ids[:5]) + list(row_ids[-5:])
                             + list(np.random.default_rng(0).choice(
                                 row_ids, size=min(40, len(row_ids)), replace=False)))))
    pos = {int(r): i for i, r in enumerate(row_ids)}
    idx = np.array([pos[int(r)] for r in sel])

    feats = d[FEATURE_ORDER].to_numpy(dtype=float)
    x_all = model.transform.scale_features(feats)
    cats = d[["slot_next", "dow_next"]].to_numpy().astype(np.int64)
    sigma = d["sigma_ewma"].to_numpy(dtype=float)
    scale = model.transform.target_scale(sigma, cats[:, 0])

    win = np.stack([x_all[t - L + 1: t + 1] for t in sel]).astype(np.float32)
    c = cats[sel]
    sc = scale[sel]

    q = predict_members(model, win, c, sc)      # (K, n, 7)
    saved = raw_saved[:, idx, :]

    err = float(np.max(np.abs(q - saved)))
    tol = BF16_TOL if use_bf16 else FP32_TOL
    assert err <= tol, (
        f"{arch}: max |live - saved| = {err:.3e} exceeds the tolerance {tol:.1e} "
        f"(device={device}, bf16={use_bf16})")


@pytest.mark.parametrize("arch", ["patchtst", "lstm"])
def test_saved_transform_is_the_frozen_one(arch):
    """The transform used at inference must be the one stored in the checkpoint."""
    model = load_model(CKPT / f"2026_{arch}.pt", ModelConfig(device="cpu"))
    ck = torch.load(CKPT / f"2026_{arch}.pt", map_location="cpu", weights_only=False)
    t = ck["transform"]
    assert np.allclose(model.transform.median, np.asarray(t["median"]))
    assert np.allclose(model.transform.iqr, np.asarray(t["iqr"]))
    assert np.allclose(model.transform.slot_factors, np.asarray(t["slot_factors"]))
    # the seasonal factors are geometrically normalised to 1
    assert abs(float(np.exp(np.log(model.transform.slot_factors).mean())) - 1.0) < 1e-9
