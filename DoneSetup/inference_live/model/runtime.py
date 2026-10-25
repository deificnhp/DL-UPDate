"""Model runtime: load a trained checkpoint and run the ensemble forward.

The inference recipe, verified bit-exactly against the training project's saved member
predictions (MODEL/ARTIFACTS/members/lockbox/2026_PatchTST-Q.npz):

    x     = clip((features - median) / iqr, -5, 5)          # frozen transform
    win   = features[t-L+1 : t+1]                            # L = 300
    c     = (slot_next[t], dow_next[t])                      # TARGET bar's categoricals
    q_raw = model(win, c)                                   # (K, 7) scaled units
    q     = q_raw * (sigma_ewma[t] * slot_factors[slot_next[t]])   # raw log-return units

Precision: the shipped member predictions were produced under bf16 autocast on CUDA.
Re-running the same recipe under bf16 autocast reproduces them EXACTLY (max abs error
0.0). Re-running in fp32 on the same weights differs by ~2e-5 per member. The default
here is bf16 autocast on CUDA, with fp32 on CPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..config import ModelConfig
from ..core.scaling import FrozenTransform
from ..logging_setup import get_logger
from .architectures import Ensemble, LSTM, PatchTST

log = get_logger("model.runtime")


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return requested


@dataclass
class LoadedModel:
    """One trained ensemble, ready for inference."""
    name: str                 # e.g. "patchtst"
    fold: str                 # e.g. "2026"
    ensemble: Ensemble
    transform: FrozenTransform
    config: dict
    device: str
    use_bf16: bool
    taus: List[float]

    @property
    def lookback(self) -> int:
        return int(self.config["lookback"])

    @property
    def K(self) -> int:
        return int(self.config["K"])


def _ctor_for(architecture: str):
    if architecture == "patchtst":
        return lambda cfg: PatchTST(cfg, Fdim=11)
    if architecture == "lstm":
        return lambda cfg: LSTM(cfg, Fdim=11)
    raise ValueError(f"unknown architecture: {architecture!r}")


def load_model(checkpoint_path: Path,
               mcfg: Optional[ModelConfig] = None) -> LoadedModel:
    """Load a trained checkpoint and build the inference-time ensemble.

    The checkpoint contains: params, config, architecture, transform, logs.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("params", "config", "architecture", "transform"):
        if key not in ck:
            raise ValueError(f"checkpoint {checkpoint_path.name} is missing '{key}'")

    mcfg = mcfg or ModelConfig()
    device = resolve_device(mcfg.device)
    cfg = dict(ck["config"])

    # The frozen config governs; a mismatch means the checkpoint was trained with a
    # different geometry than the one this system is configured for.
    for key, want in (("lookback", mcfg.lookback), ("K", mcfg.K)):
        if key in cfg and int(cfg[key]) != int(want):
            raise ValueError(
                f"checkpoint {checkpoint_path.name}: {key}={cfg[key]} but the live "
                f"configuration expects {want}. Refusing to run with a mismatched "
                f"geometry.")

    params = {k: v.to(device) if torch.is_tensor(v) else v for k, v in ck["params"].items()}
    model = Ensemble(_ctor_for(ck["architecture"]), cfg, device=device, state_dict=params)
    model.eval()

    transform = FrozenTransform.from_provenance(ck["transform"])

    # bf16 autocast needs bf16-capable hardware; on CPU we run fp32.
    use_bf16 = mcfg.use_bf16_autocast and device == "cuda"
    if mcfg.use_bf16_autocast and device != "cuda":
        log.warning("bf16 autocast requested but the device is %s; running fp32 "
                    "(predictions will differ from the shipped ones by ~2e-5 per member)",
                    device)

    name = str(ck["architecture"])
    fold = checkpoint_path.stem.split("_")[0]
    log.info("loaded %s fold %s: L=%d K=%d device=%s bf16=%s",
             name, fold, cfg["lookback"], cfg["K"], device, use_bf16)

    return LoadedModel(name=name, fold=fold, ensemble=model, transform=transform,
                       config=cfg, device=device, use_bf16=use_bf16,
                       taus=list(mcfg.taus))


@torch.no_grad()
def predict_members(model: LoadedModel,
                    windows: np.ndarray,
                    cats: np.ndarray,
                    scale: np.ndarray) -> np.ndarray:
    """Run the ensemble over a batch of decision rows.

    windows : (B, L, 11)  scaled feature windows, row i = features[t_i-L+1 .. t_i]
    cats    : (B, 2)      (slot_next, dow_next) of the TARGET bar for each row
    scale   : (B,)        the multiplicative scale to raw log-return units

    Returns (K, B, 7): per-member quantiles in raw log-return units.
    """
    if windows.ndim != 3:
        raise ValueError(f"windows must be (B, L, F), got {windows.shape}")
    if cats.ndim != 2 or cats.shape[1] != 2:
        raise ValueError(f"cats must be (B, 2), got {cats.shape}")
    B, L, F = windows.shape
    if len(scale) != B:
        raise ValueError(f"scale must have {B} entries, got {len(scale)}")
    if F != model.transform.median.shape[0]:
        raise ValueError(f"window feature count {F} does not match the frozen transform")

    dev = model.device
    x = torch.as_tensor(windows, dtype=torch.float32, device=dev)
    c = torch.as_tensor(cats, dtype=torch.long, device=dev)
    s = torch.as_tensor(scale, dtype=torch.float32, device=dev)
    K = model.K

    xt = x.unsqueeze(0).expand(K, -1, -1, -1)      # (K, B, L, F)
    ct = c.unsqueeze(0).expand(K, -1, -1)           # (K, B, 2)

    if model.use_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = model.ensemble(xt, ct)
    else:
        q = model.ensemble(xt, ct)

    q = q.float() * s.view(1, -1, 1)               # (K, B, 7) in raw log-return units
    return q.cpu().numpy()


def load_provenance(provenance_path: Path) -> dict:
    """Load a fold's provenance JSON (calibration + transform + fold identity)."""
    provenance_path = Path(provenance_path)
    if not provenance_path.exists():
        raise FileNotFoundError(f"provenance not found: {provenance_path}")
    with provenance_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_artifacts(checkpoint_dir: Path,
                   fold: str,
                   architectures: tuple) -> Dict[str, Path]:
    """Locate {arch: checkpoint_path} and {arch: provenance_path} under `checkpoint_dir`.

    Expected layout (what the README tells you to copy):
        CKPTs/checkpoints/{fold}_{arch}.pt
        CKPTs/provenance/{fold}_{arch}.json
    """
    checkpoint_dir = Path(checkpoint_dir)
    out: Dict[str, Path] = {}
    for arch in architectures:
        ck = checkpoint_dir / "checkpoints" / f"{fold}_{arch}.pt"
        pr = checkpoint_dir / "provenance" / f"{fold}_{arch}.json"
        if not ck.exists():
            raise FileNotFoundError(f"checkpoint not found: {ck}")
        if not pr.exists():
            raise FileNotFoundError(f"provenance not found: {pr}")
        out[f"checkpoint:{arch}"] = ck
        out[f"provenance:{arch}"] = pr
    return out
