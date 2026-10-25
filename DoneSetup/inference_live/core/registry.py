"""Checkpoint registry loader and pointer resolver."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from ..logging_setup import get_logger

log = get_logger("core.registry")


def load_registry(ckpt_dir: Path | str) -> Optional[dict[str, Any]]:
    """Load checkpoint_registry.yaml if present."""
    reg_file = Path(ckpt_dir) / "checkpoint_registry.yaml"
    if not reg_file.exists():
        return None
    try:
        with open(reg_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "pointers" in data:
                return data
    except Exception as exc:
        log.warning("Could not load checkpoint registry from %s: %s", reg_file, exc)
    return None


def resolve_registry_pointer(ckpt_dir: Path | str, role: str = "active") -> Optional[dict[str, Any]]:
    """Resolve active or specified pointer (champion, standby, shadow)."""
    reg = load_registry(ckpt_dir)
    if reg is None:
        return None
    active_role = reg.get("active", "champion") if role == "active" else role
    ptr = reg.get("pointers", {}).get(active_role)
    if not ptr:
        return None
    # Resolve relative paths relative to ckpt_dir
    base = Path(ckpt_dir)
    ckpt_path = base / ptr["checkpoint"]
    prov_path = base / ptr["provenance"]
    return {
        "role": active_role,
        "model_version": ptr.get("model_version", active_role),
        "checkpoint_path": ckpt_path,
        "provenance_path": prov_path,
    }
