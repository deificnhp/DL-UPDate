"""Logging configuration for inference_live."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a given sub-name under inference_live."""
    full_name = f"inference_live.{name}" if not name.startswith("inference_live") else name
    return logging.getLogger(full_name)


def setup_logging(log_level: str = "INFO", log_file: Path | str | None = None) -> None:
    """Configure the root logger and inference_live formatters."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(p)))

    for h in handlers:
        h.setFormatter(formatter)

    root = logging.getLogger("inference_live")
    root.setLevel(level)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
