"""
Shared utility helpers: config loading, logging setup, and small
filesystem helpers used across the project.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Raises FileNotFoundError with a clear message if the file is missing,
    since a silent failure here would surface as a confusing KeyError later.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    """Resolve a path from config relative to the project root, unless
    it is already absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path_str: str) -> Path:
    """Create a directory (and parents) if it doesn't already exist."""
    p = resolve_path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(logging_cfg: Dict[str, Any]) -> logging.Logger:
    """Configure a project-wide logger that writes to both console and
    a rotating-free simple file handler (kept dependency-free)."""
    log_dir = ensure_dir(logging_cfg.get("log_dir", "logs"))
    log_level = getattr(logging, str(logging_cfg.get("log_level", "INFO")).upper(), logging.INFO)
    log_file = log_dir / logging_cfg.get("log_filename", "app.log")

    logger = logging.getLogger("formal_attire_detection")
    logger.setLevel(log_level)
    logger.propagate = False

    if logger.handlers:
        # Avoid duplicate handlers if setup_logging is called more than once
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
