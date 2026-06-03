"""Shared helpers for EEV CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eevCode.yaml"


def load_eev_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root = PROJECT_ROOT / cfg.get("data_root", "scout_data/eevCode")
    paths = cfg.get("paths") or {}
    resolved = {k: PROJECT_ROOT / v for k, v in paths.items()}
    cfg["_project_root"] = PROJECT_ROOT
    cfg["_data_root"] = root
    cfg["_paths"] = resolved
    return cfg


def path_cfg(cfg: dict[str, Any], key: str) -> Path:
    return cfg["_paths"][key]
