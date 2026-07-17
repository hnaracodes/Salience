"""Validate Horikawa TRIBE intermediate NPZ artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

VALID_SOURCES = {
    "tribev2_subcortical_checkpoint",
    "tribev2_subcortical_checkpoint_regions",
}


def npz_path(inter_dir: Path, sid: str) -> Path:
    p = inter_dir / f"{sid}_both.npz"
    if p.is_file():
        return p
    if sid.isdigit():
        alt = inter_dir / f"{int(sid):04d}_both.npz"
        if alt.is_file():
            return alt
    return p


def classify_npz(path: Path) -> str:
    """Return ok | missing | stale | invalid."""
    if not path.is_file():
        return "missing"
    try:
        with np.load(path, allow_pickle=True) as d:
            if "preds" not in d or "preds_subcortical" not in d:
                return "invalid"
            cort = d["preds"]
            sub = d["preds_subcortical"]
            if cort.ndim != 2 or sub.ndim != 2:
                return "invalid"
            if cort.shape[1] != 20484 or sub.shape[1] != 8802:
                return "invalid"
            if not np.all(np.isfinite(cort)) or not np.all(np.isfinite(sub)):
                return "invalid"
            src = str(np.asarray(d["prediction_source"])) if "prediction_source" in d else ""
            if src in VALID_SOURCES:
                return "ok"
            return "stale"
    except OSError:
        return "invalid"
