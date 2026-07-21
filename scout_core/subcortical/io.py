"""Load/save preds_subcortical.npz session artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scout_core.subcortical.atlas import (
    N_SUBCORTICAL_VOXELS,
    SUBCORTICAL_REGION_LABELS,
    load_subcortical_manifest,
)

SUBCORTICAL_NPZ_NAME = "preds_subcortical.npz"


def subcortical_npz_path(session_dir: Path) -> Path:
    return session_dir / SUBCORTICAL_NPZ_NAME


def save_subcortical_npz(
    session_dir: Path,
    preds: np.ndarray,
    *,
    atlas_id: str | None = None,
    tribe_checkpoint: str | None = None,
    region_labels: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write preds_subcortical.npz; returns path."""
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2 or preds.shape[1] != N_SUBCORTICAL_VOXELS:
        raise ValueError(
            f"subcortical preds must be (T, {N_SUBCORTICAL_VOXELS}), got {preds.shape}"
        )

    session_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_subcortical_manifest()
    out_path = subcortical_npz_path(session_dir)
    labels = region_labels or manifest.get("region_labels") or list(SUBCORTICAL_REGION_LABELS)
    meta = {
        "shape": [int(preds.shape[0]), int(preds.shape[1])],
        "atlas_id": atlas_id or manifest.get("atlas_id", "harvard_oxford_subcortical_tribev2"),
        "region_labels": labels,
        "tribe_checkpoint": tribe_checkpoint or manifest.get(
            "tribe_subcortical_checkpoint", "facebook/tribev2-subcortical"
        ),
    }
    if extra_meta:
        meta.update(extra_meta)

    np.savez_compressed(
        out_path,
        preds=preds,
        atlas_id=np.array(meta["atlas_id"]),
        region_labels=np.array(json.dumps(labels)),
        tribe_checkpoint=np.array(meta["tribe_checkpoint"]),
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
    )
    return out_path


def load_subcortical_preds(
    session_dir: Path,
    *,
    require: bool = False,
) -> np.ndarray | None:
    """Load (T, 8802) array or None if missing."""
    path = subcortical_npz_path(session_dir)
    if not path.is_file():
        if require:
            raise FileNotFoundError(f"Missing subcortical preds: {path}")
        return None
    data = np.load(path)
    preds = np.asarray(data["preds"], dtype=np.float32)
    if preds.ndim != 2 or preds.shape[1] != N_SUBCORTICAL_VOXELS:
        raise ValueError(f"invalid subcortical shape in {path}: {preds.shape}")
    return preds


def validate_subcortical_alignment(cortical_t: int, subcortical_preds: np.ndarray) -> dict[str, Any]:
    """Assert T dimension matches cortical preds row count."""
    sub_t = int(subcortical_preds.shape[0])
    ok = sub_t == cortical_t
    return {
        "ok": ok,
        "n_cortical": cortical_t,
        "n_subcortical": sub_t,
        "message": (
            f"subcortical T={sub_t} vs cortical T={cortical_t}"
            + ("" if ok else " — MISALIGNED")
        ),
    }
