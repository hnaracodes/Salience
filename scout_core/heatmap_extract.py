"""Extract DINOv2 attention heatmaps via Modal or local placeholder."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from scout_core.session_align import load_manifest, tr_duration_from_manifest


def capture_size_from_manifest(session_dir: Path, default_h: int = 1080, default_w: int = 1920) -> tuple[int, int]:
    manifest = load_manifest(session_dir)
    if manifest:
        cap = manifest.get("capture") or {}
        return int(cap.get("height", default_h)), int(cap.get("width", default_w))
    return default_h, default_w


def fps_from_manifest(session_dir: Path, default_fps: float = 1.0) -> float:
    manifest = load_manifest(session_dir)
    if manifest:
        tr = tr_duration_from_manifest(manifest)
        return 1.0 / tr if tr > 0 else default_fps
    return default_fps


def extract_heatmap_modal(frame_path: Path, capture_h: int, capture_w: int) -> np.ndarray:
    """Call ``TribeInference.extract_frame_attention`` on Modal."""
    from tribe import TribeInference

    inference = TribeInference()
    frame_bytes = frame_path.read_bytes()
    result_bytes = inference.extract_frame_attention.remote(frame_bytes, capture_h, capture_w)
    return np.load(io.BytesIO(result_bytes)).astype(np.float32)


def write_heatmaps_manifest(
    heatmaps_dir: Path,
    entries: list[dict[str, Any]],
) -> None:
    path = heatmaps_dir / "manifest.json"
    path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
