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


def extract_heatmap_modal(
    frame_path: Path,
    capture_h: int,
    capture_w: int,
    *,
    inference: Any | None = None,
) -> np.ndarray:
    """Call ``TribeInference.extract_frame_attention`` on Modal.

    When ``inference`` is provided, the caller must already be inside
    ``modal.enable_output()`` + ``app.run()`` (one context for many frames).
    """
    from tribe import TribeInference, app

    import modal

    frame_bytes = frame_path.read_bytes()

    def _remote_call(inst: Any) -> bytes:
        return inst.extract_frame_attention.remote(frame_bytes, capture_h, capture_w)

    if inference is not None:
        result_bytes = _remote_call(inference)
    else:
        with modal.enable_output():
            with app.run():
                inst = TribeInference()
                result_bytes = _remote_call(inst)

    return np.load(io.BytesIO(result_bytes)).astype(np.float32)


def read_heatmaps_manifest(heatmaps_dir: Path) -> dict[int, dict[str, Any]]:
    """Return ``{t_idx: entry}`` from an existing heatmaps/manifest.json, or empty dict."""
    path = heatmaps_dir / "manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {int(e["t_idx"]): e for e in data.get("entries", []) if "t_idx" in e}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {}


def write_heatmaps_manifest(
    heatmaps_dir: Path,
    entries: list[dict[str, Any]],
    *,
    merge: bool = False,
) -> None:
    """Write heatmaps/manifest.json.

    When ``merge=True``, existing entries are preserved unless a new entry for
    the same ``t_idx`` replaces them, so provenance from earlier runs is not lost.
    """
    path = heatmaps_dir / "manifest.json"
    if merge and path.is_file():
        existing = read_heatmaps_manifest(heatmaps_dir)
        for e in entries:
            existing[int(e["t_idx"])] = e
        final = sorted(existing.values(), key=lambda x: int(x["t_idx"]))
    else:
        final = entries
    path.write_text(json.dumps({"entries": final}, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
