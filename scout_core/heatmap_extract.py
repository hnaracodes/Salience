"""Heatmap extraction adapters and provenance helpers."""

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


def extract_heatmap_deepgaze(
    frame_path: Path,
    capture_h: int,
    capture_w: int,
    *,
    model: Any,
    device: str,
    pixels_per_degree: float,
    centerbias: str = "mit1003",
    max_long_side: int | None = None,
    predictor: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a sum-one DeepGaze fixation density and serializable metadata."""
    if predictor is None:
        from scout_core.deepgaze_msdb.inference import predict_saliency_path

        predictor = predict_saliency_path

    result = predictor(
        frame_path,
        pixels_per_degree=pixels_per_degree,
        allow_default_ppd=True,
        device=device,
        centerbias=centerbias,
        max_long_side=max_long_side,
        model=model,
        image_id=frame_path.stem,
    )
    density = np.asarray(result.density, dtype=np.float32)
    if density.shape != (capture_h, capture_w):
        from scout_core.deepgaze_msdb.preprocess import resize_density

        density = resize_density(density, capture_h, capture_w)
    if density.ndim != 2 or not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError("DeepGaze returned an invalid fixation density")
    density_sum = float(density.sum(dtype=np.float64))
    if not np.isclose(density_sum, 1.0, rtol=1e-4, atol=1e-6):
        raise ValueError(f"DeepGaze density must sum to 1, got {density_sum:.8f}")

    metadata = result.to_metadata()
    metadata["value_semantics"] = "fixation_probability_density"
    metadata["normalization"] = "sum_1"
    metadata["density_sum"] = density_sum
    metadata["dom_scoring_adapter"] = "max_normalize"
    return density, metadata


def adapt_heatmap_for_dom_scoring(
    heatmap: np.ndarray,
    manifest_entry: dict[str, Any] | None = None,
) -> np.ndarray:
    """Adapt saved maps to the legacy relative-intensity DOM scoring scale.

    DeepGaze artifacts remain probability densities on disk. Their pixel values
    are too small for the legacy six-decimal DOM score rounding, so consumers
    max-normalize a copy in memory when provenance identifies density semantics.
    """
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Heatmap contains non-finite values")

    semantics = (manifest_entry or {}).get("value_semantics")
    if semantics != "fixation_probability_density":
        return array

    peak = float(array.max(initial=0.0))
    if peak <= 0:
        return np.zeros_like(array, dtype=np.float32)
    return (array / peak).astype(np.float32, copy=False)


def load_heatmap_for_dom_scoring(
    path: Path,
    manifest_entry: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a heatmap and apply its provenance-declared scoring adapter."""
    return adapt_heatmap_for_dom_scoring(
        np.load(path).astype(np.float32),
        manifest_entry,
    )


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
