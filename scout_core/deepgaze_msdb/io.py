"""Serialize DeepGaze MSDB predictions to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scout_core.deepgaze_msdb.inference import SaliencyResult
from scout_core.deepgaze_msdb.preprocess import load_rgb_image
from scout_core.deepgaze_msdb.visualize import (
    density_to_heatmap_rgb,
    overlay_heatmap,
    save_rgb_png,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def write_prediction_artifacts(
    result: SaliencyResult,
    out_dir: Path | str,
    *,
    stem: str,
    source_image: Path | str | np.ndarray | None = None,
    write_overlay: bool = True,
    overlay_alpha: float = 0.45,
) -> dict[str, str]:
    """Write density.npy, heatmap.png, optional overlay.png, and metadata.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density_path = out_dir / f"{stem}_density.npy"
    log_path = out_dir / f"{stem}_log_density.npy"
    heatmap_path = out_dir / f"{stem}_heatmap.png"
    meta_path = out_dir / f"{stem}_meta.json"

    np.save(density_path, result.density)
    np.save(log_path, result.log_density)
    save_rgb_png(heatmap_path, density_to_heatmap_rgb(result.density))

    paths: dict[str, str] = {
        "density": str(density_path.resolve()),
        "log_density": str(log_path.resolve()),
        "heatmap": str(heatmap_path.resolve()),
        "meta": str(meta_path.resolve()),
    }

    if write_overlay:
        if source_image is None:
            source_image = result.provenance.get("source_path")
        if source_image is None:
            raise ValueError("source_image required to write overlay")
        rgb = load_rgb_image(source_image)
        overlay = overlay_heatmap(rgb, result.density, alpha=overlay_alpha)
        overlay_path = out_dir / f"{stem}_overlay.png"
        save_rgb_png(overlay_path, overlay)
        paths["overlay"] = str(overlay_path.resolve())

    meta = result.to_metadata()
    meta["artifacts"] = paths
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return paths
