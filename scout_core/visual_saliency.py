"""Deterministic website saliency maps from DOM geometry and screenshots.

This module is intentionally heuristic. It is not eye tracking and it is not a
TRIBE-derived spatial map. It provides an always-available "where" signal for
website analysis by scoring visible DOM elements with layout, contrast, color,
and affordance cues, then rasterizing those scores to the capture resolution.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "visual_saliency.yaml"

DEFAULT_WEIGHTS: dict[str, float] = {
    "area": 0.24,
    "position": 0.22,
    "contrast": 0.22,
    "color": 0.14,
    "element_type": 0.12,
    "isolation": 0.06,
}

TYPE_WEIGHTS: dict[str, float] = {
    "BUTTON": 1.0,
    "A": 0.86,
    "IMG": 0.84,
    "VIDEO": 0.84,
    "H1": 0.92,
    "H2": 0.80,
    "H3": 0.68,
    "INPUT": 0.74,
    "TEXTAREA": 0.70,
    "SELECT": 0.70,
    "FORM": 0.62,
    "NAV": 0.42,
    "P": 0.36,
    "LI": 0.34,
    "SECTION": 0.28,
    "MAIN": 0.22,
}


def load_saliency_config(path: Path | None = None) -> dict[str, Any]:
    """Load saliency weights with safe defaults."""
    cfg_path = path or DEFAULT_CONFIG
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    weights = dict(DEFAULT_WEIGHTS)
    weights.update({k: float(v) for k, v in (cfg.get("weights") or {}).items()})
    type_weights = dict(TYPE_WEIGHTS)
    type_weights.update({k.upper(): float(v) for k, v in (cfg.get("type_weights") or {}).items()})
    return {
        **cfg,
        "weights": weights,
        "type_weights": type_weights,
        "min_element_area": int(cfg.get("min_element_area", 64)),
        "edge_margin_px": int(cfg.get("edge_margin_px", 24)),
        "opacity": float(cfg.get("opacity", 0.86)),
    }


def _load_image_rgb(frame_path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - requirements include pillow via deps in normal runs.
        raise RuntimeError("Pillow is required for visual saliency extraction") from exc
    with Image.open(frame_path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _clip_viewport_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    *,
    img_h: int,
    img_w: int,
    scroll_y: int = 0,
    scroll_x: int = 0,
) -> tuple[int, int, int, int] | None:
    x, y, w, h = [int(v) for v in bbox]
    x1 = max(0, x - scroll_x)
    y1 = max(0, y - scroll_y)
    x2 = min(img_w, x - scroll_x + max(0, w))
    y2 = min(img_h, y - scroll_y + max(0, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _box_stats(image: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0, 0.0, 0.0
    lum = _luminance(patch)
    saturation = patch.max(axis=2) - patch.min(axis=2)
    warm = np.maximum(patch[..., 0] - patch[..., 2], 0.0)
    return float(lum.mean()), float(saturation.mean()), float(warm.mean())


def _local_background_luminance(image: np.ndarray, box: tuple[int, int, int, int], margin: int) -> float:
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    bx1 = max(0, x1 - margin)
    by1 = max(0, y1 - margin)
    bx2 = min(w, x2 + margin)
    by2 = min(h, y2 + margin)
    outer = image[by1:by2, bx1:bx2]
    if outer.size == 0:
        return 0.5
    mask = np.ones(outer.shape[:2], dtype=bool)
    ix1, iy1 = x1 - bx1, y1 - by1
    ix2, iy2 = x2 - bx1, y2 - by1
    mask[iy1:iy2, ix1:ix2] = False
    bg = outer[mask]
    if bg.size == 0:
        return float(_luminance(outer).mean())
    return float(_luminance(bg.reshape(-1, 3)).mean())


def _area_score(box: tuple[int, int, int, int], img_h: int, img_w: int) -> float:
    x1, y1, x2, y2 = box
    area_frac = ((x2 - x1) * (y2 - y1)) / max(img_h * img_w, 1)
    return float(min(1.0, math.log1p(area_frac * 90.0) / math.log1p(90.0)))


def _position_score(box: tuple[int, int, int, int], img_h: int, img_w: int) -> float:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / max(img_w, 1)
    cy = (y1 + y2) / 2.0 / max(img_h, 1)
    center_bias = 1.0 - min(1.0, math.sqrt((cx - 0.5) ** 2 + (cy - 0.42) ** 2) / 0.72)
    fold_bias = 1.0 - min(1.0, cy * 0.72)
    return float(0.62 * center_bias + 0.38 * fold_bias)


def _type_score(element: dict[str, Any], type_weights: dict[str, float]) -> float:
    tag = str(element.get("tag") or "").upper()
    role = str(element.get("role") or "").lower()
    dom_id = str(element.get("dom_id") or "").lower()
    text = str(element.get("text") or "").lower()
    base = type_weights.get(tag, 0.30)
    if role == "button" or "button" in dom_id or "btn" in dom_id or "cta" in dom_id:
        base = max(base, 1.0)
    if any(token in text for token in ("get started", "try", "buy", "sign up", "start", "book", "demo")):
        base = max(base, 0.92)
    return float(min(1.0, base))


def score_element_saliency(
    image: np.ndarray,
    element: dict[str, Any],
    *,
    scroll_y: int = 0,
    scroll_x: int = 0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a saliency row for one DOM element, or ``None`` if not visible."""
    cfg = config or load_saliency_config()
    img_h, img_w = image.shape[:2]
    bbox = element.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    if not element.get("is_intersecting_viewport", True):
        return None
    box = _clip_viewport_bbox(bbox, img_h=img_h, img_w=img_w, scroll_y=scroll_y, scroll_x=scroll_x)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if (x2 - x1) * (y2 - y1) < int(cfg.get("min_element_area", 64)):
        return None

    weights = cfg["weights"]
    elem_lum, elem_sat, elem_warm = _box_stats(image, box)
    bg_lum = _local_background_luminance(image, box, int(cfg.get("edge_margin_px", 24)))
    contrast = min(1.0, abs(elem_lum - bg_lum) * 2.2)
    color = min(1.0, 0.68 * elem_sat + 0.32 * elem_warm)
    isolation = min(1.0, contrast * 0.7 + _position_score(box, img_h, img_w) * 0.3)

    parts = {
        "area": _area_score(box, img_h, img_w),
        "position": _position_score(box, img_h, img_w),
        "contrast": contrast,
        "color": color,
        "element_type": _type_score(element, cfg["type_weights"]),
        "isolation": isolation,
    }
    total_weight = max(sum(float(v) for v in weights.values()), 1e-6)
    score = sum(parts[k] * float(weights.get(k, 0.0)) for k in parts) / total_weight
    return {
        "dom_id": element.get("dom_id", ""),
        "tag": element.get("tag", ""),
        "role": element.get("role", ""),
        "text": element.get("text", ""),
        "bbox": [int(v) for v in bbox],
        "viewport_bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
        "saliency_score": round(float(np.clip(score, 0.0, 1.0)), 6),
        "saliency_parts": {k: round(float(v), 6) for k, v in parts.items()},
    }


def score_snapshot_elements(
    frame_path: Path,
    snapshot: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Score all visible elements in one snapshot sorted by saliency descending."""
    image = _load_image_rgb(frame_path)
    cfg = load_saliency_config(config_path)
    rows = []
    for element in snapshot.get("elements") or []:
        row = score_element_saliency(
            image,
            element,
            scroll_y=int(snapshot.get("scrollY", 0)),
            scroll_x=int(snapshot.get("scrollX", 0)),
            config=cfg,
        )
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: r["saliency_score"], reverse=True)
    return rows


def build_saliency_heatmap(
    frame_path: Path,
    snapshot: dict[str, Any],
    *,
    capture_h: int | None = None,
    capture_w: int | None = None,
    config_path: Path | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Rasterize per-element saliency into a heatmap matching capture resolution."""
    image = _load_image_rgb(frame_path)
    img_h, img_w = image.shape[:2]
    out_h = int(capture_h or img_h)
    out_w = int(capture_w or img_w)
    cfg = load_saliency_config(config_path)
    heatmap = np.zeros((out_h, out_w), dtype=np.float32)
    scored: list[dict[str, Any]] = []
    scale_x = out_w / max(img_w, 1)
    scale_y = out_h / max(img_h, 1)

    for element in snapshot.get("elements") or []:
        row = score_element_saliency(
            image,
            element,
            scroll_y=int(snapshot.get("scrollY", 0)),
            scroll_x=int(snapshot.get("scrollX", 0)),
            config=cfg,
        )
        if row is None:
            continue
        vx, vy, vw, vh = row["viewport_bbox"]
        x1 = max(0, min(out_w, int(round(vx * scale_x))))
        y1 = max(0, min(out_h, int(round(vy * scale_y))))
        x2 = max(0, min(out_w, int(round((vx + vw) * scale_x))))
        y2 = max(0, min(out_h, int(round((vy + vh) * scale_y))))
        if x2 <= x1 or y2 <= y1:
            continue
        value = float(row["saliency_score"])
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], value)
        scored.append(row)

    if np.max(heatmap) > 0:
        heatmap = heatmap / float(np.max(heatmap))
    return heatmap.astype(np.float32), sorted(scored, key=lambda r: r["saliency_score"], reverse=True)
