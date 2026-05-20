"""DOM-to-heatmap spatial intersection for neural spike grounding.

Computes normalized attention density per DOM bounding box and selects the
element with the highest density as the grounding winner.

Input contract:
    heatmap  — float32 ndarray ``(capture_h, capture_w)`` at capture resolution.
    elements — list of dicts from ``session_manifest.json`` dom_snapshots[].elements[].

Output contract:
    grounding dict: ``{dom_id, tag, bbox, attention_density}``
    Matches the ``analysis_bundle.json`` events[].grounding schema v2.

Scroll handling:
    DOM bounding boxes in ``session_manifest.json`` are expressed in *document*
    coordinates (origin = top-left of full page, not viewport). Subtracting
    ``scrollX``/``scrollY`` from the manifest snapshot converts them to viewport
    (image) coordinates before sampling the heatmap.

Occlusion guardrail:
    Elements where ``is_intersecting_viewport`` is ``False`` are skipped.
    These are off-screen, scrolled out of view, or behind an overlaying modal.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Low-level bbox helpers
# ---------------------------------------------------------------------------

def _clip_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    img_h: int,
    img_w: int,
    scroll_y: int = 0,
    scroll_x: int = 0,
) -> tuple[int, int, int, int] | None:
    """Translate a document-space bbox to image pixel coords and clip to bounds.

    Args:
        x, y: Top-left corner in document (page) space.
        w, h: Width and height of the element.
        img_h, img_w: Heatmap dimensions.
        scroll_y: Vertical scroll offset from ``dom_snapshot.scrollY``.
        scroll_x: Horizontal scroll offset from ``dom_snapshot.scrollX``.

    Returns:
        ``(x1, y1, x2, y2)`` in image coordinates (exclusive end indices),
        or ``None`` if the element is fully outside the image bounds.
    """
    vx = x - scroll_x
    vy = y - scroll_y
    x1 = max(0, vx)
    y1 = max(0, vy)
    x2 = min(img_w, vx + w)
    y2 = min(img_h, vy + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Core density computation
# ---------------------------------------------------------------------------

def compute_attention_density(
    heatmap: np.ndarray,
    bbox: list[int],
    scroll_y: int = 0,
    scroll_x: int = 0,
) -> float | None:
    """Return mean heatmap value inside a bbox given in document coordinates.

    ``attention_density = sum(H[y, x] for (x,y) in box) / (W * H)``

    Args:
        heatmap:  2D float array ``(img_h, img_w)`` at capture resolution.
        bbox:     ``[X, Y, W, H]`` in document / capture coordinates.
        scroll_y: Vertical scroll offset (px) from ``dom_snapshot.scrollY``.
        scroll_x: Horizontal scroll offset (px) from ``dom_snapshot.scrollX``.

    Returns:
        ``attention_density`` ∈ [0, ∞), or ``None`` if fully off-screen / empty.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    img_h, img_w = heatmap.shape
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    clipped = _clip_bbox(x, y, w, h, img_h, img_w, scroll_y, scroll_x)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    box_area = (x2 - x1) * (y2 - y1)
    if box_area == 0:
        return None
    return float(heatmap[y1:y2, x1:x2].sum() / box_area)


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------

def select_winner(
    heatmap: np.ndarray,
    elements: list[dict[str, Any]],
    scroll_y: int = 0,
    scroll_x: int = 0,
) -> dict[str, Any] | None:
    """Select the DOM element with the highest normalized attention density.

    Iterates all eligible candidates (``is_intersecting_viewport=True`` and
    a valid ``[X, Y, W, H]`` ``bbox``), scoring each with
    :func:`compute_attention_density`, and returns the ``argmax``.

    Args:
        heatmap:  2D float array ``(img_h, img_w)`` at capture resolution.
        elements: Element dicts from ``dom_snapshots[].elements[]``. Required
                  keys: ``bbox``. Optional: ``dom_id``, ``tag``, ``z_index``,
                  ``is_intersecting_viewport``.
        scroll_y: Vertical scroll offset from the matching dom_snapshot.
        scroll_x: Horizontal scroll offset from the matching dom_snapshot.

    Returns:
        Dict ``{dom_id, tag, bbox, attention_density}`` for the winner,
        or ``None`` if no eligible candidate exists.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    best_density = -1.0
    winner: dict[str, Any] | None = None

    for el in elements:
        if not el.get("is_intersecting_viewport", True):
            continue

        bbox = el.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue

        density = compute_attention_density(heatmap, bbox, scroll_y, scroll_x)
        if density is None:
            continue

        if density > best_density:
            best_density = density
            winner = {
                "dom_id": el.get("dom_id", ""),
                "tag": el.get("tag", ""),
                "bbox": [int(b) for b in bbox],
                "attention_density": round(best_density, 6),
            }

    return winner


# ---------------------------------------------------------------------------
# Snapshot-level convenience helpers
# ---------------------------------------------------------------------------

def ground_snapshot(
    heatmap: np.ndarray,
    dom_snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    """Ground one DOM snapshot against a heatmap.

    Thin wrapper over :func:`select_winner` for a single ``dom_snapshots[]``
    entry — applies ``scrollY``/``scrollX`` from the snapshot automatically.

    Args:
        heatmap:      2D float array ``(img_h, img_w)``.
        dom_snapshot: One entry from ``session_manifest.json`` dom_snapshots[].
                      Expected keys: ``elements``, optional ``scrollY``, ``scrollX``.

    Returns:
        Winner dict or ``None``.
    """
    elements = dom_snapshot.get("elements", [])
    scroll_y = int(dom_snapshot.get("scrollY", 0))
    scroll_x = int(dom_snapshot.get("scrollX", 0))
    return select_winner(heatmap, elements, scroll_y=scroll_y, scroll_x=scroll_x)


def find_nearest_snapshot(
    manifest: dict[str, Any],
    t_spike: int,
) -> dict[str, Any] | None:
    """Find the ``dom_snapshot`` whose ``t_idx`` is closest to ``t_spike``.

    Args:
        manifest: Parsed ``session_manifest.json`` dict.
        t_spike:  Row index into ``preds[T, 20484]`` (TRIBE TR index).

    Returns:
        The nearest snapshot dict, or ``None`` if the manifest has none.
    """
    snapshots = manifest.get("dom_snapshots", [])
    if not snapshots:
        return None
    return min(snapshots, key=lambda s: abs(int(s.get("t_idx", 0)) - t_spike))
