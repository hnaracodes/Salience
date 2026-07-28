"""Attribute TRIBE time-series signals to DOM elements via visual saliency.

TRIBE engagement/activation tracks are temporal scalars. This module combines
those scalars with element-level saliency/clickability so the UI can explain
"which visible elements likely carried the moment" without claiming eye-tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from scout_core.clarity_adapter import click_rows_by_selector, parse_clarity_click_csv
from scout_core.dom_intersect import (
    filter_elements_to_section,
    find_nearest_snapshot,
    score_all_elements,
)
from scout_core.element_goals import clickability_score
from scout_core.heatmap_extract import (
    load_heatmap_for_dom_scoring,
    read_heatmaps_manifest,
)
from scout_core.session_align import is_timestep_in_manifest


def _as_float_list(values: list[Any]) -> list[float | None]:
    out: list[float | None] = []
    for value in values or []:
        try:
            out.append(float(value) if value is not None else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _normal_positive(value: float | None, *, scale: float = 3.0) -> float:
    """Map engagement/activation Z to [0,1] for attribution blending (|Z|/scale)."""
    if value is None or not np.isfinite(value):
        return 0.0
    return float(np.clip(abs(value) / scale, 0.0, 1.0))


def brain_weight_at_t(
    bundle: dict[str, Any],
    t: int,
    *,
    engagement_blend: float = 0.72,
) -> tuple[float, float, float]:
    """Return blended brain weight and per-track normalized weights at TR ``t``."""
    from scout_core.neural_moments import neural_moment_strength

    engagement = _as_float_list((bundle.get("engagement_track") or {}).get("scores") or [])
    activation = _as_float_list((bundle.get("activation_track") or {}).get("scores") or [])
    eng = _normal_positive(engagement[t] if t < len(engagement) else None)
    act = _normal_positive(activation[t] if t < len(activation) else None)
    moments = bundle.get("neural_moments_by_t") or {}
    moment = moments.get(str(t)) or moments.get(t)
    moment_w = neural_moment_strength(moment) / 3.0 if moment else 0.0
    moment_w = float(np.clip(moment_w, 0.0, 1.0))
    blended = engagement_blend * eng + (1.0 - engagement_blend) * act
    if moment_w > 0:
        blended = float(np.clip(0.65 * blended + 0.35 * moment_w, 0.0, 1.0))
    return blended, eng, act


def _section_brain_weight(
    bundle: dict[str, Any],
    t_indices: list[int],
    *,
    engagement_blend: float = 0.72,
) -> tuple[float, float, float]:
    engagement = _as_float_list((bundle.get("engagement_track") or {}).get("scores") or [])
    activation = _as_float_list((bundle.get("activation_track") or {}).get("scores") or [])
    eng_vals = [_normal_positive(engagement[t]) for t in t_indices if t < len(engagement)]
    act_vals = [_normal_positive(activation[t]) for t in t_indices if t < len(activation)]
    eng_mean = float(np.mean(eng_vals)) if eng_vals else 0.0
    act_mean = float(np.mean(act_vals)) if act_vals else 0.0
    blended = engagement_blend * eng_mean + (1.0 - engagement_blend) * act_mean
    return blended, eng_mean, act_mean


def load_or_compute_heatmap(
    session_dir: Path,
    t: int,
    manifest: dict[str, Any],
    *,
    heatmaps_subdir: str = "heatmaps",
    frames_subdir: str = "frames",
) -> np.ndarray | None:
    """Load cached heatmap or compute CPU visual saliency for timestep ``t``."""
    heatmap_path = session_dir / heatmaps_subdir / f"t_{t}.npy"
    if heatmap_path.is_file():
        hm_manifest = read_heatmaps_manifest(session_dir / heatmaps_subdir)
        return load_heatmap_for_dom_scoring(heatmap_path, hm_manifest.get(t))

    frame_path = session_dir / frames_subdir / f"t_{t}.jpg"
    if not frame_path.is_file():
        return None

    snapshot = find_nearest_snapshot(manifest, t)
    if snapshot is None:
        return None

    capture = manifest.get("capture") or {}
    from scout_core.visual_saliency import build_saliency_heatmap

    heatmap, _ = build_saliency_heatmap(
        frame_path,
        snapshot,
        capture_h=int(capture.get("height", 1080)),
        capture_w=int(capture.get("width", 1920)),
    )
    return heatmap


def _visibility_weight(element: dict[str, Any]) -> float:
    ratio = element.get("visibility_ratio")
    if ratio is None:
        return 1.0
    try:
        return float(np.clip(float(ratio), 0.0, 1.0))
    except (TypeError, ValueError):
        return 1.0


def aggregate_per_tr_element_scores(
    section: dict[str, Any],
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    session_dir: Path,
    *,
    min_element_area: int = 400,
    engagement_blend: float = 0.72,
    max_tr_per_section: int = 30,
) -> list[dict[str, Any]]:
    """Sum saliency × visibility × brain_weight(t) across section dwell TRs."""
    t_indices = [int(t) for t in (section.get("t_indices") or [])]
    if max_tr_per_section > 0 and len(t_indices) > max_tr_per_section:
        step = max(1, len(t_indices) // max_tr_per_section)
        t_indices = t_indices[::step][:max_tr_per_section]

    root_bbox = section.get("section_root_bbox")
    accum: dict[str, dict[str, Any]] = {}

    for t in t_indices:
        if not is_timestep_in_manifest(t, manifest, tolerance=1):
            continue
        heatmap = load_or_compute_heatmap(session_dir, t, manifest)
        if heatmap is None:
            continue
        snapshot = find_nearest_snapshot(manifest, t)
        if snapshot is None:
            continue
        elements = filter_elements_to_section(snapshot.get("elements") or [], root_bbox)
        scroll_y = int(snapshot.get("scrollY", 0))
        scroll_x = int(snapshot.get("scrollX", 0))
        brain_w, _, _ = brain_weight_at_t(bundle, t, engagement_blend=engagement_blend)
        if brain_w <= 0:
            continue

        scored = score_all_elements(
            heatmap, elements, scroll_y, scroll_x, min_area=min_element_area,
        )
        for row in scored:
            dom_id = row.get("dom_id") or ""
            key = dom_id or str(row.get("bbox"))
            vis = _visibility_weight(row)
            contrib = float(row.get("attention_density") or 0.0) * vis * brain_w
            if contrib <= 0:
                continue
            if key not in accum:
                accum[key] = {
                    "dom_id": dom_id,
                    "tag": row.get("tag", ""),
                    "role": row.get("role", ""),
                    "text": row.get("text", ""),
                    "bbox": row.get("bbox", []),
                    "contribution": 0.0,
                    "n_tr": 0,
                    "engagement_attributed": 0.0,
                    "activation_attributed": 0.0,
                }
            _, eng_w, act_w = brain_weight_at_t(bundle, t, engagement_blend=engagement_blend)
            dens = float(row.get("attention_density") or 0.0)
            accum[key]["contribution"] += contrib
            accum[key]["n_tr"] += 1
            accum[key]["engagement_attributed"] += eng_w * dens * vis
            accum[key]["activation_attributed"] += act_w * dens * vis
            if row.get("text") and not accum[key].get("text"):
                accum[key]["text"] = row.get("text", "")

    rolled: list[dict[str, Any]] = []
    for entry in accum.values():
        n_tr = max(int(entry["n_tr"]), 1)
        mean_contrib = float(entry["contribution"]) / n_tr
        mean_eng = float(entry["engagement_attributed"]) / n_tr
        mean_act = float(entry["activation_attributed"]) / n_tr
        rolled.append({
            "dom_id": entry["dom_id"],
            "tag": entry["tag"],
            "role": entry.get("role", ""),
            "text": entry.get("text", ""),
            "bbox": entry["bbox"],
            "mean_attention_density": round(mean_contrib, 6),
            "summed_attention_density": round(float(entry["contribution"]), 6),
            "n_samples": n_tr,
            "engagement_attributed": round(mean_eng, 4),
            "activation_attributed": round(mean_act, 4),
            "attribution_confidence": round(min(1.0, n_tr / 10.0), 3),
            "heatmap_source": "per_tr_sum",
        })
    rolled.sort(key=lambda x: x["mean_attention_density"], reverse=True)
    return rolled


def _enrich_section_blend(
    section: dict[str, Any],
    bundle: dict[str, Any],
    *,
    attention_weight: float,
    click_weight: float,
    viewport_h: int,
    engagement_blend: float = 0.72,
) -> None:
    """Legacy section-average brain weight × rolled-up saliency."""
    t_indices = [int(t) for t in (section.get("t_indices") or [])]
    brain_weight, eng_weight, act_weight = _section_brain_weight(
        bundle, t_indices, engagement_blend=engagement_blend,
    )
    elements = section.get("top_elements") or []
    if not elements:
        section["element_attribution"] = {
            "brain_weight": round(brain_weight, 4),
            "engagement_weight": round(eng_weight, 4),
            "activation_weight": round(act_weight, 4),
            "mode": "section_blend",
            "n_elements": 0,
        }
        return

    densities = [
        float(el.get("mean_attention_density") or el.get("attention_density") or 0.0)
        for el in elements
    ]
    max_density = max(max(densities), 1e-6)
    enriched: list[dict[str, Any]] = []
    for el, density in zip(elements, densities):
        saliency_norm = float(np.clip(density / max_density, 0.0, 1.0))
        attention_score = 100.0 * saliency_norm * (0.35 + 0.65 * brain_weight)
        click_score = clickability_score(el, viewport_h=viewport_h)
        combined = attention_weight * attention_score + click_weight * click_score
        flags: list[str] = []
        if attention_score >= 70 and click_score < 40:
            flags.append("high_saliency_low_clickability")
        if click_score >= 70 and attention_score < 45:
            flags.append("cta_low_saliency")
        new_el = dict(el)
        new_el.update({
            "attention_score": round(float(np.clip(attention_score, 0.0, 100.0)), 2),
            "clickability": click_score,
            "combined_score": round(float(np.clip(combined, 0.0, 100.0)), 2),
            "engagement_attributed": round(float(eng_weight * saliency_norm), 4),
            "activation_attributed": round(float(act_weight * saliency_norm), 4),
            "attribution_flags": flags,
        })
        enriched.append(new_el)
    enriched.sort(key=lambda row: row.get("combined_score", 0.0), reverse=True)
    section["top_elements"] = enriched
    section["element_attribution"] = {
        "brain_weight": round(brain_weight, 4),
        "engagement_weight": round(eng_weight, 4),
        "activation_weight": round(act_weight, 4),
        "mode": "section_blend",
        "attention_weight": attention_weight,
        "click_weight": click_weight,
        "n_elements": len(enriched),
    }


def _enrich_per_tr_sum(
    section: dict[str, Any],
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    session_dir: Path,
    *,
    attention_weight: float,
    click_weight: float,
    viewport_h: int,
    min_element_area: int,
    engagement_blend: float,
    max_tr_per_section: int,
    top_k: int,
) -> None:
    """Per-TR contribution sum; replaces top_elements when heatmap/frame data exists."""
    per_tr = aggregate_per_tr_element_scores(
        section,
        bundle,
        manifest,
        session_dir,
        min_element_area=min_element_area,
        engagement_blend=engagement_blend,
        max_tr_per_section=max_tr_per_section,
    )
    if per_tr:
        section["top_elements"] = per_tr[:top_k]

    elements = section.get("top_elements") or []
    if not elements:
        section["element_attribution"] = {
            "mode": "per_tr_sum",
            "n_elements": 0,
        }
        return

    densities = [float(el.get("mean_attention_density") or 0.0) for el in elements]
    max_density = max(max(densities), 1e-6)
    enriched: list[dict[str, Any]] = []
    for el, density in zip(elements, densities):
        saliency_norm = float(np.clip(density / max_density, 0.0, 1.0))
        attention_score = 100.0 * saliency_norm
        click_score = clickability_score(el, viewport_h=viewport_h)
        combined = attention_weight * attention_score + click_weight * click_score
        flags: list[str] = []
        if attention_score >= 70 and click_score < 40:
            flags.append("high_saliency_low_clickability")
        if click_score >= 70 and attention_score < 45:
            flags.append("cta_low_saliency")
        new_el = dict(el)
        new_el.update({
            "attention_score": round(float(np.clip(attention_score, 0.0, 100.0)), 2),
            "clickability": click_score,
            "combined_score": round(float(np.clip(combined, 0.0, 100.0)), 2),
            "attribution_flags": flags,
        })
        enriched.append(new_el)
    enriched.sort(key=lambda row: row.get("combined_score", 0.0), reverse=True)
    section["top_elements"] = enriched
    section["element_attribution"] = {
        "mode": "per_tr_sum",
        "attention_weight": attention_weight,
        "click_weight": click_weight,
        "engagement_activation_blend": engagement_blend,
        "max_tr_per_section": max_tr_per_section,
        "n_elements": len(enriched),
    }


def _normalize_selector(value: str) -> str:
    return (value or "").strip().lower()


def _match_clarity_row(
    element: dict[str, Any],
    by_selector: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    dom_id = _normalize_selector(str(element.get("dom_id") or ""))
    if dom_id and dom_id in by_selector:
        return by_selector[dom_id]
    for sel, row in by_selector.items():
        if sel and dom_id and (sel in dom_id or dom_id in sel):
            return row
    text = (element.get("text") or "").strip().lower()
    if text:
        for row in by_selector.values():
            row_text = (row.get("text") or "").strip().lower()
            if row_text and (row_text in text or text in row_text):
                return row
    return None


def _clarity_clickability_norm(clicks: int, max_clicks: int) -> float:
    if max_clicks <= 0:
        return 0.0
    return float(np.clip(100.0 * clicks / max_clicks, 0.0, 100.0))


def enrich_sections_with_clarity(
    sections: list[dict[str, Any]],
    clarity_rows: list[dict[str, Any]],
    *,
    blend_weight: float = 0.5,
    attention_weight: float = 0.68,
    click_weight: float = 0.32,
) -> dict[str, Any]:
    """Merge offline Clarity CSV clicks into element clickability and combined_score."""
    by_selector = {
        _normalize_selector(str(k)): v
        for k, v in click_rows_by_selector(clarity_rows).items()
    }
    max_clicks = max((int(r.get("clicks") or 0) for r in clarity_rows), default=0)
    n_matched = 0

    for section in sections:
        for el in section.get("top_elements") or []:
            row = _match_clarity_row(el, by_selector)
            if row is None:
                el["clarity_matched"] = False
                continue
            n_matched += 1
            heuristic = float(el.get("clickability") or 0.0)
            clarity_norm = _clarity_clickability_norm(int(row.get("clicks") or 0), max_clicks)
            if row.get("click_rate") is not None:
                clarity_norm = max(clarity_norm, float(row["click_rate"]) * 100.0)
            effective = (1.0 - blend_weight) * heuristic + blend_weight * clarity_norm
            el["clarity_matched"] = True
            el["clarity_clicks"] = int(row.get("clicks") or 0)
            el["clarity_click_rate"] = row.get("click_rate")
            el["clickability"] = round(effective, 2)
            attention = float(el.get("attention_score") or 0.0)
            el["combined_score"] = round(
                float(np.clip(attention_weight * attention + click_weight * effective, 0.0, 100.0)),
                2,
            )

        section["top_elements"] = sorted(
            section.get("top_elements") or [],
            key=lambda r: r.get("combined_score", 0.0),
            reverse=True,
        )

    return {
        "enabled": True,
        "n_rows": len(clarity_rows),
        "n_matched": n_matched,
        "blend_weight": blend_weight,
        "source": "microsoft_clarity_csv",
    }


def resolve_clarity_csv(session_dir: Path, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        p = explicit
        return p if p.is_file() else None
    default = session_dir / "clarity_clicks.csv"
    return default if default.is_file() else None


def load_clarity_rows(path: Path) -> list[dict[str, Any]]:
    return parse_clarity_click_csv(path)


def clarity_host_warning(manifest: dict[str, Any], clarity_rows: list[dict[str, Any]]) -> str | None:
    session_host = urlparse(str(manifest.get("initial_url") or "")).netloc.lower()
    if not session_host:
        return None
    for row in clarity_rows:
        url_host = urlparse(str(row.get("url") or "")).netloc.lower()
        if url_host and url_host != session_host:
            return f"Clarity CSV host {url_host!r} differs from session {session_host!r}"
    return None


def enrich_sections_with_attribution(
    sections: list[dict[str, Any]],
    bundle: dict[str, Any],
    *,
    attention_weight: float = 0.68,
    click_weight: float = 0.32,
    viewport_h: int = 1080,
    mode: str = "per_tr_sum",
    session_dir: Path | None = None,
    manifest: dict[str, Any] | None = None,
    min_element_area: int = 400,
    engagement_blend: float = 0.72,
    max_tr_per_section: int = 30,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Attach attention/clickability/combined scores to each top element."""
    use_per_tr = (
        mode == "per_tr_sum"
        and session_dir is not None
        and manifest is not None
        and manifest.get("dom_snapshots")
    )

    for section in sections:
        if use_per_tr:
            _enrich_per_tr_sum(
                section,
                bundle,
                manifest,
                session_dir,
                attention_weight=attention_weight,
                click_weight=click_weight,
                viewport_h=viewport_h,
                min_element_area=min_element_area,
                engagement_blend=engagement_blend,
                max_tr_per_section=max_tr_per_section,
                top_k=top_k,
            )
        else:
            _enrich_section_blend(
                section,
                bundle,
                attention_weight=attention_weight,
                click_weight=click_weight,
                viewport_h=viewport_h,
                engagement_blend=engagement_blend,
            )
    return sections
