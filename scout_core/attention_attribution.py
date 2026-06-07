"""Attribute TRIBE time-series signals to DOM elements via visual saliency.

TRIBE engagement/activation tracks are temporal scalars. This module combines
those scalars with element-level saliency/clickability so the UI can explain
"which visible elements likely carried the moment" without claiming eye-tracking.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scout_core.element_goals import clickability_score


def _as_float_list(values: list[Any]) -> list[float | None]:
    out: list[float | None] = []
    for value in values or []:
        try:
            out.append(float(value) if value is not None else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _normal_positive(value: float | None) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    # Map typical Z values to 0..1 while keeping negative/flat values low.
    return float(np.clip((value + 0.5) / 3.0, 0.0, 1.0))


def _section_brain_weight(bundle: dict[str, Any], t_indices: list[int]) -> tuple[float, float, float]:
    engagement = _as_float_list((bundle.get("engagement_track") or {}).get("scores") or [])
    activation = _as_float_list((bundle.get("activation_track") or {}).get("scores") or [])
    eng_vals = [_normal_positive(engagement[t]) for t in t_indices if t < len(engagement)]
    act_vals = [_normal_positive(activation[t]) for t in t_indices if t < len(activation)]
    eng_mean = float(np.mean(eng_vals)) if eng_vals else 0.0
    act_mean = float(np.mean(act_vals)) if act_vals else 0.0
    # Engagement is the primary "how much"; activation stabilizes low-variance sessions.
    blended = 0.72 * eng_mean + 0.28 * act_mean
    return blended, eng_mean, act_mean


def enrich_sections_with_attribution(
    sections: list[dict[str, Any]],
    bundle: dict[str, Any],
    *,
    attention_weight: float = 0.68,
    click_weight: float = 0.32,
    viewport_h: int = 1080,
) -> list[dict[str, Any]]:
    """Attach attention/clickability/combined scores to each top element."""
    for section in sections:
        t_indices = [int(t) for t in (section.get("t_indices") or [])]
        brain_weight, eng_weight, act_weight = _section_brain_weight(bundle, t_indices)
        elements = section.get("top_elements") or []
        if not elements:
            section["element_attribution"] = {
                "brain_weight": round(brain_weight, 4),
                "engagement_weight": round(eng_weight, 4),
                "activation_weight": round(act_weight, 4),
                "n_elements": 0,
            }
            continue

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
            "attention_weight": attention_weight,
            "click_weight": click_weight,
            "n_elements": len(enriched),
        }
    return sections
