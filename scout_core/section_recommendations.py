"""Rule-based UX improvement suggestions from section-level metrics."""

from __future__ import annotations

from typing import Any


def generate_section_recommendations(section: dict[str, Any]) -> list[str]:
    """Return marketing-readable suggestion strings for one section report row."""
    tips: list[str] = []
    engagement = section.get("engagement") or {}
    emotion = section.get("emotion") or {}
    mean_z = (emotion.get("mean_z") or {})
    flags = section.get("flags") or []
    top_elements = section.get("top_elements") or []
    dwell_sec = float(section.get("dwell_sec") or 0)
    pct_boring = engagement.get("pct_boring")
    pct_engaging = engagement.get("pct_engaging")
    eng_mean = engagement.get("mean")

    if pct_boring is not None and pct_boring > 0.4 and dwell_sec < 8:
        tips.append(
            "Users disengage quickly in this section — shorten copy or move the primary CTA higher."
        )
    elif pct_boring is not None and pct_boring > 0.35:
        tips.append(
            "Elevated boredom proxy — test a stronger visual hook or reduce passive text blocks."
        )

    if "high_arousal" in flags:
        fear_z = mean_z.get("fear", 0)
        anger_z = mean_z.get("anger", 0)
        if fear_z >= 1.0 or anger_z >= 1.0:
            tips.append(
                "Friction or anxiety proxy detected — simplify the flow, clarify pricing, and reduce visual clutter."
            )
        else:
            tips.append(
                "Arousal spike without clear valence — review unexpected motion, pop-ups, or dense forms in this block."
            )

    if "positive_valence" in flags and (pct_engaging is None or (pct_engaging or 0) < 0.15):
        tips.append(
            "Pleasant but low engagement — strengthen primary CTA contrast and hierarchy while keeping the tone."
        )

    if eng_mean is not None and eng_mean > 1.5 and dwell_sec >= 10:
        tips.append(
            "Strong engagement proxy with sustained dwell — consider reinforcing this layout pattern elsewhere on the site."
        )

    peak = emotion.get("peak") or {}
    if peak.get("z_score", 0) > 2.0 and (pct_engaging is None or (pct_engaging or 0) < 0.1):
        ch = peak.get("channel", "emotion")
        tips.append(
            f"High {ch} spike without matching engagement — test a clearer value proposition above the fold in this section."
        )

    if top_elements:
        top = top_elements[0]
        tag = str(top.get("tag", "")).upper()
        bbox = top.get("bbox") or []
        area = int(bbox[2]) * int(bbox[3]) if len(bbox) == 4 else 0
        if tag in ("A", "SPAN") and area < 2000:
            tips.append(
                "Model attention clusters on a low-salience text control — promote a hero CTA or larger interactive target."
            )

    if not tips:
        tips.append(
            "No strong negative signals in this section — monitor A/B tests on layout and copy for incremental gains."
        )

    return tips


def apply_recommendations(section_report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for sec in section_report:
        sec["recommendations"] = generate_section_recommendations(sec)
    return section_report
