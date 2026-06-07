"""Element role labels and site-goal resolution for interpretive UX narratives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_SITE_GOAL = (
    "Help visitors quickly understand the product value and feel confident "
    "enough to take the primary conversion action."
)


def classify_role(
    dom_id: str,
    tag: str = "",
    role: str = "",
    text: str = "",
) -> str:
    """Heuristic display role — not the goal source, only a label hint for LLM/UI."""
    dom = (dom_id or "").lower()
    tag_u = (tag or "").upper()
    role_l = (role or "").lower()
    txt = (text or "").lower()

    if tag_u == "BUTTON" or role_l == "button" or "cta" in dom or "btn" in dom:
        return "CTA"
    if tag_u in ("H1", "H2") or "hero" in dom or "headline" in dom:
        return "headline"
    if tag_u == "NAV" or role_l == "navigation" or dom.startswith("nav"):
        return "nav"
    if tag_u in ("FORM",) or role_l == "form" or "form" in dom:
        return "form"
    if tag_u == "IMG" or role_l == "img" or "image" in dom or "img" in dom:
        return "image"
    if tag_u in ("HEADER", "SECTION", "MAIN") and len(txt) > 40:
        return "hero"
    if tag_u in ("P", "SPAN", "ARTICLE") and len(txt) > 20:
        return "body"
    if tag_u in ("A",):
        return "link"
    return "component"


CTA_TERMS = (
    "get started",
    "start",
    "try",
    "sign up",
    "signup",
    "book",
    "demo",
    "buy",
    "pricing",
    "contact",
    "learn more",
    "subscribe",
    "download",
)


def _bbox_area_score(bbox: list[int] | tuple[int, int, int, int] | None) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    area = max(int(bbox[2]), 0) * max(int(bbox[3]), 0)
    if area <= 0:
        return 0.0
    # A 220x56 CTA should score high without letting giant panels dominate.
    return min(1.0, area / 12_000.0) ** 0.35


def _fold_score(bbox: list[int] | tuple[int, int, int, int] | None, *, viewport_h: int = 1080) -> float:
    if not bbox or len(bbox) != 4:
        return 0.5
    y = max(float(bbox[1]), 0.0)
    center_y = y + max(float(bbox[3]), 0.0) / 2.0
    return max(0.0, min(1.0, 1.0 - (center_y / max(float(viewport_h) * 1.6, 1.0))))


def _style_contrast_score(element: dict[str, Any]) -> float:
    style = element.get("style") or {}
    if not isinstance(style, dict):
        return 0.5
    fg = str(style.get("color") or "")
    bg = str(style.get("background_color") or style.get("backgroundColor") or "")
    if fg and bg and fg != bg:
        return 0.72
    return 0.5


def clickability_score(
    element: dict[str, Any],
    *,
    viewport_h: int = 1080,
) -> float:
    """Heuristic 0-100 score for how likely an element is intended to be clicked.

    This is an affordance score, not observed behavior. Real click data can
    replace or blend with it later via the Clarity CSV adapter.
    """
    dom_id = str(element.get("dom_id") or "")
    tag = str(element.get("tag") or "")
    role = str(element.get("role") or "")
    text = str(element.get("text") or "")
    display_role = classify_role(dom_id, tag, role, text)
    tag_u = tag.upper()
    role_l = role.lower()
    text_l = text.lower()
    dom_l = dom_id.lower()

    role_score = {
        "CTA": 1.0,
        "link": 0.74,
        "form": 0.70,
        "nav": 0.50,
        "image": 0.24,
        "headline": 0.18,
        "hero": 0.16,
        "body": 0.08,
        "component": 0.18,
    }.get(display_role, 0.18)
    if tag_u in ("BUTTON", "A", "INPUT", "TEXTAREA", "SELECT") or role_l in ("button", "link", "textbox"):
        role_score = max(role_score, 0.78)
    if any(term in text_l or term in dom_l for term in CTA_TERMS):
        role_score = max(role_score, 0.86)

    size_score = _bbox_area_score(element.get("bbox"))
    fold = _fold_score(element.get("bbox"), viewport_h=viewport_h)
    contrast = _style_contrast_score(element)
    score = 100.0 * (
        0.48 * role_score
        + 0.20 * size_score
        + 0.18 * fold
        + 0.14 * contrast
    )
    return round(max(0.0, min(100.0, score)), 2)


def resolve_site_goal(
    *,
    override: str | None = None,
    script_path: Path | None = None,
) -> str:
    """Resolve the single paragraph site goal from CLI override or walkthrough YAML."""
    if override and override.strip():
        return override.strip()
    if script_path and script_path.is_file():
        with script_path.open(encoding="utf-8") as f:
            script = yaml.safe_load(f) or {}
        goal = script.get("site_goal")
        if goal and str(goal).strip():
            return str(goal).strip()
    return DEFAULT_SITE_GOAL


def collect_scored_elements(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten unique top_elements from section_report with section context."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for sec in bundle.get("section_report") or []:
        sid = sec.get("section_id", "")
        flags = sec.get("flags") or []
        emotion = sec.get("emotion") or {}
        for el in sec.get("top_elements") or []:
            dom_id = el.get("dom_id") or ""
            if not dom_id or dom_id in seen:
                continue
            seen.add(dom_id)
            density = (
                el.get("mean_attention_density")
                or el.get("attention_density")
                or el.get("score")
            )
            rows.append({
                "dom_id": dom_id,
                "section_id": sid,
                "tag": el.get("tag", ""),
                "role": el.get("role", ""),
                "text": el.get("text", ""),
                "display_role": classify_role(
                    dom_id,
                    el.get("tag", ""),
                    el.get("role", ""),
                    el.get("text", ""),
                ),
                "attention_density": density,
                "attention_score": el.get("attention_score"),
                "clickability": el.get("clickability"),
                "combined_score": el.get("combined_score"),
                "engagement_attributed": el.get("engagement_attributed"),
                "section_flags": flags,
                "dominant_emotion": emotion.get("dominant"),
                "emotion_mean_z": emotion.get("mean_z"),
            })
    return rows
