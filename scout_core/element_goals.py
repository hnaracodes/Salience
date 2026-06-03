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
                "section_flags": flags,
                "dominant_emotion": emotion.get("dominant"),
                "emotion_mean_z": emotion.get("mean_z"),
            })
    return rows
