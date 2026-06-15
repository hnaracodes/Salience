"""Heuristic copy engagement signals derived from DOM snapshots.

Scores per section:
  clarity    — readability and conciseness (0–100)
  urgency    — CTA strength and imperative language (0–100)
  goal_fit   — keyword overlap with site_goal (0–100, 0 if no site_goal)
  copy_score — weighted mean of the three (0–100)

copy_source is always "heuristic" for this module; set to "llm" if overridden
by build_copy_signals_llm() (future Tier B).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from activation_store import SESSIONS_DIR

logger = logging.getLogger(__name__)

# Terms that signal strong calls to action
CTA_VERBS = frozenset([
    "get", "start", "try", "buy", "sign", "subscribe", "join",
    "download", "book", "request", "contact", "learn", "discover",
    "see", "watch", "explore", "schedule", "claim", "unlock",
])

URGENCY_TERMS = frozenset([
    "now", "today", "free", "limited", "exclusive", "instantly",
    "guarantee", "risk-free", "no credit card", "cancel anytime",
])

# Tags whose text is always collected regardless of viewport state
ALWAYS_INCLUDE_TAGS = frozenset(["H1", "H2", "H3", "P", "BUTTON", "A"])


# ---------------------------------------------------------------------------
# Text collection helpers
# ---------------------------------------------------------------------------


def _collect_section_text(elements: list[dict[str, Any]]) -> list[str]:
    """Collect visible text strings from a list of DOM snapshot elements."""
    texts: list[str] = []
    for el in elements:
        tag = (el.get("tag") or "").upper()
        in_viewport = el.get("in_viewport") or el.get("is_intersecting_viewport") or False
        text = (el.get("text") or "").strip()
        if not text:
            continue
        if in_viewport or tag in ALWAYS_INCLUDE_TAGS:
            texts.append(text)
    return texts


def _avg_sentence_length(combined: str) -> float:
    """Rough average sentence length in words."""
    sentences = re.split(r"[.!?]+", combined)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


# ---------------------------------------------------------------------------
# Per-section scorers
# ---------------------------------------------------------------------------


def _score_clarity(word_count: int, avg_sent_len: float) -> float:
    """Clarity score 0–100.

    Decays linearly from 100 at avg_sentence_len ≤ 15 to 40 at avg ≥ 40.
    Penalised for sparse (< 10 words) or wall-of-text (> 500 words) content.
    """
    if avg_sent_len <= 15:
        base = 100.0
    elif avg_sent_len >= 40:
        base = 40.0
    else:
        # Linear interpolation: 100 → 40 as avg goes 15 → 40
        base = 100.0 - (avg_sent_len - 15.0) / 25.0 * 60.0

    if word_count < 10:
        base *= 0.6  # sparse penalty
    elif word_count > 500:
        base *= 0.45  # heavy wall-of-text penalty

    return float(max(0.0, min(100.0, base)))


def _score_urgency(combined_lower: str) -> float:
    """Urgency score 0–100 based on CTA verbs and urgency term counts."""
    count = 0
    words = set(re.findall(r"\b\w+\b", combined_lower))
    for verb in CTA_VERBS:
        if verb in words or verb in combined_lower:
            count += 1
    for term in URGENCY_TERMS:
        if term in combined_lower:
            count += 1
    # 16 pts per matched signal ensures a 4-match phrase ("get started free today")
    # scores 64 > 60, which is a reliable "strong CTA" threshold.
    return float(min(100.0, count * 16.0))


def _score_goal_fit(combined_lower: str, site_goal: str | None) -> float:
    """Goal fit score 0–100; 0 when no site_goal provided."""
    if not site_goal:
        return 0.0
    goal_words = set(re.findall(r"\b\w+\b", site_goal.lower()))
    goal_words -= {"a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with"}
    if not goal_words:
        return 0.0
    text_words = set(re.findall(r"\b\w+\b", combined_lower))
    overlap = len(goal_words & text_words)
    fraction = overlap / len(goal_words)
    return float(min(100.0, fraction * 200.0))


def _copy_score(
    clarity: float,
    urgency: float,
    goal_fit: float,
    has_goal: bool,
) -> float:
    """Weighted composite copy score."""
    if has_goal:
        return 0.5 * clarity + 0.3 * urgency + 0.2 * goal_fit
    return 0.6 * clarity + 0.4 * urgency


def _flags(word_count: int, urgency: float) -> list[str]:
    f: list[str] = []
    if word_count < 10:
        f.append("sparse_content")
    if word_count > 400:
        f.append("wall_of_text")
    if urgency > 70:
        f.append("strong_cta")
    return f


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def _group_snapshots_by_section(
    dom_snapshots: list[dict[str, Any]],
    section_report: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return mapping {section_id: [elements]} for each section.

    If section_report provides t_indices, use those to map snapshots to sections.
    Otherwise fall back to grouping by URL path.
    """
    # Build t_idx → section_id lookup from section_report when t_indices are available.
    t_to_section: dict[int, str] = {}
    for sec in section_report or []:
        sid = str(sec.get("section_id") or "")
        for t in sec.get("t_indices") or sec.get("sample_t_indices") or []:
            t_to_section[int(t)] = sid

    groups: dict[str, list[dict[str, Any]]] = {}

    for snap in dom_snapshots:
        t_idx = snap.get("t_idx")
        if t_idx is not None and t_to_section:
            sid = t_to_section.get(int(t_idx), f"t{t_idx}")
        else:
            # Fall back: group by URL path
            url = snap.get("url") or snap.get("page_url") or ""
            sid = urlparse(url).path or "/"

        if sid not in groups:
            groups[sid] = []
        groups[sid].extend(snap.get("elements") or [])

    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_copy_signals(
    session_id: str,
    site_goal: str | None = None,
) -> list[dict[str, Any]]:
    """Compute heuristic copy engagement signals for each section of a session.

    Reads ``session_manifest.json`` and ``analysis_bundle.json`` from the
    session directory, scores each section's visible copy, writes results to
    ``copy_signals.json``, and returns the list.

    Parameters
    ----------
    session_id:
        Session directory name under SESSIONS_DIR.
    site_goal:
        Optional plain-language goal used for goal_fit scoring.

    Returns
    -------
    List of per-section signal dicts, or an empty list if the session has
    no DOM snapshots.
    """
    session_dir = SESSIONS_DIR / session_id
    manifest_path = session_dir / "session_manifest.json"
    bundle_path = session_dir / "analysis_bundle.json"

    # Graceful no-op if manifest is missing.
    if not manifest_path.is_file():
        logger.warning("copy_signals: no session_manifest.json for %s; returning []", session_id)
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dom_snapshots: list[dict[str, Any]] = manifest.get("dom_snapshots") or []
    if not dom_snapshots:
        logger.warning("copy_signals: no dom_snapshots in manifest for %s", session_id)
        return []

    section_report: list[dict[str, Any]] = []
    if bundle_path.is_file():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        section_report = bundle.get("section_report") or []

    groups = _group_snapshots_by_section(dom_snapshots, section_report)
    has_goal = bool(site_goal)
    results: list[dict[str, Any]] = []

    for section_id, elements in groups.items():
        texts = _collect_section_text(elements)
        combined = " ".join(texts)
        combined_lower = combined.lower()
        words = combined.split()
        word_count = len(words)

        avg_sent_len = _avg_sentence_length(combined)
        clarity = _score_clarity(word_count, avg_sent_len)
        urgency = _score_urgency(combined_lower)
        goal_fit = _score_goal_fit(combined_lower, site_goal)
        score = _copy_score(clarity, urgency, goal_fit, has_goal)

        results.append({
            "section_id": str(section_id),
            "clarity": round(clarity, 1),
            "urgency": round(urgency, 1),
            "goal_fit": round(goal_fit, 1),
            "copy_score": round(score, 1),
            "copy_source": "heuristic",
            "flags": _flags(word_count, urgency),
        })

    out_path = session_dir / "copy_signals.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("copy_signals: wrote %d section entries → %s", len(results), out_path)
    return results
