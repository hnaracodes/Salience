"""Build LLM prompts from structured analysis_bundle fields (no raw DOM/HTML)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scout_core.element_goals import classify_role, collect_scored_elements
from scout_core.schemas import (
    ElementInsight,
    FrameInsight,
    MarketingNarrative,
    MarketingNarrativeSection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "configs" / "llm_narrative.yaml"


def load_llm_config(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_CFG
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _engagement_at_t(bundle: dict[str, Any], t: int) -> float | None:
    scores = (bundle.get("engagement_track") or {}).get("scores") or []
    if t < 0 or t >= len(scores):
        return None
    v = scores[t]
    return None if v is None else float(v)


def _infer_reaction(
    engagement_z: float | None,
    flags: list[str],
    *,
    density: float | None = None,
) -> str:
    flag_set = {str(f).lower() for f in flags}
    if "high_arousal" in flag_set or "friction" in flag_set:
        return "friction"
    if engagement_z is not None:
        if engagement_z >= 1.0:
            return "engaging"
        if engagement_z <= -1.0:
            return "boring"
    if density is not None and density >= 0.003:
        return "engaging"
    return "neutral"


def build_narrative_payload(
    bundle: dict[str, Any],
    *,
    site_goal: str = "",
    max_chars: int = 8000,
) -> dict[str, Any]:
    """Compact JSON-safe payload for LLM — excludes DOM trees and heatmaps."""
    sections = []
    for sec in bundle.get("section_report") or []:
        sections.append({
            "section_id": sec.get("section_id"),
            "dwell_sec": sec.get("dwell_sec"),
            "flags": sec.get("flags"),
            "engagement": sec.get("engagement"),
            "activation": sec.get("activation"),
            "emotion": {
                "dominant": (sec.get("emotion") or {}).get("dominant"),
                "mean_z": (sec.get("emotion") or {}).get("mean_z"),
                "peak_z": (sec.get("emotion") or {}).get("peak_z"),
            },
            "recommendations": sec.get("recommendations"),
            "top_elements": [
                {
                    "dom_id": e.get("dom_id"),
                    "tag": e.get("tag"),
                    "role": e.get("role"),
                    "text": e.get("text"),
                    "display_role": classify_role(
                        e.get("dom_id", ""),
                        e.get("tag", ""),
                        e.get("role", ""),
                        e.get("text", ""),
                    ),
                    "attention_density": (
                        e.get("mean_attention_density")
                        or e.get("attention_density")
                        or e.get("score")
                    ),
                    "heatmap_source": e.get("heatmap_source"),
                }
                for e in (sec.get("top_elements") or [])[:5]
            ],
            "heatmap_stats": sec.get("heatmap_stats"),
        })

    element_rows = []
    for row in collect_scored_elements(bundle):
        samples = (bundle.get("section_report") or [])
        sec = next((s for s in samples if s.get("section_id") == row["section_id"]), {})
        salient_t = (sec.get("sample_t_indices") or [None])[0]
        t_idx = int(salient_t) if salient_t is not None else 0
        eng_z = _engagement_at_t(bundle, t_idx)
        element_rows.append({
            **row,
            "salient_t": t_idx,
            "engagement_z": eng_z,
        })

    triggers = bundle.get("grounding_triggers") or []
    events_summary = []
    for ev in (bundle.get("events") or [])[:10]:
        g = ev.get("grounding")
        events_summary.append({
            "t_spike": ev.get("t_spike"),
            "triggers": ev.get("triggers"),
            "dom_id": g.get("dom_id") if g else None,
            "tag": g.get("tag") if g else None,
            "text": g.get("text") if g else None,
        })

    sample_ts = sorted({
        int(t)
        for sec in bundle.get("section_report") or []
        for t in (sec.get("sample_t_indices") or [])
    })[:12]

    frame_samples = []
    for t in sample_ts:
        frame_samples.append({
            "t": t,
            "engagement_z": _engagement_at_t(bundle, t),
            "activation_raw": (
                (bundle.get("activation_track") or {}).get("raw_scores") or [None]
            )[t] if t < len((bundle.get("activation_track") or {}).get("raw_scores") or []) else None,
        })

    payload = {
        "session_id": bundle.get("session_id"),
        "site_goal": site_goal,
        "section_report": sections,
        "scored_elements": element_rows,
        "frame_samples": frame_samples,
        "grounding_triggers_count": len(triggers),
        "events_summary": events_summary,
    }
    at = bundle.get("activation_track") or {}
    if at.get("raw_scores"):
        payload["activation_track"] = {
            "comparison_mode": at.get("comparison_mode"),
            "baseline_flag": at.get("baseline_flag"),
            "session_summary": {
                "mean_raw": round(float(np.mean(at["raw_scores"])), 6),
                "mean_z": round(float(np.mean(at.get("scores") or at["raw_scores"])), 4),
            },
        }
    ms = bundle.get("marketing_scores") or {}
    if ms:
        payload["marketing_scores"] = {
            "overall_score": ms.get("overall_score"),
            "session_metrics": [
                {
                    "key": m.get("key"),
                    "label": m.get("label"),
                    "score": m.get("score"),
                    "summary": m.get("summary"),
                }
                for m in (ms.get("session_metrics") or [])
            ],
            "drop_moments": ms.get("drop_moments") or [],
            "focus_windows": ms.get("focus_windows") or [],
            "sections": [
                {
                    "section_id": s.get("section_id"),
                    "score": s.get("score"),
                    "rank": s.get("rank"),
                    "label": s.get("label"),
                }
                for s in (ms.get("sections") or [])
            ],
            "display_curve_summary": {
                "avg_score": (ms.get("display_curve") or {}).get("avg_score"),
                "max_score": (ms.get("display_curve") or {}).get("max_score"),
                "min_score": (ms.get("display_curve") or {}).get("min_score"),
            },
            "activation_analysis": {
                "avg_score": ((ms.get("activation_analysis") or {}).get("display_curve") or {}).get("avg_score"),
                "sections": [
                    {
                        "section_id": s.get("section_id"),
                        "score": s.get("score"),
                        "rank": s.get("rank"),
                    }
                    for s in ((ms.get("activation_analysis") or {}).get("sections") or [])
                ],
            } if ms.get("activation_analysis") else None,
        }
    text = json.dumps(payload, indent=2)
    if len(text) > max_chars:
        payload["scored_elements"] = element_rows[: max(1, len(element_rows) // 2)]
        payload["section_report"] = sections[: max(1, len(sections) // 2)]
        payload["_truncated"] = True
    return payload


def _template_element_insights(
    bundle: dict[str, Any],
    site_goal: str,
) -> list[ElementInsight]:
    out: list[ElementInsight] = []
    for row in collect_scored_elements(bundle):
        sec = next(
            (s for s in (bundle.get("section_report") or []) if s.get("section_id") == row["section_id"]),
            {},
        )
        salient_t = (sec.get("sample_t_indices") or [0])[0]
        t_idx = int(salient_t) if salient_t is not None else 0
        eng_z = _engagement_at_t(bundle, t_idx)
        density = row.get("attention_density")
        reaction = _infer_reaction(eng_z, row.get("section_flags") or [], density=density)
        text_snip = (row.get("text") or "")[:80]
        label = row.get("display_role") or "component"
        quote_part = f'("{text_snip}") ' if text_snip else ""
        if density is not None:
            plain = (
                f"The {label.lower()} {quote_part}"
                f"in section '{row.get('section_id')}' drew model attention "
                f"(density {density:.4f})."
            )
        else:
            plain = f"The {label.lower()} in section '{row.get('section_id')}' was sampled."
        rec = (
            f"To support the site goal, clarify how this {label.lower()} "
            f"helps visitors: {site_goal[:120]}…"
            if len(site_goal) > 120 else
            f"To support the site goal ({site_goal}), strengthen this {label.lower()}."
        )
        if reaction == "friction":
            rec = f"Reduce visual or cognitive friction here so this block better supports: {site_goal[:100]}."
        elif reaction == "boring":
            rec = f"Increase contrast, specificity, or urgency so this {label.lower()} re-engages attention toward: {site_goal[:100]}."
        out.append(
            ElementInsight(
                dom_id=row["dom_id"],
                section_id=row.get("section_id") or "",
                role=label,
                text=row.get("text") or "",
                job=f"Support the site goal in the '{row.get('section_id')}' section.",
                reaction=reaction,
                attention_density=float(density) if density is not None else None,
                engagement_z=eng_z,
                emotion=row.get("dominant_emotion"),
                salient_t=t_idx,
                plain_summary=plain,
                recommendation=rec,
            )
        )
    return out


def _template_frame_insights(bundle: dict[str, Any]) -> list[FrameInsight]:
    frames: list[FrameInsight] = []
    sample_ts = sorted({
        int(t)
        for sec in bundle.get("section_report") or []
        for t in (sec.get("sample_t_indices") or [])
    })[:12]
    for t in sample_ts:
        eng = _engagement_at_t(bundle, t)
        label = "neutral"
        if eng is not None:
            if eng >= 1.0:
                label = "engaging"
            elif eng <= -1.0:
                label = "boring"
        sec = next(
            (s for s in (bundle.get("section_report") or []) if t in (s.get("sample_t_indices") or [])),
            {},
        )
        sid = sec.get("section_id", "page")
        caption = (
            f"At TR {t} ({sid}), engagement reads {label}"
            + (f" (Z={eng:.2f})" if eng is not None else "")
            + " — model-relative attention and neural proxies, not eye-tracking."
        )
        frames.append(FrameInsight(t=t, caption=caption))
    return frames


def _template_narrative(bundle: dict[str, Any], site_goal: str) -> MarketingNarrative:
    """Deterministic fallback when no LLM API is configured."""
    sections_out: list[MarketingNarrativeSection] = []
    weak = 0
    ms = bundle.get("marketing_scores") or {}
    ms_sections = {s.get("section_id"): s for s in (ms.get("sections") or [])}
    for sec in bundle.get("section_report") or []:
        flags = sec.get("flags") or []
        recs = sec.get("recommendations") or []
        sid = sec.get("section_id", "unknown")
        if flags:
            weak += 1
        ms_row = ms_sections.get(sid) or {}
        score_bit = ""
        if ms_row.get("score") is not None:
            score_bit = f" Marketing score {ms_row['score']}/100."
        narrative = (
            f"Section '{sid}' showed flags {', '.join(flags) or 'none'} "
            f"over {sec.get('dwell_sec', 0):.0f}s dwell (model-relative proxies).{score_bit}"
        )
        sections_out.append(
            MarketingNarrativeSection(
                section_id=sid,
                narrative=narrative,
                actions=recs[:3] if recs else ["Review layout and CTA hierarchy in this block."],
            )
        )
    summary = (
        f"Site goal: {site_goal} "
        f"Analysis covers {len(sections_out)} page section(s); "
        f"{weak} had elevated friction or valence flags. "
        "Scores are session-relative neural hypotheses, not eye-tracking or clinical labels."
    )
    if ms.get("overall_score") is not None:
        summary = (
            f"Session marketing score {ms['overall_score']}/100 (session-relative). "
            + summary
        )
    return MarketingNarrative(
        executive_summary=summary,
        site_goal=site_goal,
        sections=sections_out,
        element_insights=_template_element_insights(bundle, site_goal),
        frame_insights=_template_frame_insights(bundle),
        provider="template",
        model=None,
    )


def _call_openai(cfg: dict[str, Any], system: str, user: str) -> str:
    api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc
    client = OpenAI(api_key=api_key)
    model = cfg.get("model", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=float(cfg.get("temperature", 0.3)),
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


def _call_gemini(cfg: dict[str, Any], system: str, user: str) -> str:
    api_key = os.environ.get(cfg.get("api_key_env", "GEMINI_API_KEY"), "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    model = cfg.get("model", "gemini-2.0-flash")
    prompt = f"{system}\n\n{user}" if system else user
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=float(cfg.get("temperature", 0.3)),
            response_mime_type="application/json",
        ),
    )
    return resp.text or "{}"


def _parse_llm_json(raw: str, provider: str, model: str | None, site_goal: str) -> MarketingNarrative:
    data = json.loads(raw)
    sections = [
        MarketingNarrativeSection(
            section_id=s.get("section_id", ""),
            narrative=s.get("narrative", ""),
            actions=list(s.get("actions") or []),
        )
        for s in data.get("sections", [])
    ]
    element_insights = [
        ElementInsight(
            dom_id=e.get("dom_id", ""),
            section_id=e.get("section_id", ""),
            role=e.get("role", ""),
            text=e.get("text", ""),
            job=e.get("job", ""),
            reaction=e.get("reaction", "neutral"),
            attention_density=e.get("attention_density"),
            engagement_z=e.get("engagement_z"),
            emotion=e.get("emotion"),
            salient_t=e.get("salient_t"),
            plain_summary=e.get("plain_summary", ""),
            recommendation=e.get("recommendation", ""),
        )
        for e in data.get("element_insights", [])
    ]
    frame_insights = [
        FrameInsight(t=int(f.get("t", 0)), caption=f.get("caption", ""))
        for f in data.get("frame_insights", [])
    ]
    return MarketingNarrative(
        executive_summary=data.get("executive_summary", ""),
        site_goal=data.get("site_goal") or site_goal,
        sections=sections,
        element_insights=element_insights,
        frame_insights=frame_insights,
        provider=provider,
        model=model,
    )


def generate_marketing_narrative(
    bundle: dict[str, Any],
    *,
    site_goal: str = "",
    config_path: Path | None = None,
    provider: str | None = None,
) -> MarketingNarrative:
    """Produce marketing_narrative from structured bundle fields."""
    from scout_core.element_goals import DEFAULT_SITE_GOAL

    goal = site_goal.strip() or DEFAULT_SITE_GOAL
    cfg = load_llm_config(config_path)
    prov = provider or cfg.get("provider", "template")
    payload = build_narrative_payload(
        bundle,
        site_goal=goal,
        max_chars=int(cfg.get("max_section_chars", 8000)),
    )
    prompt_cfg = cfg.get("prompt") or {}
    system = prompt_cfg.get("system", "")
    user_task = prompt_cfg.get("user_task", "")
    user = f"{user_task}\n\n```json\n{json.dumps(payload, indent=2)}\n```"

    if prov == "template":
        return _template_narrative(bundle, goal)

    try:
        if prov == "openai":
            raw = _call_openai(cfg.get("openai") or {}, system, user)
            return _parse_llm_json(raw, "openai", (cfg.get("openai") or {}).get("model"), goal)
        if prov == "gemini":
            raw = _call_gemini(cfg.get("gemini") or {}, system, user)
            return _parse_llm_json(raw, "gemini", (cfg.get("gemini") or {}).get("model"), goal)
        raise ValueError(f"Unsupported LLM provider: {prov}")
    except Exception:
        fallback = _template_narrative(bundle, goal)
        fallback.executive_summary = (
            f"[LLM unavailable — template fallback] {fallback.executive_summary}"
        )
        return fallback
