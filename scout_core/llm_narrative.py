"""Build LLM prompts from structured analysis_bundle fields (no raw DOM/HTML)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scout_core.schemas import MarketingNarrative, MarketingNarrativeSection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "configs" / "llm_narrative.yaml"


def load_llm_config(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_CFG
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_narrative_payload(bundle: dict[str, Any], *, max_chars: int = 8000) -> dict[str, Any]:
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

    triggers = bundle.get("grounding_triggers") or []
    events_summary = []
    for ev in (bundle.get("events") or [])[:10]:
        g = ev.get("grounding")
        events_summary.append({
            "t_spike": ev.get("t_spike"),
            "triggers": ev.get("triggers"),
            "dom_id": g.get("dom_id") if g else None,
            "tag": g.get("tag") if g else None,
        })

    payload = {
        "session_id": bundle.get("session_id"),
        "section_report": sections,
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
        payload["section_report"] = sections[: max(1, len(sections) // 2)]
        payload["_truncated"] = True
    return payload


def _template_narrative(bundle: dict[str, Any]) -> MarketingNarrative:
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
        sections=sections_out,
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


def _parse_llm_json(raw: str, provider: str, model: str | None) -> MarketingNarrative:
    data = json.loads(raw)
    sections = [
        MarketingNarrativeSection(
            section_id=s.get("section_id", ""),
            narrative=s.get("narrative", ""),
            actions=list(s.get("actions") or []),
        )
        for s in data.get("sections", [])
    ]
    return MarketingNarrative(
        executive_summary=data.get("executive_summary", ""),
        sections=sections,
        provider=provider,
        model=model,
    )


def generate_marketing_narrative(
    bundle: dict[str, Any],
    *,
    config_path: Path | None = None,
    provider: str | None = None,
) -> MarketingNarrative:
    """Produce marketing_narrative from structured bundle fields."""
    cfg = load_llm_config(config_path)
    prov = provider or cfg.get("provider", "template")
    payload = build_narrative_payload(
        bundle,
        max_chars=int(cfg.get("max_section_chars", 8000)),
    )
    prompt_cfg = cfg.get("prompt") or {}
    system = prompt_cfg.get("system", "")
    user_task = prompt_cfg.get("user_task", "")
    user = f"{user_task}\n\n```json\n{json.dumps(payload, indent=2)}\n```"

    if prov == "template":
        return _template_narrative(bundle)

    if prov == "openai":
        raw = _call_openai(cfg.get("openai") or {}, system, user)
        return _parse_llm_json(raw, "openai", (cfg.get("openai") or {}).get("model"))

    raise ValueError(f"Unsupported LLM provider: {prov}")
