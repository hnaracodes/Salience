"""Tests for LLM narrative payload builder (no API calls)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scout_core.llm_narrative import build_narrative_payload, generate_marketing_narrative


def _sample_bundle():
    return {
        "session_id": "x",
        "section_report": [
            {
                "section_id": "hero",
                "dwell_sec": 5.0,
                "flags": ["high_arousal"],
                "engagement": {"mean": 1.2},
                "emotion": {"dominant": "anger", "mean_z": {"anger": 1.5}},
                "recommendations": ["Simplify CTA"],
                "sample_t_indices": [0],
                "top_elements": [{
                    "dom_id": "#cta",
                    "tag": "BUTTON",
                    "text": "Start now",
                    "attention_density": 0.4,
                }],
            }
        ],
        "engagement_track": {"scores": [0.5], "labels": ["neutral"]},
        "grounding_triggers": [{"t_idx": 3}],
    }


def test_build_narrative_payload_excludes_dom_elements():
    bundle = _sample_bundle()
    payload = build_narrative_payload(bundle, site_goal="Convert visitors.")
    text = json.dumps(payload)
    assert '"elements":' not in text
    assert payload["site_goal"] == "Convert visitors."
    assert payload["scored_elements"][0]["text"] == "Start now"


def test_build_narrative_payload_includes_marketing_scores():
    bundle = {
        "session_id": "x",
        "section_report": [{"section_id": "hero", "dwell_sec": 5.0, "flags": [], "recommendations": []}],
        "marketing_scores": {
            "overall_score": 72,
            "session_metrics": [{"key": "opening", "label": "Opening", "score": 80, "summary": "ok"}],
            "sections": [{"section_id": "hero", "score": 74, "rank": 1, "label": "strongest"}],
            "display_curve": {"avg_score": 65.0},
        },
    }
    payload = build_narrative_payload(bundle)
    assert payload["marketing_scores"]["overall_score"] == 72


def test_template_narrative_with_element_insights():
    bundle = _sample_bundle()
    nar = generate_marketing_narrative(
        bundle,
        site_goal="Drive sign-ups.",
        provider="template",
    )
    assert nar.executive_summary
    assert nar.site_goal == "Drive sign-ups."
    assert len(nar.sections) == 1
    assert len(nar.element_insights) >= 1
    assert nar.element_insights[0].dom_id == "#cta"
    assert len(nar.frame_insights) >= 1
    assert nar.provider == "template"


def test_gemini_parse_mocked():
    bundle = _sample_bundle()
    fake_json = json.dumps({
        "executive_summary": "Summary.",
        "site_goal": "Goal.",
        "sections": [{"section_id": "hero", "narrative": "N", "actions": ["A"]}],
        "element_insights": [{
            "dom_id": "#cta",
            "section_id": "hero",
            "role": "CTA",
            "text": "Start now",
            "job": "Convert",
            "reaction": "engaging",
            "plain_summary": "Strong CTA.",
            "recommendation": "Make button larger.",
        }],
        "frame_insights": [{"t": 0, "caption": "Opening moment."}],
    })
    with patch("scout_core.llm_narrative._call_gemini", return_value=fake_json):
        nar = generate_marketing_narrative(bundle, site_goal="Goal.", provider="gemini")
    assert nar.provider == "gemini"
    assert nar.element_insights[0].job == "Convert"


def test_gemini_fallback_to_template_on_error():
    bundle = _sample_bundle()
    with patch("scout_core.llm_narrative._call_gemini", side_effect=RuntimeError("no key")):
        nar = generate_marketing_narrative(bundle, site_goal="Goal.", provider="gemini")
    assert nar.provider == "template"
    assert "[LLM unavailable" in nar.executive_summary

