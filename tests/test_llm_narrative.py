"""Tests for LLM narrative payload builder (no API calls)."""

from __future__ import annotations

import json

from scout_core.llm_narrative import build_narrative_payload, generate_marketing_narrative


def test_build_narrative_payload_excludes_dom_elements():
    bundle = {
        "session_id": "x",
        "section_report": [
            {
                "section_id": "hero",
                "dwell_sec": 5.0,
                "flags": ["high_arousal"],
                "engagement": {"mean": 1.2},
                "emotion": {"dominant": "anger", "mean_z": {"anger": 1.5}},
                "recommendations": ["Simplify CTA"],
                "top_elements": [{"dom_id": "#hero", "tag": "SECTION", "attention_density": 0.4}],
            }
        ],
        "grounding_triggers": [{"t_idx": 3}],
    }
    payload = build_narrative_payload(bundle)
    text = json.dumps(payload)
    assert '"elements":' not in text
    assert payload["section_report"][0]["section_id"] == "hero"


def test_template_narrative():
    bundle = {
        "section_report": [
            {"section_id": "pricing", "dwell_sec": 8, "flags": [], "recommendations": []},
        ],
    }
    nar = generate_marketing_narrative(bundle, provider="template")
    assert nar.executive_summary
    assert len(nar.sections) == 1
    assert nar.provider == "template"
