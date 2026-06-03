"""Tests for element role labels and site goal resolution."""

from __future__ import annotations

from pathlib import Path

from scout_core.element_goals import (
    DEFAULT_SITE_GOAL,
    classify_role,
    collect_scored_elements,
    resolve_site_goal,
)


def test_classify_role_cta():
    assert classify_role("#cta", "BUTTON", "", "Start free trial") == "CTA"


def test_classify_role_headline():
    assert classify_role("#hero-h1", "H1", "", "Ship faster") == "headline"


def test_classify_role_nav():
    assert classify_role("nav.main", "NAV", "navigation", "") == "nav"


def test_resolve_site_goal_cli_override(tmp_path: Path):
    script = tmp_path / "script.yaml"
    script.write_text("site_goal: From YAML\n", encoding="utf-8")
    assert resolve_site_goal(override="From CLI", script_path=script) == "From CLI"


def test_resolve_site_goal_from_script(tmp_path: Path):
    script = tmp_path / "script.yaml"
    script.write_text(
        "site_goal: |\n  Make visitors trust the product.\n",
        encoding="utf-8",
    )
    goal = resolve_site_goal(script_path=script)
    assert "trust" in goal


def test_resolve_site_goal_default():
    assert resolve_site_goal() == DEFAULT_SITE_GOAL


def test_collect_scored_elements_dedupes():
    bundle = {
        "section_report": [
            {
                "section_id": "hero",
                "flags": [],
                "emotion": {"dominant": "neutral"},
                "top_elements": [
                    {"dom_id": "#cta", "tag": "BUTTON", "text": "Go", "attention_density": 0.01},
                    {"dom_id": "#cta", "tag": "BUTTON", "text": "Go", "attention_density": 0.02},
                ],
            }
        ],
    }
    rows = collect_scored_elements(bundle)
    assert len(rows) == 1
    assert rows[0]["display_role"] == "CTA"
    assert rows[0]["text"] == "Go"
