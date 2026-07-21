from __future__ import annotations

import json

import pytest

from scout_core.attention_calibration import (
    fit_attention_weight,
    load_calibration_config,
    model_scores_from_section,
    spearman_rho,
    top1_hit,
    validate_session,
)


def test_spearman_perfect_correlation():
    x = [0.1, 0.5, 0.9]
    y = [0.2, 0.6, 1.0]
    assert spearman_rho(x, y) == 1.0


def test_top1_hit():
    model = {"#a": 0.9, "#b": 0.1}
    beh = {"#a": 0.8, "#b": 0.2}
    assert top1_hit(model, beh) is True


def test_model_scores_from_section():
    sec = {
        "top_elements": [
            {"dom_id": "#cta", "combined_score": 80},
        ],
    }
    scores = model_scores_from_section(sec)
    assert scores["#cta"] == 0.8


def test_model_scores_can_refuse_raw_components():
    sec = {
        "top_elements": [
            {
                "dom_id": "#cta",
                "attention_score": 90,
                "clickability": 10,
                "combined_score": 1,
            },
        ],
    }
    scores = model_scores_from_section(sec, attention_weight=0.75)
    assert scores["#cta"] == pytest.approx(0.7)


def test_load_calibration_config_defaults():
    cfg = load_calibration_config()
    assert "attention_weight" in cfg


def test_nan_spearman_cannot_pass(tmp_path):
    bundle = {
        "section_report": [
            {
                "top_elements": [
                    {"dom_id": "#a", "combined_score": 90},
                    {"dom_id": "#b", "combined_score": 10},
                ]
            }
        ]
    }
    (tmp_path / "analysis_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    (tmp_path / "clarity_clicks.csv").write_text(
        "Selector,Clicks\n#a,5\n#b,5\n",
        encoding="utf-8",
    )
    result = validate_session(
        tmp_path,
        thresholds={"spearman_min": 0.4, "top1_cta_hit_min": 0.0},
    )
    assert result["spearman_mean"] is None
    assert result["passed"] is False


def test_clarity_enriched_bundle_is_rejected_as_leakage(tmp_path):
    bundle = {"section_report": [], "clarity_attribution": {"enabled": True}}
    (tmp_path / "analysis_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    result = validate_session(tmp_path)
    assert result["status"] == "invalid"
    assert result["reason"] == "clarity_leakage_in_analysis_bundle"


def test_weight_fit_rejects_holdout_role(tmp_path):
    with pytest.raises(ValueError, match="Holdout"):
        fit_attention_weight(tmp_path, roles={"holdout"})
