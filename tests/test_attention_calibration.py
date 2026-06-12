from __future__ import annotations

from scout_core.attention_calibration import (
    load_calibration_config,
    model_scores_from_section,
    spearman_rho,
    top1_hit,
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


def test_load_calibration_config_defaults():
    cfg = load_calibration_config()
    assert "attention_weight" in cfg
