from __future__ import annotations

from scout_core.attention_attribution import enrich_sections_with_attribution
from scout_core.element_goals import clickability_score


def test_clickability_scores_cta_higher_than_body():
    cta = {"dom_id": "#cta", "tag": "BUTTON", "text": "Get started", "bbox": [20, 20, 120, 44]}
    body = {"dom_id": "#copy", "tag": "P", "text": "Some body copy", "bbox": [20, 240, 420, 40]}
    assert clickability_score(cta) > clickability_score(body)


def test_attribution_adds_combined_scores():
    sections = [
        {
            "section_id": "hero",
            "t_indices": [0, 1, 2],
            "top_elements": [
                {"dom_id": "#cta", "tag": "BUTTON", "text": "Get started", "bbox": [20, 20, 120, 44], "mean_attention_density": 0.8},
                {"dom_id": "#copy", "tag": "P", "text": "Body", "bbox": [20, 200, 420, 40], "mean_attention_density": 0.2},
            ],
        }
    ]
    bundle = {
        "engagement_track": {"scores": [2.0, 1.2, 0.5]},
        "activation_track": {"scores": [1.0, 0.5, 0.0]},
    }
    enrich_sections_with_attribution(sections, bundle)
    top = sections[0]["top_elements"][0]
    assert top["dom_id"] == "#cta"
    assert top["combined_score"] > 0
    assert top["clickability"] > 50
    assert "element_attribution" in sections[0]
