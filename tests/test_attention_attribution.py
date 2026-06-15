from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scout_core.attention_attribution import (
    aggregate_per_tr_element_scores,
    enrich_sections_with_attribution,
    enrich_sections_with_clarity,
)
from scout_core.element_goals import clickability_score


def test_clickability_scores_cta_higher_than_body():
    cta = {"dom_id": "#cta", "tag": "BUTTON", "text": "Get started", "bbox": [20, 20, 120, 44]}
    body = {"dom_id": "#copy", "tag": "P", "text": "Some body copy", "bbox": [20, 240, 420, 40]}
    assert clickability_score(cta) > clickability_score(body)


def test_attribution_adds_combined_scores_section_blend():
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
    enrich_sections_with_attribution(sections, bundle, mode="section_blend")
    top = sections[0]["top_elements"][0]
    assert top["dom_id"] == "#cta"
    assert top["combined_score"] > 0
    assert top["clickability"] > 50
    assert sections[0]["element_attribution"]["mode"] == "section_blend"


def test_per_tr_sum_ranks_peak_tr_element(tmp_path):
    """Element salient only when engagement peaks should win under per_tr_sum."""
    session_dir = tmp_path
    heatmaps = session_dir / "heatmaps"
    heatmaps.mkdir()
    manifest = {
        "schema_version": 2,
        "capture": {"width": 200, "height": 200},
        "dom_snapshots": [
            {
                "t_idx": 0,
                "pts_sec": 0.0,
                "url": "http://test/",
                "scrollY": 0,
                "scrollX": 0,
                "elements": [
                    {"dom_id": "#low", "tag": "P", "bbox": [10, 10, 80, 30], "is_intersecting_viewport": True, "visibility_ratio": 1.0},
                ],
            },
            {
                "t_idx": 1,
                "pts_sec": 1.0,
                "url": "http://test/",
                "scrollY": 0,
                "scrollX": 0,
                "elements": [
                    {"dom_id": "#low", "tag": "P", "bbox": [10, 10, 80, 30], "is_intersecting_viewport": True, "visibility_ratio": 1.0},
                ],
            },
            {
                "t_idx": 2,
                "pts_sec": 2.0,
                "url": "http://test/",
                "scrollY": 0,
                "scrollX": 0,
                "elements": [
                    {"dom_id": "#peak", "tag": "BUTTON", "text": "Buy", "bbox": [10, 10, 100, 40], "is_intersecting_viewport": True, "visibility_ratio": 1.0},
                ],
            },
        ],
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    low_map = np.zeros((200, 200), dtype=np.float32)
    low_map[10:40, 10:90] = 0.2
    peak_map = np.zeros((200, 200), dtype=np.float32)
    peak_map[10:50, 10:110] = 1.0
    np.save(heatmaps / "t_0.npy", low_map)
    np.save(heatmaps / "t_1.npy", low_map)
    np.save(heatmaps / "t_2.npy", peak_map)

    bundle = {
        "engagement_track": {"scores": [0.0, 0.0, 3.0]},
        "activation_track": {"scores": [0.0, 0.0, 0.0]},
    }
    section = {"section_id": "hero", "t_indices": [0, 1, 2], "top_elements": []}
    rolled = aggregate_per_tr_element_scores(section, bundle, manifest, session_dir, min_element_area=1)
    assert rolled[0]["dom_id"] == "#peak"

    sections = [{"section_id": "hero", "t_indices": [0, 1, 2], "top_elements": [
        {"dom_id": "#low", "tag": "P", "bbox": [10, 10, 80, 30], "mean_attention_density": 0.9},
    ]}]
    enrich_sections_with_attribution(
        sections,
        bundle,
        mode="per_tr_sum",
        session_dir=session_dir,
        manifest=manifest,
        min_element_area=1,
        top_k=5,
    )
    assert sections[0]["top_elements"][0]["dom_id"] == "#peak"
    assert sections[0]["element_attribution"]["mode"] == "per_tr_sum"


def test_clarity_enrichment_boosts_matched_cta():
    sections = [
        {
            "section_id": "hero",
            "top_elements": [
                {"dom_id": "#cta", "tag": "BUTTON", "attention_score": 60.0, "clickability": 40.0, "combined_score": 53.6},
                {"dom_id": "#copy", "tag": "P", "attention_score": 80.0, "clickability": 20.0, "combined_score": 60.8},
            ],
        }
    ]
    rows = [
        {"selector": "#cta", "text": "Get started", "clicks": 50, "sessions": 100, "click_rate": 0.5},
    ]
    meta = enrich_sections_with_clarity(sections, rows, blend_weight=0.5)
    assert meta["n_matched"] == 1
    cta = sections[0]["top_elements"][0]
    assert cta["dom_id"] == "#cta"
    assert cta["clarity_matched"] is True
    assert cta["clarity_clicks"] == 50
    assert cta["clickability"] > 40.0
