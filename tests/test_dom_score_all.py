"""Tests for dom_intersect score_all_elements and section filters."""

from __future__ import annotations

import numpy as np
import pytest

from scout_core.dom_intersect import (
    filter_elements_to_section,
    rollup_section_elements,
    score_all_elements,
    select_winner,
)


def test_score_all_sorted():
    h = np.zeros((100, 100), dtype=np.float32)
    h[40:60, 40:60] = 2.0
    elements = [
        {"dom_id": "#low", "tag": "DIV", "bbox": [0, 0, 10, 10], "is_intersecting_viewport": True},
        {"dom_id": "#high", "tag": "DIV", "bbox": [40, 40, 20, 20], "is_intersecting_viewport": True},
    ]
    scored = score_all_elements(h, elements)
    assert len(scored) == 2
    assert scored[0]["dom_id"] == "#high"
    assert select_winner(h, elements)["dom_id"] == "#high"


def test_filter_elements_to_section():
    elements = [
        {"dom_id": "in", "bbox": [10, 10, 20, 20]},
        {"dom_id": "out", "bbox": [500, 500, 20, 20]},
    ]
    root = [0, 0, 100, 100]
    filtered = filter_elements_to_section(elements, root, overlap_min=0.5)
    assert any(e["dom_id"] == "in" for e in filtered)


def test_rollup_mean_density():
    samples = [
        (0, [{"dom_id": "#a", "tag": "BTN", "bbox": [0, 0, 10, 10], "attention_density": 0.2}]),
        (1, [{"dom_id": "#a", "tag": "BTN", "bbox": [0, 0, 10, 10], "attention_density": 0.4}]),
    ]
    top = rollup_section_elements(samples, top_k=5)
    assert top[0]["dom_id"] == "#a"
    assert top[0]["mean_attention_density"] == pytest.approx(0.3, abs=0.01)
