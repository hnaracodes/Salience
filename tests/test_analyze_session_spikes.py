"""Unit tests for analyze_session spike selection (Z-based emotion triggers)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_session import _select_spikes  # noqa: E402


def test_select_spikes_anger_z_above_threshold():
    bundle = {
        "grounding_triggers": [
            {
                "t_idx": 5,
                "trigger_type": "emotion",
                "channel": "anger",
                "value": 2.5,
                "raw_cosine": 0.062,
                "z_score": 2.5,
            },
        ],
    }
    spikes = _select_spikes(bundle, eng_min=2.0, anger_z_min=2.0, combine="or", cooldown_trs=0, max_spikes=0)
    assert len(spikes) == 1
    assert spikes[0]["t_idx"] == 5
    assert spikes[0]["triggers"]["kragel_anger_z"] == pytest.approx(2.5)


def test_select_spikes_anger_z_below_threshold():
    bundle = {
        "grounding_triggers": [
            {
                "t_idx": 5,
                "trigger_type": "emotion",
                "channel": "anger",
                "value": 1.5,
                "raw_cosine": 0.9,
                "z_score": 1.5,
            },
        ],
    }
    spikes = _select_spikes(bundle, eng_min=2.0, anger_z_min=2.0, combine="or", cooldown_trs=0, max_spikes=0)
    assert spikes == []


def test_select_spikes_engagement_or_anger_z():
    bundle = {
        "grounding_triggers": [
            {"t_idx": 1, "trigger_type": "engagement", "channel": "engagement", "value": 2.1},
            {"t_idx": 2, "trigger_type": "emotion", "channel": "anger", "value": 2.1, "z_score": 2.1},
        ],
    }
    spikes = _select_spikes(bundle, eng_min=2.0, anger_z_min=2.0, combine="or", cooldown_trs=0, max_spikes=0)
    assert len(spikes) == 2
