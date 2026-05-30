"""Unit tests for analyze_session spike selection (Z-based emotion triggers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_session import _run_grounding_step, _select_spikes  # noqa: E402


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
    spikes = _select_spikes(bundle, eng_min=2.0, emotion_z_min=2.0, combine="or", cooldown_trs=0, max_spikes=0)
    assert len(spikes) == 1
    assert spikes[0]["t_idx"] == 5
    # Anger channel backward-compat: kragel_anger_z still written.
    assert spikes[0]["triggers"]["kragel_anger_z"] == pytest.approx(2.5)
    assert spikes[0]["triggers"]["emotion_channel"] == "anger"
    assert spikes[0]["triggers"]["emotion_z"] == pytest.approx(2.5)


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
    spikes = _select_spikes(bundle, eng_min=2.0, emotion_z_min=2.0, combine="or", cooldown_trs=0, max_spikes=0)
    assert spikes == []


def test_select_spikes_engagement_or_anger_z():
    bundle = {
        "grounding_triggers": [
            {"t_idx": 1, "trigger_type": "engagement", "channel": "engagement", "value": 2.1},
            {"t_idx": 2, "trigger_type": "emotion", "channel": "anger", "value": 2.1, "z_score": 2.1},
        ],
    }
    spikes = _select_spikes(bundle, eng_min=2.0, emotion_z_min=2.0, combine="or", cooldown_trs=0, max_spikes=0)
    assert len(spikes) == 2


def test_select_spikes_non_anger_emotion_triggers():
    """Any emotion channel (sadness, fear, etc.) should qualify as a trigger."""
    bundle = {
        "grounding_triggers": [
            {"t_idx": 3, "trigger_type": "emotion", "channel": "sadness", "value": 2.02},
            {"t_idx": 7, "trigger_type": "emotion", "channel": "fear", "value": 2.8},
        ],
    }
    spikes = _select_spikes(bundle, eng_min=2.0, emotion_z_min=2.0, combine="or", cooldown_trs=0, max_spikes=0)
    assert len(spikes) == 2
    assert spikes[0]["t_idx"] == 3
    assert spikes[0]["triggers"]["emotion_channel"] == "sadness"
    assert spikes[0]["triggers"]["emotion_z"] == pytest.approx(2.02)
    # sadness != anger so kragel_anger_z should NOT be present.
    assert "kragel_anger_z" not in spikes[0]["triggers"]
    assert spikes[1]["triggers"]["emotion_channel"] == "fear"


def test_select_spikes_all_qualifying_emotion_channels_per_timestep():
    """When multiple emotion channels fire at the same timestep, emit one spike each."""
    bundle = {
        "grounding_triggers": [
            {"t_idx": 5, "trigger_type": "emotion", "channel": "sadness", "value": 2.1},
            {"t_idx": 5, "trigger_type": "emotion", "channel": "anger", "value": 3.0},
            {"t_idx": 5, "trigger_type": "emotion", "channel": "fear", "value": 1.5},
        ],
    }
    spikes = _select_spikes(bundle, eng_min=2.0, emotion_z_min=2.0, combine="or", cooldown_trs=0, max_spikes=0)
    assert len(spikes) == 2
    assert {s["triggers"]["emotion_channel"] for s in spikes} == {"anger", "sadness"}
    anger = next(s for s in spikes if s["triggers"]["emotion_channel"] == "anger")
    assert anger["triggers"]["emotion_z"] == pytest.approx(3.0)
    assert anger["triggers"]["kragel_anger_z"] == pytest.approx(3.0)
    sadness = next(s for s in spikes if s["triggers"]["emotion_channel"] == "sadness")
    assert sadness["triggers"]["emotion_z"] == pytest.approx(2.1)


def test_select_spikes_requires_both_signals_for_and_mode():
    bundle = {
        "grounding_triggers": [
            {"t_idx": 1, "trigger_type": "engagement", "channel": "engagement", "value": 2.2},
            {"t_idx": 1, "trigger_type": "emotion", "channel": "anger", "value": 2.4, "z_score": 2.4},
            {"t_idx": 2, "trigger_type": "engagement", "channel": "engagement", "value": 2.5},
            {"t_idx": 3, "trigger_type": "emotion", "channel": "anger", "value": 2.6, "z_score": 2.6},
        ],
    }

    spikes = _select_spikes(bundle, eng_min=2.0, emotion_z_min=2.0, combine="and", cooldown_trs=0, max_spikes=0)

    assert len(spikes) == 2
    assert all(spike["t_idx"] == 1 for spike in spikes)
    eng = [s for s in spikes if "engagement_score" in s["triggers"]]
    anger = [s for s in spikes if s["triggers"].get("emotion_channel") == "anger"]
    assert len(eng) == 1
    assert len(anger) == 1
    assert eng[0]["triggers"]["engagement_score"] == pytest.approx(2.2)
    assert anger[0]["triggers"]["kragel_anger_z"] == pytest.approx(2.4)


def test_select_spikes_applies_cooldown_and_max_spikes():
    bundle = {
        "grounding_triggers": [
            {"t_idx": 1, "trigger_type": "engagement", "channel": "engagement", "value": 2.1},
            {"t_idx": 2, "trigger_type": "engagement", "channel": "engagement", "value": -2.3},
            {"t_idx": 3, "trigger_type": "emotion", "channel": "anger", "value": 2.5, "z_score": 2.5},
            {"t_idx": 10, "trigger_type": "emotion", "channel": "anger", "value": 2.7, "z_score": 2.7},
        ],
    }

    spikes = _select_spikes(bundle, eng_min=2.0, emotion_z_min=2.0, combine="or", cooldown_trs=2, max_spikes=2)

    assert [spike["t_idx"] for spike in spikes] == [1, 3]
    assert spikes[0]["triggers"]["engagement_score"] == pytest.approx(2.1)
    assert spikes[1]["triggers"]["kragel_anger_z"] == pytest.approx(2.5)


def test_run_grounding_step_returns_grounded_event(tmp_path):
    session_dir = tmp_path / "session"
    heatmaps_dir = session_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True)

    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[10:20, 10:20] = 5.0
    np.save(heatmaps_dir / "t_5.npy", heatmap)

    manifest = {
        "dom_snapshots": [
            {
                "t_idx": 5,
                "scrollY": 0,
                "scrollX": 0,
                "elements": [
                    {"dom_id": "#hero", "tag": "SECTION", "bbox": [0, 0, 30, 30], "is_intersecting_viewport": True},
                    {"dom_id": "#cta", "tag": "BUTTON", "bbox": [10, 10, 10, 10], "is_intersecting_viewport": True},
                ],
            }
        ]
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bundle = {
        "grounding_triggers": [
            {"t_idx": 5, "trigger_type": "emotion", "channel": "anger", "value": 2.5, "z_score": 2.5},
        ],
    }
    iso_cfg = {
        "trigger": {"engagement_score_min": 2.0, "kragel_anger_z_min": 2.0, "combine": "or"},
        "spike_policy": {"max_spikes_per_session": 20, "cooldown_trs": 0},
        "heatmaps_dir": "heatmaps",
    }

    events = _run_grounding_step(session_dir, bundle, iso_cfg)

    assert len(events) == 1
    assert events[0]["type"] == "neural_spike_grounding"
    assert events[0]["t_spike"] == 5
    assert events[0]["triggers"]["kragel_anger_z"] == pytest.approx(2.5)
    assert events[0]["grounding"]["dom_id"] == "#cta"
    assert events[0]["grounding_skip_reason"] is None
    # Provenance should be attached even when no manifest.json is present.
    assert "heatmap_provenance" in events[0]


def test_run_grounding_step_sadness_trigger_fires(tmp_path):
    """Non-anger emotion triggers (sadness) must produce grounding events."""
    session_dir = tmp_path / "session"
    heatmaps_dir = session_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True)

    heatmap = np.zeros((50, 50), dtype=np.float32)
    heatmap[5:15, 5:15] = 4.0
    np.save(heatmaps_dir / "t_3.npy", heatmap)

    manifest = {
        "dom_snapshots": [
            {
                "t_idx": 3,
                "scrollY": 0,
                "scrollX": 0,
                "elements": [
                    {"dom_id": "#btn", "tag": "BUTTON", "bbox": [4, 4, 12, 12], "is_intersecting_viewport": True},
                ],
            }
        ]
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bundle = {
        "grounding_triggers": [
            {"t_idx": 3, "trigger_type": "emotion", "channel": "sadness", "value": 2.1},
        ],
    }
    iso_cfg = {
        "trigger": {"engagement_score_min": 2.0, "emotion_z_min": 2.0, "combine": "or"},
        "spike_policy": {"max_spikes_per_session": 20, "cooldown_trs": 0},
        "heatmaps_dir": "heatmaps",
    }

    events = _run_grounding_step(session_dir, bundle, iso_cfg)

    assert len(events) == 1
    assert events[0]["grounding_skip_reason"] is None
    assert events[0]["triggers"]["emotion_channel"] == "sadness"
    assert "kragel_anger_z" not in events[0]["triggers"]


def test_run_grounding_step_records_skip_reason_when_heatmap_missing(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    manifest = {
        "dom_snapshots": [
            {"t_idx": 5, "scrollY": 0, "scrollX": 0, "elements": []},
        ]
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bundle = {
        "grounding_triggers": [
            {"t_idx": 5, "trigger_type": "emotion", "channel": "anger", "value": 2.5, "z_score": 2.5},
        ],
    }
    iso_cfg = {
        "trigger": {"engagement_score_min": 2.0, "kragel_anger_z_min": 2.0, "combine": "or"},
        "spike_policy": {"max_spikes_per_session": 20, "cooldown_trs": 0},
        "heatmaps_dir": "heatmaps",
    }

    events = _run_grounding_step(session_dir, bundle, iso_cfg)

    assert len(events) == 1
    assert events[0]["grounding"] is None
    assert events[0]["grounding_skip_reason"] == "heatmap_not_found"


def test_run_grounding_step_multi_emotion_same_timestep(tmp_path):
    """Each qualifying emotion at the same TR gets its own grounded event."""
    session_dir = tmp_path / "session"
    heatmaps_dir = session_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True)

    heatmap = np.zeros((50, 50), dtype=np.float32)
    heatmap[5:15, 5:15] = 4.0
    np.save(heatmaps_dir / "t_3.npy", heatmap)

    manifest = {
        "dom_snapshots": [
            {
                "t_idx": 3,
                "scrollY": 0,
                "scrollX": 0,
                "elements": [
                    {"dom_id": "#btn", "tag": "BUTTON", "bbox": [4, 4, 12, 12], "is_intersecting_viewport": True},
                ],
            }
        ]
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bundle = {
        "grounding_triggers": [
            {"t_idx": 3, "trigger_type": "emotion", "channel": "sadness", "value": 2.1},
            {"t_idx": 3, "trigger_type": "emotion", "channel": "fear", "value": 2.5},
        ],
    }
    iso_cfg = {
        "trigger": {"engagement_score_min": 2.0, "emotion_z_min": 2.0, "combine": "or"},
        "spike_policy": {"max_spikes_per_session": 20, "cooldown_trs": 0},
        "heatmaps_dir": "heatmaps",
    }

    events = _run_grounding_step(session_dir, bundle, iso_cfg)

    assert len(events) == 2
    channels = {ev["triggers"]["emotion_channel"] for ev in events}
    assert channels == {"sadness", "fear"}
    assert all(ev["grounding"]["dom_id"] == "#btn" for ev in events)
    assert all(ev["grounding_skip_reason"] is None for ev in events)


def test_run_grounding_step_records_skip_reason_when_no_snapshot_exists(tmp_path):
    session_dir = tmp_path / "session"
    heatmaps_dir = session_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True)
    np.save(heatmaps_dir / "t_5.npy", np.ones((20, 20), dtype=np.float32))
    (session_dir / "session_manifest.json").write_text(json.dumps({"dom_snapshots": []}), encoding="utf-8")

    bundle = {
        "grounding_triggers": [
            {"t_idx": 5, "trigger_type": "emotion", "channel": "anger", "value": 2.5, "z_score": 2.5},
        ],
    }
    iso_cfg = {
        "trigger": {"engagement_score_min": 2.0, "kragel_anger_z_min": 2.0, "combine": "or"},
        "spike_policy": {"max_spikes_per_session": 20, "cooldown_trs": 0},
        "heatmaps_dir": "heatmaps",
    }

    events = _run_grounding_step(session_dir, bundle, iso_cfg)

    assert len(events) == 1
    assert events[0]["grounding"] is None
    assert events[0]["grounding_skip_reason"] == "no_manifest_snapshot"
