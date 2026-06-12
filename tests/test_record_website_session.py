"""Tests for walkthrough manifest v2, step scheduling, and alignment helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scout_core.session_align import (
    find_walkthrough_video,
    is_timestep_in_manifest,
    tr_duration_from_manifest,
    validate_preds_manifest_alignment,
)
from scout_core.walkthrough import (
    _run_step,
    build_manifest_v2,
    build_step_schedule,
    expected_tr_count,
    interaction_t_idx,
    load_walkthrough_script,
    plan_linear_scroll_y_targets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "configs" / "walkthrough_scripts" / "localhost_demo.yaml"


def test_load_walkthrough_script():
    script = load_walkthrough_script(SCRIPT_PATH)
    assert "initial_url" in script
    assert script.get("viewport", {}).get("width") == 1280


def test_expected_tr_count():
    assert expected_tr_count(5.0, 1.0) == 5
    assert expected_tr_count(5.1, 1.0) == 6


def test_build_manifest_v2_capture_matches_viewport():
    snaps = [{"t_idx": 0, "pts_sec": 0.0, "url": "http://x/", "scrollY": 0, "scrollX": 0, "elements": []}]
    m = build_manifest_v2(
        session_id="abc",
        initial_url="http://x/",
        dom_snapshots=snaps,
        width=1280,
        height=720,
        interval_sec=1.0,
    )
    assert m["schema_version"] == 2
    assert m["capture"]["width"] == 1280
    assert m["capture"]["height"] == 720
    assert m["capture"]["tr_duration_sec"] == 1.0
    assert m["tr_mapping"]["type"] == "linear"
    assert m["video"]["path"] == "walkthrough.mp4"


def test_build_manifest_v2_respects_duration_and_custom_paths():
    snaps = [{"t_idx": 0, "pts_sec": 0.0, "url": "http://x/", "scrollY": 0, "scrollX": 0, "elements": []}]
    m = build_manifest_v2(
        session_id="abc",
        initial_url="http://x/",
        dom_snapshots=snaps,
        width=1440,
        height=900,
        interval_sec=0.5,
        duration_sec=7.25,
        video_path="walkthrough.webm",
        walkthrough_script="configs/walkthrough_scripts/demo.yaml",
    )
    assert m["video"]["path"] == "walkthrough.webm"
    assert m["video"]["duration_sec"] == pytest.approx(7.25)
    assert m["capture"]["fps"] == pytest.approx(2.0)
    assert m["walkthrough_script"] == "configs/walkthrough_scripts/demo.yaml"


def test_validate_preds_manifest_alignment_ok():
    m = {"dom_snapshots": [{"t_idx": i} for i in range(10)]}
    r = validate_preds_manifest_alignment(10, m, tolerance=1)
    assert r["ok"] is True


def test_validate_preds_manifest_alignment_warn():
    m = {"dom_snapshots": [{"t_idx": i} for i in range(5)]}
    r = validate_preds_manifest_alignment(10, m, tolerance=1)
    assert r["ok"] is False


def test_find_walkthrough_video_prefers_manifest_path(tmp_path):
    manifest = {
        "video": {"path": "nested/custom_walkthrough.webm"},
        "dom_snapshots": [],
    }
    nested = tmp_path / "nested"
    nested.mkdir()
    expected = nested / "custom_walkthrough.webm"
    expected.write_bytes(b"video")
    (tmp_path / "walkthrough.mp4").write_bytes(b"fallback")
    (tmp_path / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert find_walkthrough_video(tmp_path) == expected


def test_find_walkthrough_video_falls_back_to_glob(tmp_path):
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"video")

    assert find_walkthrough_video(tmp_path) == video


def test_tr_duration_from_manifest_prefers_explicit_value():
    manifest = {"capture": {"tr_duration_sec": 2.5, "fps": 99.0}}
    assert tr_duration_from_manifest(manifest) == pytest.approx(2.5)


def test_tr_duration_from_manifest_falls_back_to_fps():
    manifest = {"capture": {"fps": 4.0}}
    assert tr_duration_from_manifest(manifest) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# build_step_schedule
# ---------------------------------------------------------------------------

def test_build_step_schedule_wait_ms_advances_clock():
    steps = [
        {"action": "wait_ms", "ms": 500},
        {"action": "scroll_to_y", "y": 400},
        {"action": "wait_ms", "ms": 500},
        {"action": "scroll_to_y", "y": 800},
    ]
    sched = build_step_schedule(steps)
    assert len(sched) == 4
    offsets = [t for t, _ in sched]
    assert offsets[0] == pytest.approx(0.0)   # wait_ms starts at t=0
    assert offsets[1] == pytest.approx(0.5)   # scroll after first 500ms
    assert offsets[2] == pytest.approx(0.5)   # second wait_ms starts immediately after scroll
    assert offsets[3] == pytest.approx(1.0)   # second scroll after 1000ms total


def test_build_step_schedule_non_wait_does_not_advance_clock():
    steps = [
        {"action": "scroll_to_y", "y": 100},
        {"action": "scroll_to_y", "y": 200},
        {"action": "click", "selector": "#btn"},
    ]
    sched = build_step_schedule(steps)
    offsets = [t for t, _ in sched]
    assert all(o == pytest.approx(0.0) for o in offsets)


def test_build_step_schedule_empty():
    assert build_step_schedule([]) == []


def test_plan_linear_scroll_y_targets():
    # 4000px page, 720px viewport → max scroll 3280; step 540 → 0,540,1080,1620,2160,2700,3280
    targets = plan_linear_scroll_y_targets(4000, 720, 540)
    assert targets[0] == 0
    assert targets[-1] == 3280
    assert len(targets) == 8
    assert all(targets[i] <= targets[i + 1] for i in range(len(targets) - 1))


def test_plan_linear_scroll_short_page():
    assert plan_linear_scroll_y_targets(800, 720, 540) == [0, 80]


def test_build_step_schedule_preserves_step_identity():
    step = {"action": "wait_ms", "ms": 1000}
    sched = build_step_schedule([step])
    _, returned_step = sched[0]
    assert returned_step is step


# ---------------------------------------------------------------------------
# Extended alignment helpers
# ---------------------------------------------------------------------------

def test_validate_alignment_returns_mapping_exact():
    m = {"dom_snapshots": [{"t_idx": i} for i in range(5)]}
    r = validate_preds_manifest_alignment(5, m, tolerance=1)
    assert r["mapping"] == "exact"
    assert r["overlap_end"] == 4
    assert r["out_of_range_count"] == 0


def test_validate_alignment_returns_mapping_tolerated():
    m = {"dom_snapshots": [{"t_idx": i} for i in range(5)]}
    r = validate_preds_manifest_alignment(6, m, tolerance=1)
    assert r["ok"] is True
    assert r["mapping"] == "tolerated"
    assert r["out_of_range_count"] == 1


def test_validate_alignment_returns_mapping_clamped():
    m = {"dom_snapshots": [{"t_idx": i} for i in range(5)]}
    r = validate_preds_manifest_alignment(10, m, tolerance=1)
    assert r["ok"] is False
    assert r["mapping"] == "clamped"
    assert r["out_of_range_count"] == 5


def test_validate_alignment_returns_mapping_invalid():
    m = {"dom_snapshots": []}
    r = validate_preds_manifest_alignment(5, m, tolerance=1)
    assert r["mapping"] == "invalid"


def test_is_timestep_in_manifest():
    m = {"dom_snapshots": [{"t_idx": i} for i in range(5)]}  # max index 4
    assert is_timestep_in_manifest(0, m) is True
    assert is_timestep_in_manifest(4, m) is True
    assert is_timestep_in_manifest(5, m) is False
    assert is_timestep_in_manifest(5, m, tolerance=1) is True
    assert is_timestep_in_manifest(6, m, tolerance=1) is False


def test_build_manifest_v2_round_trips_interaction_events():
    events = [
        {
            "action": "click",
            "selector": "#cta-hero",
            "pts_sec": 1.2,
            "t_idx": 1,
            "target": {"dom_id": "#cta-hero", "tag": "BUTTON"},
        },
        {
            "action": "hover",
            "selector": "#nav",
            "pts_sec": 2.0,
            "t_idx": 2,
            "target": {"dom_id": "#nav", "tag": "NAV"},
        },
    ]
    snaps = [{"t_idx": i, "pts_sec": float(i), "url": "http://x/", "scrollY": 0, "scrollX": 0, "elements": []} for i in range(3)]
    manifest = build_manifest_v2(
        session_id="sess",
        initial_url="http://x/",
        dom_snapshots=snaps,
        width=1280,
        height=720,
        interval_sec=1.0,
        interaction_events=events,
    )
    assert len(manifest["interaction_events"]) == 2
    assert manifest["interaction_events"][0]["action"] == "click"
    assert manifest["interaction_events"][0]["t_idx"] == 1
    assert manifest["interaction_events"][1]["selector"] == "#nav"


def test_interaction_t_idx_maps_event_time_to_tr():
    assert interaction_t_idx(0.0, 1.0) == 0
    assert interaction_t_idx(1.2, 1.0) == 1
    assert interaction_t_idx(2.6, 1.0) == 3
    assert interaction_t_idx(0.5, 0.5) == 1


def test_run_step_click_and_hover_return_markers():
    class FakePage:
        async def click(self, selector, timeout=30_000):
            self.last_click = selector

        async def hover(self, selector, timeout=30_000):
            self.last_hover = selector

        async def evaluate(self, js, selector=None):
            return {
                "dom_id": selector,
                "tag": "BUTTON",
                "role": "",
                "text": "Get started",
                "bbox": [10, 10, 100, 40],
            }

    page = FakePage()
    click_marker = asyncio.run(_run_step(page, {"action": "click", "selector": "#cta-hero"}))
    hover_marker = asyncio.run(_run_step(page, {"action": "hover", "selector": "#cta-hero"}))

    assert click_marker["action"] == "click"
    assert click_marker["selector"] == "#cta-hero"
    assert click_marker["target"]["dom_id"] == "#cta-hero"
    assert hover_marker["action"] == "hover"
