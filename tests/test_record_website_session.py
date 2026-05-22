"""Tests for walkthrough manifest v2 and alignment helpers (no live Playwright required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scout_core.session_align import validate_preds_manifest_alignment
from scout_core.walkthrough import (
    build_manifest_v2,
    expected_tr_count,
    load_walkthrough_script,
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


def test_validate_preds_manifest_alignment_ok():
    m = {"dom_snapshots": [{"t_idx": i} for i in range(10)]}
    r = validate_preds_manifest_alignment(10, m, tolerance=1)
    assert r["ok"] is True


def test_validate_preds_manifest_alignment_warn():
    m = {"dom_snapshots": [{"t_idx": i} for i in range(5)]}
    r = validate_preds_manifest_alignment(10, m, tolerance=1)
    assert r["ok"] is False
