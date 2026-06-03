"""Tests for manifest snapshot padding vs video duration."""

from __future__ import annotations

from pathlib import Path

import pytest

from scout_core.session_align import pad_manifest_snapshots_to_video_duration


def test_pad_manifest_forward_fills_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    manifest = {
        "dom_snapshots": [
            {"t_idx": 0, "pts_sec": 0.0, "scrollY": 0, "elements": []},
            {"t_idx": 1, "pts_sec": 1.0, "scrollY": 100, "elements": []},
        ],
        "video": {"path": "walkthrough.webm", "duration_sec": 5.0},
        "capture": {"tr_duration_sec": 1.0, "fps": 1.0},
    }

    monkeypatch.setattr(
        "scout_core.session_align.probe_video_duration_sec",
        lambda _path: 5.0,
    )

    out = pad_manifest_snapshots_to_video_duration(manifest, session_dir)
    assert len(out["dom_snapshots"]) == 5
    assert out["dom_snapshots"][-1]["t_idx"] == 4
    assert out["dom_snapshots"][-1].get("synthetic_pad") is True
