"""Tests for engagement calibration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scout_core.engagement_calibration import spearman_rho, validate_session


def test_spearman_perfect():
    assert spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_validate_session_skips_without_bundle(tmp_path: Path):
    try:
        validate_session(tmp_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_validate_session_with_synthetic_bundle(tmp_path: Path):
    scores = [0.1, 0.5, 1.2, 0.8, -0.2]
    bundle = {
        "engagement_track": {"scores": scores, "source": "parcel_network_norms"},
    }
    (tmp_path / "analysis_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    manifest = {
        "tr_mapping": {"tr_duration_sec": 1.0},
        "dom_snapshots": [
            {"t_idx": i, "scrollY": 0 if i < 3 else 100, "elements": [{"visibility_ratio": 0.5}]}
            for i in range(len(scores))
        ],
    }
    (tmp_path / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_session(tmp_path)
    assert result["status"] == "ok"
    assert result["engagement_source"] == "parcel_network_norms"
