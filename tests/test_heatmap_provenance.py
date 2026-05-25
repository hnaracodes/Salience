"""Tests for heatmap extraction provenance: manifest merge, placeholder detection, overwrite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scout_core.heatmap_extract import read_heatmaps_manifest, write_heatmaps_manifest  # noqa: E402


# ---------------------------------------------------------------------------
# read_heatmaps_manifest
# ---------------------------------------------------------------------------

def test_read_heatmaps_manifest_returns_empty_when_missing(tmp_path):
    result = read_heatmaps_manifest(tmp_path / "nonexistent")
    assert result == {}


def test_read_heatmaps_manifest_returns_empty_for_corrupt_json(tmp_path):
    d = tmp_path / "heatmaps"
    d.mkdir()
    (d / "manifest.json").write_text("{bad json", encoding="utf-8")
    result = read_heatmaps_manifest(d)
    assert result == {}


def test_read_heatmaps_manifest_indexes_by_t_idx(tmp_path):
    d = tmp_path / "heatmaps"
    d.mkdir()
    entries = [
        {"t_idx": 0, "heatmap_path": "t_0.npy", "source": "uniform_placeholder", "placeholder": True},
        {"t_idx": 5, "heatmap_path": "t_5.npy", "source": "modal", "placeholder": False},
    ]
    (d / "manifest.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    result = read_heatmaps_manifest(d)
    assert 0 in result
    assert 5 in result
    assert result[0]["placeholder"] is True
    assert result[5]["source"] == "modal"


# ---------------------------------------------------------------------------
# write_heatmaps_manifest — merge mode
# ---------------------------------------------------------------------------

def test_write_heatmaps_manifest_overwrites_by_default(tmp_path):
    d = tmp_path / "heatmaps"
    d.mkdir()
    old = [{"t_idx": 0, "source": "old"}]
    write_heatmaps_manifest(d, old)
    new = [{"t_idx": 1, "source": "new"}]
    write_heatmaps_manifest(d, new, merge=False)
    result = read_heatmaps_manifest(d)
    assert 0 not in result
    assert 1 in result


def test_write_heatmaps_manifest_merge_preserves_existing(tmp_path):
    d = tmp_path / "heatmaps"
    d.mkdir()
    old = [{"t_idx": 0, "source": "uniform_placeholder", "placeholder": True}]
    write_heatmaps_manifest(d, old)
    new_entry = [{"t_idx": 5, "source": "modal", "placeholder": False}]
    write_heatmaps_manifest(d, new_entry, merge=True)
    result = read_heatmaps_manifest(d)
    assert 0 in result
    assert 5 in result
    assert result[0]["placeholder"] is True
    assert result[5]["source"] == "modal"


def test_write_heatmaps_manifest_merge_replaces_same_t_idx(tmp_path):
    d = tmp_path / "heatmaps"
    d.mkdir()
    old = [{"t_idx": 5, "source": "uniform_placeholder", "placeholder": True}]
    write_heatmaps_manifest(d, old)
    replacement = [{"t_idx": 5, "source": "modal", "placeholder": False, "sha256": "abc123"}]
    write_heatmaps_manifest(d, replacement, merge=True)
    result = read_heatmaps_manifest(d)
    assert result[5]["source"] == "modal"
    assert result[5]["placeholder"] is False
    assert result[5]["sha256"] == "abc123"


def test_write_heatmaps_manifest_merge_sorts_by_t_idx(tmp_path):
    d = tmp_path / "heatmaps"
    d.mkdir()
    entries_a = [{"t_idx": 10, "source": "modal"}]
    write_heatmaps_manifest(d, entries_a)
    entries_b = [{"t_idx": 2, "source": "modal"}]
    write_heatmaps_manifest(d, entries_b, merge=True)
    raw = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    t_indices = [e["t_idx"] for e in raw["entries"]]
    assert t_indices == sorted(t_indices)


# ---------------------------------------------------------------------------
# Placeholder detection semantics
# ---------------------------------------------------------------------------

def test_placeholder_detected_via_placeholder_field(tmp_path):
    d = tmp_path / "heatmaps"
    d.mkdir()
    entries = [{"t_idx": 0, "placeholder": True, "source": "anything"}]
    write_heatmaps_manifest(d, entries)
    m = read_heatmaps_manifest(d)
    assert m[0]["placeholder"] is True


def test_real_heatmap_not_flagged_as_placeholder(tmp_path):
    d = tmp_path / "heatmaps"
    d.mkdir()
    entries = [{"t_idx": 0, "source": "modal", "placeholder": False}]
    write_heatmaps_manifest(d, entries)
    m = read_heatmaps_manifest(d)
    assert m[0]["placeholder"] is False
