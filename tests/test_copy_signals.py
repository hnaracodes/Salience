"""Unit tests for scout_core/copy_signals.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scout_core.copy_signals import (
    _score_clarity,
    _score_goal_fit,
    _score_urgency,
    build_copy_signals,
)


# ---------------------------------------------------------------------------
# Low-level scorer unit tests
# ---------------------------------------------------------------------------


def test_clarity_penalizes_wall_of_text():
    word_count = 600
    # Wall of text: avg sentence length 15 (perfect) but >500 words → penalty
    clarity = _score_clarity(word_count, avg_sent_len=15.0)
    assert clarity < 50, f"Expected clarity < 50 for 600-word wall of text; got {clarity}"


def test_clarity_short_text_baseline():
    # Under 10 words gets a sparse penalty
    clarity = _score_clarity(5, avg_sent_len=8.0)
    # Still a valid 0–100 value but penalised
    assert 0.0 <= clarity <= 100.0


def test_urgency_strong_cta():
    text = "get started free today"
    urgency = _score_urgency(text)
    assert urgency > 60, f"Expected urgency > 60 for strong CTA text; got {urgency}"


def test_urgency_zero_for_generic_text():
    text = "the page describes a product overview"
    urgency = _score_urgency(text)
    assert urgency < 30


def test_goal_fit_keyword_overlap():
    score = _score_goal_fit("ai chatbot for customer service", site_goal="AI chatbot")
    assert score > 0, "Expected positive goal_fit when keywords overlap"


def test_goal_fit_no_goal_returns_zero():
    score = _score_goal_fit("some page text here", site_goal=None)
    assert score == 0.0


def test_goal_fit_no_overlap():
    score = _score_goal_fit("completely unrelated words about cooking", site_goal="AI chatbot")
    assert score == 0.0


# ---------------------------------------------------------------------------
# Integration-level tests using a temporary session directory
# ---------------------------------------------------------------------------


def _make_session(tmp_path: Path, snapshots: list[dict]) -> tuple[Path, str]:
    """Write a minimal session_manifest.json and return (session_dir, session_id)."""
    session_id = "test-session-001"
    session_dir = tmp_path / session_id
    session_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "session_id": session_id,
        "dom_snapshots": snapshots,
    }
    (session_dir / "session_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return session_dir, session_id


def test_missing_copy_signals_returns_empty_list(tmp_path):
    """Session with no dom_snapshots returns an empty list gracefully."""
    session_id = "empty-session"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manifest = {"schema_version": 2, "session_id": session_id, "dom_snapshots": []}
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest))

    with patch("scout_core.copy_signals.SESSIONS_DIR", tmp_path):
        result = build_copy_signals(session_id)

    assert result == []


def test_missing_manifest_returns_empty_list(tmp_path):
    """Session directory with no manifest returns an empty list without crashing."""
    session_id = "no-manifest"
    (tmp_path / session_id).mkdir()

    with patch("scout_core.copy_signals.SESSIONS_DIR", tmp_path):
        result = build_copy_signals(session_id)

    assert result == []


def test_write_and_reload(tmp_path):
    """build_copy_signals writes copy_signals.json; reloading it matches the return value."""
    elements = [
        {
            "tag": "H1",
            "text": "Welcome to our platform",
            "is_intersecting_viewport": True,
        },
        {
            "tag": "P",
            "text": "Start your free trial today and get started instantly.",
            "is_intersecting_viewport": True,
        },
    ]
    snapshot = {"t_idx": 0, "url": "https://example.com/", "elements": elements}
    session_dir, session_id = _make_session(tmp_path, [snapshot])

    with patch("scout_core.copy_signals.SESSIONS_DIR", tmp_path):
        result = build_copy_signals(session_id, site_goal="SaaS platform")

    assert len(result) > 0

    # Verify the file was written and its contents match.
    written = json.loads((session_dir / "copy_signals.json").read_text())
    assert written == result


def test_urgency_flag_strong_cta(tmp_path):
    """A section with strong CTA language gets the strong_cta flag."""
    elements = [
        {"tag": "BUTTON", "text": "Get started free today now", "is_intersecting_viewport": True},
        {"tag": "A", "text": "Start your trial now", "is_intersecting_viewport": True},
        {"tag": "P", "text": "Limited time offer. Subscribe instantly.", "is_intersecting_viewport": True},
    ]
    snapshot = {"t_idx": 0, "url": "https://example.com/", "elements": elements}
    _, session_id = _make_session(tmp_path, [snapshot])

    with patch("scout_core.copy_signals.SESSIONS_DIR", tmp_path):
        result = build_copy_signals(session_id)

    assert result
    flagged = any("strong_cta" in row.get("flags", []) for row in result)
    assert flagged, "Expected at least one section to have the strong_cta flag"


def test_wall_of_text_flag(tmp_path):
    """A section with 600+ words gets the wall_of_text flag."""
    long_text = " ".join(["word"] * 600)
    elements = [{"tag": "P", "text": long_text, "is_intersecting_viewport": True}]
    snapshot = {"t_idx": 0, "url": "https://example.com/", "elements": elements}
    _, session_id = _make_session(tmp_path, [snapshot])

    with patch("scout_core.copy_signals.SESSIONS_DIR", tmp_path):
        result = build_copy_signals(session_id)

    assert result
    assert any("wall_of_text" in row.get("flags", []) for row in result)
