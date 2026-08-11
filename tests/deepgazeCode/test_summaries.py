"""Unit tests for label-free density summaries."""

from __future__ import annotations

import numpy as np
import pytest

from scout_core.deepgaze_msdb.summaries import summarize_density


def test_summarize_density_peak_and_mass():
    dens = np.zeros((100, 100), dtype=np.float64)
    dens[10, 20] = 0.5
    dens[50, 50] = 0.5
    dens /= dens.sum()

    summary = summarize_density(dens)
    assert summary["height"] == 100
    assert summary["width"] == 100
    assert summary["sum"] == pytest.approx(1.0)
    assert summary["peak_y"] in (10, 50)
    assert summary["peak_x"] in (20, 50)
    assert 0.0 < summary["entropy"] < 20.0
    assert 0.0 < summary["normalized_entropy"] <= 1.0
    assert summary["top_1pct_mass"] >= 0.5
    assert "AUC" not in summary["metrics_note"] or "not computed" in summary["metrics_note"]


def test_summarize_rejects_negatives():
    dens = np.ones((4, 4), dtype=np.float64)
    dens[0, 0] = -0.1
    with pytest.raises(ValueError, match="negative"):
        summarize_density(dens)
