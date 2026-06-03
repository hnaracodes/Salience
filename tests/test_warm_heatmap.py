"""Tests for warm heatmap RGBA export."""

from __future__ import annotations

import numpy as np

from scripts.export_ux_viewer import _warm_heatmap_rgba


def test_warm_heatmap_rgba_yellow_to_red():
    norm = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    rgba = _warm_heatmap_rgba(norm)
    assert rgba.shape == (1, 3, 4)
    # low attention: more yellow (high G relative to low end)
    assert rgba[0, 0, 1] > rgba[0, 2, 1]
    # high attention: more red (R high, G lower)
    assert rgba[0, 2, 0] > rgba[0, 0, 0]
    assert rgba[0, 0, 1] > rgba[0, 2, 1]
