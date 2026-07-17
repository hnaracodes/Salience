"""Tests for scout_core/neural_moments.py and parcel-based engagement."""

from __future__ import annotations

import numpy as np
import pytest

from scout_core.neural_moments import (
    build_neural_moments_by_t,
    engagement_scores_from_network_z,
    neural_moment_strength,
)


def test_engagement_from_network_z_salventattn_minus_default():
    names = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    z = np.zeros((5, 7), dtype=np.float32)
    van_i = names.index("SalVentAttn")
    dmn_i = names.index("Default")
    z[:, van_i] = 2.0
    z[:, dmn_i] = -1.0
    result = engagement_scores_from_network_z(z, names)
    scores = np.array(result["scores"], dtype=np.float32)
    assert result["source"] == "parcel_network_norms"
    assert np.allclose(scores, 3.0)


def test_neural_moment_strength():
    moment = {"top_networks": [{"network": "SalVentAttn", "z": 2.4}]}
    assert neural_moment_strength(moment) == pytest.approx(2.4)


def test_build_neural_moments_by_t():
    T, P, N = 4, 3, 2
    z_p = np.random.default_rng(0).standard_normal((T, P)).astype(np.float32)
    z_n = np.random.default_rng(1).standard_normal((T, N)).astype(np.float32)
    parcel_ids = np.array([10, 20, 30], dtype=np.int64)
    labels = {10: "p10", 20: "p20", 30: "p30"}
    names = ["SalVentAttn", "Default"]
    out = build_neural_moments_by_t(
        z_p, parcel_ids, labels, z_n, names, t_indices=[1, 2],
    )
    assert set(out.keys()) == {"1", "2"}
    assert "top_parcels" in out["1"]
    assert "top_networks" in out["1"]
