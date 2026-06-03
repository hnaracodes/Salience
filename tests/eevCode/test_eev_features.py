"""Tests for EEV FeatureContract v1."""

from __future__ import annotations

import numpy as np

from scout_core.eevCode.constants import FEATURE_NAMES_V1
from scout_core.eevCode.features import build_feature_matrix, pack_features_npz


def test_feature_matrix_shape_and_names():
    T, V = 10, 20484
    rng = np.random.default_rng(0)
    preds = rng.standard_normal((T, V), dtype=np.float32) * 0.1
    X = build_feature_matrix(preds)
    assert X.shape == (T, len(FEATURE_NAMES_V1))
    assert X.dtype == np.float32


def test_feature_matrix_deterministic():
    T, V = 8, 20484
    preds = np.linspace(0, 1, T * V, dtype=np.float32).reshape(T, V)
    X1 = build_feature_matrix(preds)
    X2 = build_feature_matrix(preds)
    np.testing.assert_allclose(X1, X2)


def test_pack_features_npz_keys():
    preds = np.zeros((5, 20484), dtype=np.float32)
    payload = pack_features_npz(video_id="testvid", preds=preds)
    assert "X_feat" in payload
    assert "network_amp" in payload
    assert payload["X_feat"].shape[1] == len(FEATURE_NAMES_V1)
    assert str(payload["video_id"]) == "testvid"
