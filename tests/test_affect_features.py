"""Tests for affect feature extraction."""

from __future__ import annotations

import numpy as np

from scout_core.affect_features import build_affect_features, resolve_feature_spec
from scout_core.subcortical.atlas import N_SUBCORTICAL_VOXELS


def test_build_affect_features_shape():
    spec = resolve_feature_spec()
    cortical = np.random.randn(5, 20484).astype(np.float32)
    subcortical = np.random.randn(5, N_SUBCORTICAL_VOXELS).astype(np.float32)
    feats = build_affect_features(cortical, subcortical, spec=spec)
    assert feats.shape == (5, spec.n_features)


def test_build_affect_features_without_subcortical():
    spec = resolve_feature_spec()
    cortical = np.random.randn(3, 20484).astype(np.float32)
    feats = build_affect_features(cortical, None, spec=spec)
    assert feats.shape[0] == 3
    assert feats.shape[1] == spec.n_cortical_features + spec.n_subcortical_features
