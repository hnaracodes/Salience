from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_prep.hcp_cifti_network import (
    aggregate_run_features,
    dtseries_array_to_network_features,
    load_parcellation_artifact,
)
from data_prep.viability_partition import NET_COLUMN_PREFIX
from scout_core.constants import YEO7_NAMES


@pytest.fixture
def parc_artifact() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs/hcp_fslr_schaefer400_yeo7.npz"
    return load_parcellation_artifact(path)


def test_dtseries_to_network_features_shape(parc_artifact):
    ts = np.random.randn(50, parc_artifact["brain_axis_size"]).astype(np.float32)
    feats = dtseries_array_to_network_features(ts, parc_artifact)
    assert len(feats) >= 1
    for name in YEO7_NAMES:
        key = f"{NET_COLUMN_PREFIX}{name}"
        if key in feats:
            assert np.isfinite(feats[key])


def test_aggregate_run_features_means_runs(parc_artifact):
    ts = np.random.randn(30, parc_artifact["brain_axis_size"]).astype(np.float32)
    f1 = dtseries_array_to_network_features(ts, parc_artifact)
    f2 = dtseries_array_to_network_features(ts * 2, parc_artifact)
    agg = aggregate_run_features([f1, f2])
    for k in f1:
        assert agg[k] == pytest.approx((f1[k] + f2[k]) / 2, rel=1e-5)


def test_brain_axis_mismatch_raises(parc_artifact):
    ts = np.random.randn(10, int(parc_artifact["brain_axis_size"]) + 1).astype(np.float32)
    with pytest.raises(ValueError, match="brain axis"):
        dtseries_array_to_network_features(ts, parc_artifact)
