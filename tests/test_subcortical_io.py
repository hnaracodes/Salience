"""Tests for subcortical NPZ I/O and alignment."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from activation_store import save_subcortical_timeseries
from scout_core.subcortical.atlas import N_SUBCORTICAL_VOXELS, roi_means_from_voxels
from scout_core.subcortical.io import (
    load_subcortical_preds,
    save_subcortical_npz,
    validate_subcortical_alignment,
)
from scout_core.subcortical.tribev2_adapter import fake_subcortical_preds


def test_fake_subcortical_shape():
    arr = fake_subcortical_preds(5)
    assert arr.shape == (5, N_SUBCORTICAL_VOXELS)
    assert arr.dtype == np.float32


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)
        preds = np.random.randn(3, N_SUBCORTICAL_VOXELS).astype(np.float32)
        path = save_subcortical_npz(session_dir, preds)
        assert path.is_file()
        loaded = load_subcortical_preds(session_dir, require=True)
        assert loaded is not None
        np.testing.assert_allclose(loaded, preds)


def test_roi_means_shape():
    preds = np.random.randn(4, N_SUBCORTICAL_VOXELS).astype(np.float32)
    means = roi_means_from_voxels(preds)
    assert means.shape == (4, 8)


def test_validate_subcortical_alignment_ok():
    report = validate_subcortical_alignment(10, np.zeros((10, N_SUBCORTICAL_VOXELS)))
    assert report["ok"] is True


def test_validate_subcortical_alignment_mismatch():
    report = validate_subcortical_alignment(10, np.zeros((8, N_SUBCORTICAL_VOXELS)))
    assert report["ok"] is False


def test_save_subcortical_timeseries_bad_shape():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            save_subcortical_npz(Path(tmp), np.zeros((2, 100), dtype=np.float32))
