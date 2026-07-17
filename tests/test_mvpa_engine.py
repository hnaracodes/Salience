"""Tests for MVPA decoder inference."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_bootstrap_model() -> None:
    model_path = PROJECT_ROOT / "scout_models" / "horikawa_ridge_v1" / "model.joblib"
    if model_path.is_file():
        return
    py = sys.executable
    subprocess.run(
        [py, str(PROJECT_ROOT / "scripts" / "horikawaCode" / "prepare_horikawa_tribev2.py"), "--demo"],
        check=True,
        cwd=PROJECT_ROOT,
    )
    subprocess.run(
        [py, str(PROJECT_ROOT / "scripts" / "horikawaCode" / "train_horikawa_ridge_decoder.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )


def test_predict_emotion_track():
    _ensure_bootstrap_model()
    from scout_core.mvpa_engine import predict_emotion_track
    from scout_core.subcortical.atlas import N_SUBCORTICAL_VOXELS

    cortical = np.random.randn(4, 20484).astype(np.float32)
    subcortical = np.random.randn(4, N_SUBCORTICAL_VOXELS).astype(np.float32)
    track = predict_emotion_track(cortical, subcortical, model_id="horikawa_ridge_v1")
    assert track["mode"] == "decoder"
    assert len(track["probabilities"]) == 4
    assert len(track["class_names"]) >= 1
