"""Tests for offline EEV inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from scout_core.eevCode.constants import EEV_TARGET_LABELS
from scout_core.eevCode.features import build_feature_matrix
from scout_core.eevCode.inference import predict_eev_scores


def test_predict_eev_scores_clipped(tmp_path):
    T, V = 12, 20484
    preds = np.random.default_rng(1).standard_normal((T, V)).astype(np.float32) * 0.05
    X = build_feature_matrix(preds)
    valid = np.all(np.isfinite(X), axis=1)
    Y = np.random.default_rng(2).random((valid.sum(), len(EEV_TARGET_LABELS))).astype(np.float32)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", MultiOutputRegressor(SVR(kernel="linear", C=1.0))),
        ]
    )
    pipe.fit(X[valid], Y)

    model_path = tmp_path / "eev_svr_test.joblib"
    manifest_path = tmp_path / "eev_svr_test.manifest.json"
    joblib.dump(pipe, model_path)
    manifest_path.write_text(json.dumps({"label_names": list(EEV_TARGET_LABELS)}), encoding="utf-8")

    scores = predict_eev_scores(preds, model_path, manifest_path=manifest_path)
    assert scores.shape == (T, len(EEV_TARGET_LABELS))
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
