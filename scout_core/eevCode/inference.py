"""Offline EEV SVR inference (research track only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scout_core.eevCode.constants import EEV_SCHEMA_VERSION, EEV_TARGET_LABELS
from scout_core.eevCode.features import build_feature_matrix


def load_model_bundle(model_path: Path, manifest_path: Path | None = None) -> tuple[Any, dict[str, Any]]:
    manifest_path = manifest_path or model_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        manifest_path = model_path.parent / (model_path.stem.replace(".joblib", "") + ".manifest.json")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(model_path)
    return pipeline, manifest


def predict_eev_scores(
    preds: np.ndarray,
    model_path: Path,
    *,
    manifest_path: Path | None = None,
) -> np.ndarray:
    """Research-only. Returns (T, 5) clipped to [0, 1]."""
    pipeline, _manifest = load_model_bundle(model_path, manifest_path)
    X = build_feature_matrix(preds)
    valid = np.all(np.isfinite(X), axis=1)
    Y_hat = np.zeros((X.shape[0], len(EEV_TARGET_LABELS)), dtype=np.float32)
    if np.any(valid):
        Y_hat[valid] = pipeline.predict(X[valid]).astype(np.float32)
    return np.clip(Y_hat, 0.0, 1.0)


def predict_eev_track(
    preds: np.ndarray,
    model_path: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    scores = predict_eev_scores(preds, model_path, manifest_path=manifest_path)
    _, manifest = load_model_bundle(model_path, manifest_path)
    label_names = manifest.get("label_names") or list(EEV_TARGET_LABELS)
    return {
        "mode": "eev_svr",
        "schema_version": EEV_SCHEMA_VERSION,
        "label_names": list(label_names),
        "scores": scores.tolist(),
        "model_path": str(model_path),
    }
