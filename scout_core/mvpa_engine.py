"""MVPA / supervised emotion decoder inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scout_core.affect_features import AffectFeatureSpec, build_affect_features, resolve_feature_spec
from scout_core.horikawaCode.labels import label_map_to_names, load_label_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "scout_models"


def models_dir() -> Path:
    return DEFAULT_MODELS_DIR


def model_bundle_dir(model_id: str) -> Path:
    return models_dir() / model_id


def load_decoder_bundle(
    model_id: str,
    *,
    models_root: Path | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    root = models_root or models_dir()
    bundle_dir = root / model_id
    model_path = bundle_dir / "model.joblib"
    label_path = bundle_dir / "label_map.json"
    meta_path = bundle_dir / "meta.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Decoder model not found: {model_path}")
    pipeline = joblib.load(model_path)
    label_map = load_label_map(label_path) if label_path.is_file() else []
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return pipeline, label_map, meta


def _scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Map regression scores to [0,1] via sigmoid (multi-output)."""
    scores = np.asarray(scores, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)


def predict_emotion_track(
    cortical_preds: np.ndarray,
    subcortical_preds: np.ndarray | None,
    *,
    model_id: str,
    window_trs: int = 1,
    feature_spec_name: str | None = None,
    models_root: Path | None = None,
) -> dict[str, Any]:
    """Predict per-TR emotion scores/probabilities from TRIBE outputs."""
    pipeline, label_map, meta = load_decoder_bundle(model_id, models_root=models_root)
    spec_name = feature_spec_name or meta.get("feature_spec", "fused_schaefer400_subcortical_v1")
    spec = resolve_feature_spec(spec_name)
    if window_trs != spec.window_trs:
        spec = AffectFeatureSpec(
            name=spec.name,
            cortical_reducers=spec.cortical_reducers,
            include_subcortical=spec.include_subcortical,
            window_trs=window_trs,
            n_cortical_features=spec.n_cortical_features,
            n_subcortical_features=spec.n_subcortical_features,
        )

    X = build_affect_features(
        cortical_preds,
        subcortical_preds,
        spec=spec,
        window_trs=window_trs,
    )
    valid = np.all(np.isfinite(X), axis=1)
    class_names = label_map_to_names(label_map) if label_map else meta.get("class_names", [])
    n_classes = len(class_names) if class_names else int(meta.get("n_classes", 0))

    raw_scores = np.zeros((X.shape[0], max(n_classes, 1)), dtype=np.float32)
    if np.any(valid):
        pred = pipeline.predict(X[valid])
        pred_arr = np.asarray(pred, dtype=np.float32)
        if pred_arr.ndim == 1:
            pred_arr = pred_arr[:, np.newaxis]
        raw_scores[valid] = pred_arr[:, : raw_scores.shape[1]]

    probabilities = np.clip(raw_scores, 0.0, 1.0)
    if probabilities.max() > 1.0 or probabilities.min() < 0.0:
        probabilities = _scores_to_probabilities(raw_scores)

    return {
        "mode": "decoder",
        "schema_version": 2,
        "model_id": model_id,
        "class_names": class_names,
        "scores": raw_scores.tolist(),
        "probabilities": probabilities.tolist(),
        "meta": {
            "feature_spec": spec_name,
            "window_trs": window_trs,
            "cv_mean_r": meta.get("cv_mean_r"),
        },
        "diagnostic": {"template_cosine": None},
    }


def export_decoder_bundle(
    pipeline: Any,
    *,
    model_id: str,
    label_map: list[dict[str, Any]],
    meta: dict[str, Any],
    models_root: Path | None = None,
) -> Path:
    bundle_dir = (models_root or models_dir()) / model_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, bundle_dir / "model.joblib")
    (bundle_dir / "label_map.json").write_text(
        json.dumps({"classes": label_map}, indent=2),
        encoding="utf-8",
    )
    (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return bundle_dir
