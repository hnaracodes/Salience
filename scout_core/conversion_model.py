"""Supervised conversion intent model from session bundle features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "scout_data" / "models"


def extract_session_features(bundle: dict[str, Any]) -> np.ndarray:
    """Fixed-length feature vector from analysis_bundle."""
    ms = bundle.get("marketing_scores") or {}
    et = bundle.get("engagement_track") or {}
    at = bundle.get("activation_track") or {}
    eng = [float(s) for s in (et.get("scores") or []) if s is not None]
    act = [float(s) for s in (at.get("raw_scores") or []) if s is not None]
    sections = bundle.get("section_report") or []

    feats = [
        float(ms.get("overall_score") or 0),
        float(ms.get("comparison_score") or 0),
        float(np.mean(eng)) if eng else 0.0,
        float(np.std(eng)) if len(eng) > 1 else 0.0,
        float(np.mean(act)) if act else 0.0,
        float(len(sections)),
    ]
    top_combined = []
    for sec in sections:
        tops = sec.get("top_elements") or []
        if tops:
            top_combined.append(float(tops[0].get("combined_score") or 0))
    feats.append(float(np.mean(top_combined)) if top_combined else 0.0)
    return np.asarray(feats, dtype=np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class LogisticConversionModel:
    """Minimal logistic regression for conversion intent."""

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.calibration_id: str = "untrained"

    def fit(self, X: np.ndarray, y: np.ndarray, *, calibration_id: str = "trained_v1") -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        lr = 0.05
        for _ in range(500):
            z = X @ w + b
            p = _sigmoid(z)
            grad_w = (X.T @ (p - y)) / max(n, 1)
            grad_b = float(np.mean(p - y))
            w -= lr * grad_w
            b -= lr * grad_b
        self.weights = w
        self.bias = b
        self.calibration_id = calibration_id

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.full(X.shape[0], 0.5, dtype=np.float64)
        z = X @ self.weights + self.bias
        return _sigmoid(z)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights if self.weights is not None else np.array([]),
            bias=self.bias,
            calibration_id=np.array(self.calibration_id),
        )

    @classmethod
    def load(cls, path: Path) -> LogisticConversionModel:
        data = np.load(path, allow_pickle=True)
        m = cls()
        w = data.get("weights")
        m.weights = np.asarray(w, dtype=np.float64) if w is not None and w.size else None
        m.bias = float(data.get("bias", 0.0))
        cid = data.get("calibration_id")
        m.calibration_id = str(cid) if cid is not None else "loaded"
        return m


def score_bundle(bundle: dict[str, Any], model: LogisticConversionModel) -> dict[str, Any]:
    x = extract_session_features(bundle).reshape(1, -1)
    prob = float(model.predict_proba(x)[0])
    return {
        "probability": round(prob, 4),
        "calibration_id": model.calibration_id,
        "disclaimer": (
            "Predicted relative conversion intent from neural UX features; "
            "not a guarantee of revenue lift."
        ),
    }


def load_training_labels(labels_path: Path) -> dict[str, float]:
    """JSON: { session_id: 0|1 } or { labels: { session_id: 0|1 } }."""
    if not labels_path.is_file():
        return {}
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(data.get("labels"), dict):
        return {str(k): float(v) for k, v in data["labels"].items()}
    return {str(k): float(v) for k, v in data.items() if k not in ("description", "labels")}
