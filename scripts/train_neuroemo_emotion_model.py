#!/usr/bin/env python3
"""Train a supervised NeuroEmo emotion classifier for TribeV2-style outputs.

Input:
    scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz

The prepared NPZ is produced by scripts/prepare_neuroemo_tribev2.py and contains
features shaped like TribeV2 cortical predictions:

    X: (N, 20484) or (N, window_trs, 20484)
    y: (N,)
    subject: (N,)
    labels: class names ordered by y id

This script trains a scikit-learn classifier with subject-held-out cross
validation, then fits a final model on all samples and writes a joblib artifact.

Local usage:
    python scripts/train_neuroemo_emotion_model.py
    python scripts/train_neuroemo_emotion_model.py --model-type logistic_saga --cv-splits 5

Modal usage for heavier runs:
    modal run scripts/train_neuroemo_emotion_model.py::train_modal
    modal run scripts/train_neuroemo_emotion_model.py::train_modal --model-type logistic_saga

Output:
    scout_data/neuroemo/models/neuroemo_emotion_model.joblib
    scout_data/neuroemo/models/neuroemo_emotion_metrics.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scout_core.parcellation import CANONICAL_VERTEX_ORDER, normalize_vertex_order
from scout_core.roi_features import (
    RoiFeatureSpec,
    load_roi_feature_spec,
    reduce_vertices_to_rois,
    summarise_roi_feature_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_NPZ = PROJECT_ROOT / "scout_data" / "neuroemo" / "tribev2_surface" / "neuroemo_tribev2_train.npz"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "scout_data" / "neuroemo" / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "neuroemo_emotion_model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_MODEL_DIR / "neuroemo_emotion_metrics.json"
DEFAULT_VERTEX_CSV = PROJECT_ROOT / "configs" / "vertex_regions.csv"
DEFAULT_ATLAS_MANIFEST = PROJECT_ROOT / "configs" / "parcellation_manifest.yaml"


@dataclass(frozen=True)
class TrainConfig:
    train_npz: str = str(DEFAULT_TRAIN_NPZ)
    output_model: str = str(DEFAULT_MODEL_PATH)
    metrics_json: str = str(DEFAULT_METRICS_PATH)
    model_type: str = "sgd_logistic"
    cv_splits: int = 5
    max_iter: int = 3000
    alpha: float = 0.0001
    c_value: float = 1.0
    random_state: int = 13
    max_samples: int = 0
    no_cv: bool = False
    n_jobs: int = -1
    scale: bool = True
    feature_mode: str = "roi"
    vertex_csv: str = str(DEFAULT_VERTEX_CSV)
    roi_reducers: str = "mean"
    atlas_manifest: str = str(DEFAULT_ATLAS_MANIFEST)
    temporal_window_trs: int = 1
    temporal_stride_trs: int = 1
    temporal_reducer: str = "mean"
    temporal_contiguity: str = "contiguous"
    temporal_allow_class_drop: bool = False
    temporal_allow_rewindow: bool = False
    exclude_labels: str = "neutral"


@dataclass(frozen=True)
class TrainingData:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    labels: np.ndarray
    input_shape: tuple[int, ...]
    source_shape: tuple[int, ...]
    flattened_shape: tuple[int, ...]
    feature_mode: str
    roi_spec: RoiFeatureSpec | None
    metadata: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _npz_scalar(raw: Any, key: str, default: Any = None) -> Any:
    if key not in raw.files:
        return default
    value = raw[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            item = value.reshape(-1)[0]
            return item.item() if isinstance(item, np.generic) else item
        return value.tolist()
    return value


def _reduce_temporal_window(window: np.ndarray, reducer: str) -> np.ndarray:
    if reducer == "mean":
        return window.mean(axis=0)
    if reducer == "median":
        return np.median(window, axis=0)
    if reducer == "last":
        return window[-1]
    raise ValueError(f"Unknown temporal reducer: {reducer}")


def _apply_temporal_windows(
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    t_idx: np.ndarray,
    time_s: np.ndarray,
    labels: np.ndarray,
    cfg: TrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Aggregate TR-level rows into same-subject, same-label temporal windows."""
    window_trs = int(cfg.temporal_window_trs)
    stride_trs = int(cfg.temporal_stride_trs)
    if window_trs < 1:
        raise ValueError("--temporal-window-trs must be >= 1")
    if stride_trs < 1:
        raise ValueError("--temporal-stride-trs must be >= 1")
    if window_trs == 1:
        return (
            X,
            y,
            subject,
            t_idx,
            time_s,
            {
                "enabled": False,
                "window_trs": 1,
                "stride_trs": stride_trs,
                "reducer": cfg.temporal_reducer,
                "contiguity": cfg.temporal_contiguity,
                "input_samples": int(X.shape[0]),
                "output_samples": int(X.shape[0]),
            },
        )

    if cfg.temporal_contiguity not in ("contiguous", "same_label"):
        raise ValueError("--temporal-contiguity must be 'contiguous' or 'same_label'")

    samples: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[str] = []
    out_t_idx: list[int] = []
    out_time_s: list[float] = []
    window_rows: list[dict[str, Any]] = []

    for subj in sorted(set(subject.tolist())):
        subj_idx = np.flatnonzero(subject == subj)
        for class_id in sorted(np.unique(y[subj_idx]).tolist()):
            group_idx = subj_idx[y[subj_idx] == int(class_id)]
            group_idx = group_idx[np.argsort(t_idx[group_idx], kind="stable")]
            if group_idx.size < window_trs:
                continue
            for start in range(0, group_idx.size - window_trs + 1, stride_trs):
                win_idx = group_idx[start : start + window_trs]
                win_t = t_idx[win_idx]
                if cfg.temporal_contiguity == "contiguous" and not np.all(np.diff(win_t) == 1):
                    continue
                sample = _reduce_temporal_window(X[win_idx], cfg.temporal_reducer)
                samples.append(sample.astype(np.float32, copy=False))
                ys.append(int(class_id))
                subjects.append(str(subj))
                out_t_idx.append(int(win_t[-1]))
                out_time_s.append(float(time_s[win_idx][-1]))
                window_rows.append(
                    {
                        "subject": str(subj),
                        "label": str(labels[int(class_id)]) if int(class_id) < len(labels) else str(class_id),
                        "label_id": int(class_id),
                        "start_t_idx": int(win_t[0]),
                        "end_t_idx": int(win_t[-1]),
                    }
                )

    if not samples:
        raise ValueError(
            "Temporal aggregation produced no samples. Try a smaller --temporal-window-trs, "
            "--temporal-contiguity same_label, or verify train NPZ has t_idx/time_s."
        )

    X_windowed = np.stack(samples).astype(np.float32)
    y_windowed = np.asarray(ys, dtype=np.int64)
    subject_windowed = np.asarray(subjects)
    t_idx_windowed = np.asarray(out_t_idx, dtype=np.int64)
    time_s_windowed = np.asarray(out_time_s, dtype=np.float32)

    expected_classes = set(range(len(labels)))
    found_classes = set(y_windowed.tolist())
    dropped_classes = sorted(expected_classes - found_classes)
    if dropped_classes and not cfg.temporal_allow_class_drop:
        dropped_names = [
            str(labels[class_id]) if class_id < len(labels) else f"class_{class_id}"
            for class_id in dropped_classes
        ]
        raise ValueError(
            "Temporal aggregation dropped class(es) "
            f"{dropped_names}. This often happens when labels are sparse, such as the current "
            "balanced neutral samples. Use --temporal-contiguity same_label, reduce "
            "--temporal-window-trs, or pass --temporal-allow-class-drop for diagnostics only."
        )

    summary = {
        "enabled": True,
        "window_trs": window_trs,
        "stride_trs": stride_trs,
        "reducer": cfg.temporal_reducer,
        "contiguity": cfg.temporal_contiguity,
        "input_samples": int(X.shape[0]),
        "output_samples": int(X_windowed.shape[0]),
        "dropped_class_ids": dropped_classes,
        "dropped_class_names": [
            str(labels[class_id]) if class_id < len(labels) else f"class_{class_id}"
            for class_id in dropped_classes
        ],
        "class_counts": _class_counts(y_windowed, labels),
        "example_windows": window_rows[:10],
    }
    return X_windowed, y_windowed, subject_windowed, t_idx_windowed, time_s_windowed, summary


def _parse_excluded_labels(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _filter_and_remap_labels(
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    t_idx: np.ndarray,
    time_s: np.ndarray,
    labels: np.ndarray,
    *,
    exclude_labels: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    excluded = _parse_excluded_labels(exclude_labels)
    if not excluded:
        return (
            X,
            y,
            subject,
            t_idx,
            time_s,
            labels,
            {
                "excluded_labels": [],
                "input_samples": int(X.shape[0]),
                "output_samples": int(X.shape[0]),
                "label_id_map": {int(i): int(i) for i in range(len(labels))},
            },
        )

    label_names = labels.astype(str)
    keep_label_ids = [
        int(i)
        for i, name in enumerate(label_names.tolist())
        if str(name) not in excluded
    ]
    dropped_label_ids = [
        int(i)
        for i, name in enumerate(label_names.tolist())
        if str(name) in excluded
    ]
    if not keep_label_ids:
        raise ValueError(f"--exclude-labels removed every class: {sorted(excluded)}")

    keep_mask = np.isin(y, np.asarray(keep_label_ids, dtype=np.int64))
    X_keep = X[keep_mask]
    y_keep_old = y[keep_mask]
    subject_keep = subject[keep_mask]
    t_idx_keep = t_idx[keep_mask]
    time_s_keep = time_s[keep_mask]

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_label_ids)}
    y_keep = np.asarray([old_to_new[int(class_id)] for class_id in y_keep_old], dtype=np.int64)
    labels_keep = label_names[keep_label_ids]

    return (
        X_keep,
        y_keep,
        subject_keep,
        t_idx_keep,
        time_s_keep,
        labels_keep,
        {
            "excluded_labels": sorted(excluded),
            "dropped_label_ids": dropped_label_ids,
            "input_samples": int(X.shape[0]),
            "output_samples": int(X_keep.shape[0]),
            "label_id_map": {int(old): int(new) for old, new in old_to_new.items()},
            "remaining_labels": labels_keep.tolist(),
        },
    )


def _load_training_data(path: Path, cfg: TrainConfig) -> TrainingData:
    if not path.is_file():
        raise FileNotFoundError(
            f"Training NPZ not found: {path}. Run scripts/prepare_neuroemo_tribev2.py first."
        )

    raw = np.load(path, allow_pickle=False)
    required = {"X", "y", "subject", "labels"}
    missing = sorted(required - set(raw.files))
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")

    X_raw = np.asarray(raw["X"], dtype=np.float32)
    y = np.asarray(raw["y"], dtype=np.int64)
    subject = np.asarray(raw["subject"]).astype(str)
    labels = np.asarray(raw["labels"]).astype(str)
    source_mesh = str(_npz_scalar(raw, "mesh", "fsaverage5"))
    source_vertex_order = normalize_vertex_order(_npz_scalar(raw, "vertex_order", CANONICAL_VERTEX_ORDER))
    source_window_trs = int(_npz_scalar(raw, "source_window_trs", X_raw.shape[1] if X_raw.ndim == 3 else 1))
    source_sample_axis = str(
        _npz_scalar(raw, "source_sample_axis", "prep_window" if X_raw.ndim == 3 else "tr")
    )
    prep_metadata: dict[str, Any] = {}
    prep_metadata_json = _npz_scalar(raw, "prep_metadata_json", "")
    if isinstance(prep_metadata_json, str) and prep_metadata_json.strip():
        try:
            loaded = json.loads(prep_metadata_json)
            if isinstance(loaded, dict):
                prep_metadata = loaded
        except json.JSONDecodeError:
            prep_metadata = {"raw_prep_metadata_json": prep_metadata_json}

    if source_mesh != "fsaverage5":
        raise ValueError(f"Training NPZ must be in fsaverage5 space, got mesh={source_mesh!r}")
    if source_vertex_order not in (None, CANONICAL_VERTEX_ORDER):
        raise ValueError(
            f"Training NPZ vertex order {source_vertex_order!r} is not supported; "
            f"expected {CANONICAL_VERTEX_ORDER!r}"
        )

    has_t_idx = "t_idx" in raw.files
    has_time_s = "time_s" in raw.files
    if cfg.temporal_window_trs > 1 and not has_t_idx:
        raise ValueError(
            "Temporal aggregation requires source t_idx metadata. Rebuild the NeuroEmo NPZ "
            "with t_idx/time_s arrays or train without --temporal-window-trs > 1."
        )
    if cfg.temporal_window_trs > 1 and not has_time_s:
        raise ValueError(
            "Temporal aggregation requires source time_s metadata. Rebuild the NeuroEmo NPZ "
            "with t_idx/time_s arrays or train without --temporal-window-trs > 1."
        )
    t_idx = (
        np.asarray(raw["t_idx"], dtype=np.int64)
        if has_t_idx
        else np.arange(y.shape[0], dtype=np.int64)
    )
    time_s = (
        np.asarray(raw["time_s"], dtype=np.float32)
        if has_time_s
        else t_idx.astype(np.float32)
    )

    if X_raw.shape[0] != y.shape[0] or y.shape[0] != subject.shape[0] or y.shape[0] != t_idx.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X={X_raw.shape}, y={y.shape}, subject={subject.shape}, t_idx={t_idx.shape}"
        )
    if X_raw.ndim not in (2, 3):
        raise ValueError(f"Expected X to be 2D or 3D, got shape {X_raw.shape}")
    if X_raw.ndim == 3 and source_window_trs != int(X_raw.shape[1]):
        raise ValueError(
            f"Training NPZ source_window_trs={source_window_trs} does not match X shape {X_raw.shape}"
        )
    if X_raw.ndim == 2 and source_window_trs != 1:
        raise ValueError(
            f"Training NPZ source_window_trs={source_window_trs} but X is 2D; expected 1"
        )
    if X_raw.ndim == 3 and cfg.temporal_window_trs > 1 and not cfg.temporal_allow_rewindow:
        raise ValueError(
            "Training NPZ is already windowed, but a second training-time temporal window was requested. "
            "Use --temporal-allow-rewindow only for diagnostics."
        )

    input_shape = tuple(int(x) for x in X_raw.shape[1:])

    valid = y >= 0
    if not np.all(valid):
        X_raw = X_raw[valid]
        y = y[valid]
        subject = subject[valid]
        t_idx = t_idx[valid]
        time_s = time_s[valid]

    X_raw, y, subject, t_idx, time_s, labels, label_filter_summary = _filter_and_remap_labels(
        X_raw,
        y,
        subject,
        t_idx,
        time_s,
        labels,
        exclude_labels=cfg.exclude_labels,
    )

    X_raw, y, subject, t_idx, time_s, temporal_summary = _apply_temporal_windows(
        X_raw,
        y,
        subject,
        t_idx,
        time_s,
        labels,
        cfg,
    )

    if cfg.max_samples and X_raw.shape[0] > cfg.max_samples:
        rng = np.random.default_rng(cfg.random_state)
        chosen: list[int] = []
        for class_id in np.unique(y):
            idx = np.flatnonzero(y == class_id)
            n_class = max(1, int(round(cfg.max_samples * len(idx) / len(y))))
            n_class = min(n_class, len(idx))
            chosen.extend(rng.choice(idx, size=n_class, replace=False).tolist())
        if len(chosen) > cfg.max_samples:
            chosen = rng.choice(np.asarray(chosen), size=cfg.max_samples, replace=False).tolist()
        chosen_arr = np.asarray(sorted(chosen), dtype=np.int64)
        X_raw = X_raw[chosen_arr]
        y = y[chosen_arr]
        subject = subject[chosen_arr]
        t_idx = t_idx[chosen_arr]
        time_s = time_s[chosen_arr]

    if X_raw.shape[0] == 0:
        raise ValueError("No labeled samples found in training NPZ.")

    roi_spec: RoiFeatureSpec | None = None
    if cfg.feature_mode == "vertices":
        X_features = X_raw
    elif cfg.feature_mode == "roi":
        vertex_csv_path = Path(cfg.vertex_csv).expanduser()
        if not vertex_csv_path.is_absolute():
            vertex_csv_path = PROJECT_ROOT / vertex_csv_path
        atlas_manifest_path = Path(cfg.atlas_manifest).expanduser() if cfg.atlas_manifest else None
        if atlas_manifest_path is not None and not atlas_manifest_path.is_absolute():
            atlas_manifest_path = PROJECT_ROOT / atlas_manifest_path
        roi_spec = load_roi_feature_spec(
            vertex_csv_path,
            n_vertices=int(X_raw.shape[-1]),
            reducers=cfg.roi_reducers,
            manifest_path=atlas_manifest_path,
        )
        if roi_spec.mesh not in (None, source_mesh):
            raise ValueError(
                f"ROI manifest mesh {roi_spec.mesh!r} does not match training NPZ mesh {source_mesh!r}"
            )
        if roi_spec.vertex_order not in (None, source_vertex_order, CANONICAL_VERTEX_ORDER):
            raise ValueError(
                f"ROI manifest vertex order {roi_spec.vertex_order!r} does not match "
                f"training NPZ vertex order {source_vertex_order!r}"
            )
        X_features = reduce_vertices_to_rois(X_raw, roi_spec)
    else:
        raise ValueError(f"Unknown feature_mode: {cfg.feature_mode}")

    source_shape = tuple(int(x) for x in X_features.shape[1:])
    X = X_features.reshape(X_features.shape[0], -1).astype(np.float32)

    metadata = {
        "train_npz": str(path),
        "npz_arrays": {name: list(raw[name].shape) for name in raw.files},
        "dataset_id": str(raw["dataset_id"]) if "dataset_id" in raw.files else None,
        "snapshot_version": str(raw["snapshot_version"]) if "snapshot_version" in raw.files else None,
        "input_feature_shape": list(input_shape),
        "source_contract": {
            "mesh": source_mesh,
            "vertex_order": source_vertex_order or CANONICAL_VERTEX_ORDER,
            "source_window_trs": source_window_trs,
            "source_sample_axis": source_sample_axis,
            "input_ndim": int(X_raw.ndim),
            "prep_metadata": prep_metadata,
        },
        "label_filter": label_filter_summary,
        "temporal_aggregation": temporal_summary,
    }
    return TrainingData(
        X=X,
        y=y,
        subject=subject,
        labels=labels,
        input_shape=input_shape,
        source_shape=source_shape,
        flattened_shape=tuple(int(x) for x in X.shape[1:]),
        feature_mode=cfg.feature_mode,
        roi_spec=roi_spec,
        metadata=metadata,
    )


def _make_estimator(cfg: TrainConfig):
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    steps: list[tuple[str, Any]] = []
    if cfg.scale:
        steps.append(("scale", StandardScaler()))

    if cfg.model_type == "sgd_logistic":
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=cfg.alpha,
            max_iter=cfg.max_iter,
            tol=1e-4,
            class_weight="balanced",
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            early_stopping=False,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
    elif cfg.model_type == "logistic_saga":
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=cfg.c_value,
            max_iter=cfg.max_iter,
            tol=1e-4,
            class_weight="balanced",
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
    elif cfg.model_type == "linear_svc":
        clf = LinearSVC(
            C=cfg.c_value,
            class_weight="balanced",
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )
    else:
        raise ValueError(f"Unknown --model-type {cfg.model_type!r}")

    steps.append(("clf", clf))
    return Pipeline(steps)


def _class_counts(y: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    return {
        str(labels[int(i)]) if int(i) < len(labels) else f"class_{int(i)}": int(np.sum(y == i))
        for i in np.unique(y)
    }


def _normalise_probability_rows(proba: np.ndarray) -> np.ndarray:
    """Return non-negative rows that sum to 1 for metric calculations."""
    arr = np.nan_to_num(np.asarray(proba, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    row_sums = arr.sum(axis=1, keepdims=True)
    empty = row_sums[:, 0] <= 1e-12
    if np.any(empty):
        arr[empty] = 1.0 / arr.shape[1]
        row_sums = arr.sum(axis=1, keepdims=True)
    return arr / row_sums


def _softmax_scores(scores: np.ndarray) -> np.ndarray:
    """Stable softmax for classifier decision scores."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.stack([-arr, arr], axis=1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - arr.max(axis=1, keepdims=True)
    exp_scores = np.exp(arr)
    return _normalise_probability_rows(exp_scores)


def _predict_probability_matrix(estimator: Any, X: np.ndarray, n_classes: int) -> tuple[np.ndarray | None, str | None]:
    """Return robust class probabilities aligned to 0..n_classes-1 when possible.

    SGDClassifier.predict_proba can emit a RuntimeWarning from inside sklearn
    when multiclass rows underflow to zero. For the SGD path we use a stable
    softmax over decision_function scores instead, which is sufficient for
    cross-validation log-loss bookkeeping and avoids noisy local logs.
    """
    clf = estimator.named_steps.get("clf") if hasattr(estimator, "named_steps") else estimator
    class_ids = np.arange(n_classes, dtype=np.int64)

    clf_class_name = getattr(clf, "__class__", type(clf)).__name__
    if clf_class_name == "SGDClassifier" and hasattr(estimator, "decision_function"):
        proba = _softmax_scores(estimator.decision_function(X))
        seen_classes = getattr(clf, "classes_", class_ids)
        aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
        for src_col, class_id in enumerate(seen_classes):
            if int(class_id) < n_classes:
                aligned[:, int(class_id)] = proba[:, src_col]
        return _normalise_probability_rows(aligned), "decision_function_softmax"

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        seen_classes = getattr(clf, "classes_", class_ids)
        aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
        for src_col, class_id in enumerate(seen_classes):
            if int(class_id) < n_classes:
                aligned[:, int(class_id)] = proba[:, src_col]
        return _normalise_probability_rows(aligned), "predict_proba"

    if hasattr(estimator, "decision_function"):
        return _softmax_scores(estimator.decision_function(X)), "decision_function_softmax"

    return None, None


def _evaluate_group_cv(data: TrainingData, cfg: TrainConfig) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
    )
    from sklearn.model_selection import GroupKFold

    unique_subjects = np.unique(data.subject)
    n_splits = min(int(cfg.cv_splits), len(unique_subjects))
    if n_splits < 2:
        return {
            "skipped": True,
            "reason": "Need at least two subjects for subject-held-out CV.",
        }

    splitter = GroupKFold(n_splits=n_splits)
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    proba_all: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    class_ids = np.arange(len(data.labels), dtype=np.int64)

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(data.X, data.y, groups=data.subject),
        start=1,
    ):
        fold_start = time.time()
        estimator = _make_estimator(cfg)
        estimator.fit(data.X[train_idx], data.y[train_idx])
        y_pred = estimator.predict(data.X[test_idx])
        y_true = data.y[test_idx]

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        fold_metric: dict[str, Any] = {
            "fold": fold_idx,
            "train_subjects": sorted(np.unique(data.subject[train_idx]).tolist()),
            "test_subjects": sorted(np.unique(data.subject[test_idx]).tolist()),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "elapsed_seconds": round(time.time() - fold_start, 3),
        }

        proba, proba_method = _predict_probability_matrix(estimator, data.X[test_idx], len(class_ids))
        if proba is not None:
            proba_all.append(proba)
            fold_metric["probability_method"] = proba_method
            fold_metric["log_loss"] = float(log_loss(y_true, proba, labels=class_ids))

        fold_rows.append(fold_metric)

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)
    metrics: dict[str, Any] = {
        "skipped": False,
        "n_splits": n_splits,
        "folds": fold_rows,
        "mean_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
        "std_accuracy": float(np.std([row["accuracy"] for row in fold_rows])),
        "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        "std_balanced_accuracy": float(np.std([row["balanced_accuracy"] for row in fold_rows])),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])),
        "std_macro_f1": float(np.std([row["macro_f1"] for row in fold_rows])),
        "pooled_accuracy": float(accuracy_score(y_true_concat, y_pred_concat)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true_concat, y_pred_concat)),
        "pooled_macro_f1": float(f1_score(y_true_concat, y_pred_concat, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_concat, y_pred_concat, labels=class_ids).tolist(),
        "classification_report": classification_report(
            y_true_concat,
            y_pred_concat,
            labels=class_ids,
            target_names=data.labels.tolist(),
            output_dict=True,
            zero_division=0,
        ),
    }
    if proba_all:
        proba_concat = np.concatenate(proba_all)
        metrics["pooled_log_loss"] = float(log_loss(y_true_concat, proba_concat, labels=class_ids))
        metrics["mean_log_loss"] = float(np.mean([row["log_loss"] for row in fold_rows if "log_loss" in row]))
    return metrics


def train_model_from_npz(train_npz: Path, cfg: TrainConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train final model and return (artifact, metrics)."""
    data = _load_training_data(train_npz, cfg)
    print(f"Loaded {train_npz}")
    print(f"  input feature shape: {data.input_shape}")
    print(f"  model feature mode: {data.feature_mode}")
    print(f"  model feature shape: {data.source_shape}")
    print(f"  X flattened shape: {data.X.shape}")
    label_filter = data.metadata.get("label_filter", {})
    if label_filter.get("excluded_labels"):
        print(f"  excluded labels: {', '.join(label_filter['excluded_labels'])}")
    temporal_summary = data.metadata.get("temporal_aggregation", {})
    if temporal_summary.get("enabled"):
        print(
            "  temporal aggregation: "
            f"{temporal_summary['window_trs']} TRs, "
            f"stride {temporal_summary['stride_trs']}, "
            f"{temporal_summary['reducer']}, "
            f"{temporal_summary['contiguity']}"
        )
    if data.roi_spec is not None:
        print(
            f"  ROI parcels: {data.roi_spec.n_rois}; "
            f"reducers: {', '.join(data.roi_spec.reducers)}; "
            f"atlas: {data.roi_spec.atlas_id or 'unversioned'}"
        )
    print(f"  subjects: {len(np.unique(data.subject))}")
    print(f"  class counts: {_class_counts(data.y, data.labels)}")

    cv_metrics: dict[str, Any]
    if cfg.no_cv:
        cv_metrics = {"skipped": True, "reason": "--no-cv"}
    else:
        print(f"Running subject-held-out CV ({cfg.cv_splits} requested split(s)) ...")
        cv_metrics = _evaluate_group_cv(data, cfg)

    print("Training final model on all samples ...")
    start = time.time()
    estimator = _make_estimator(cfg)
    estimator.fit(data.X, data.y)
    elapsed = time.time() - start

    artifact = {
        "model": estimator,
        "labels": data.labels.tolist(),
        "input_feature_shape": list(data.input_shape),
        "feature_shape": list(data.source_shape),
        "flattened_n_features": int(data.X.shape[1]),
        "feature_mode": data.feature_mode,
        "roi_reducers": list(data.roi_spec.reducers) if data.roi_spec is not None else None,
        "roi_spec": summarise_roi_feature_spec(data.roi_spec, include_vertex_map=True)
        if data.roi_spec is not None
        else None,
        "model_type": cfg.model_type,
        "config": asdict(cfg),
        "source_contract": data.metadata.get("source_contract"),
        "label_filter": data.metadata.get("label_filter"),
        "temporal_aggregation": data.metadata.get("temporal_aggregation"),
        "training_metadata": data.metadata,
    }
    notes = [
        "Cross-validation is grouped by subject to avoid subject/session leakage.",
        "Model expects features in the same order and preprocessing as neuroemo_tribev2_train.npz.",
        "NeuroEmo-derived outputs are model-assisted emotion-state hypotheses, not clinical measurements.",
    ]
    if data.roi_spec is not None:
        notes.insert(
            2,
            "ROI mode reduces vertices within atlas parcel_id groups before fitting.",
        )

    metrics = {
        "created_at_unix_ms": int(time.time() * 1000),
        "elapsed_final_fit_seconds": round(elapsed, 3),
        "model_type": cfg.model_type,
        "n_samples": int(data.X.shape[0]),
        "n_features": int(data.X.shape[1]),
        "input_feature_shape": list(data.input_shape),
        "feature_shape": list(data.source_shape),
        "feature_mode": data.feature_mode,
        "roi_reducers": list(data.roi_spec.reducers) if data.roi_spec is not None else None,
        "roi_spec": summarise_roi_feature_spec(data.roi_spec) if data.roi_spec is not None else None,
        "source_contract": data.metadata.get("source_contract"),
        "label_filter": data.metadata.get("label_filter"),
        "temporal_aggregation": data.metadata.get("temporal_aggregation"),
        "labels": data.labels.tolist(),
        "subjects": sorted(np.unique(data.subject).tolist()),
        "class_counts": _class_counts(data.y, data.labels),
        "config": asdict(cfg),
        "cross_validation": cv_metrics,
        "notes": notes,
    }
    return artifact, _json_ready(metrics)


def save_artifacts(artifact: dict[str, Any], metrics: dict[str, Any], cfg: TrainConfig) -> None:
    model_path = Path(cfg.output_model)
    metrics_path = Path(cfg.metrics_json)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--model-type", choices=("sgd_logistic", "logistic_saga", "linear_svc"), default="sgd_logistic")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.0001, help="Regularization strength for sgd_logistic")
    parser.add_argument("--c-value", type=float, default=1.0, help="Inverse regularization for logistic_saga")
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap; 0 uses all samples")
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--no-scale", action="store_true", help="Disable StandardScaler in the sklearn pipeline")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--feature-mode",
        choices=("roi", "vertices"),
        default="roi",
        help="Feature representation. roi averages vertices by parcel_id before training; vertices keeps 20484 features.",
    )
    parser.add_argument("--vertex-csv", type=Path, default=DEFAULT_VERTEX_CSV, help="vertex_regions.csv with parcel_id column")
    parser.add_argument(
        "--roi-reducers",
        default="mean",
        help="Comma-separated ROI reducers. Supported: mean,std,mean_abs,max_abs",
    )
    parser.add_argument(
        "--roi-reducer",
        choices=("mean", "mean_abs", "std", "max_abs"),
        default=None,
        help="Deprecated alias for a single --roi-reducers value",
    )
    parser.add_argument(
        "--atlas-manifest",
        type=Path,
        default=DEFAULT_ATLAS_MANIFEST,
        help="Parcellation manifest used to validate the ROI CSV",
    )
    parser.add_argument(
        "--temporal-window-trs",
        type=int,
        default=1,
        help="Aggregate same-subject, same-label TRs into windows before ROI/model training. 1 disables.",
    )
    parser.add_argument(
        "--temporal-stride-trs",
        type=int,
        default=1,
        help="Stride between temporal windows in source TR rows.",
    )
    parser.add_argument(
        "--temporal-reducer",
        choices=("mean", "median", "last"),
        default="mean",
        help="How to reduce each temporal window.",
    )
    parser.add_argument(
        "--temporal-contiguity",
        choices=("contiguous", "same_label"),
        default="contiguous",
        help=(
            "contiguous requires source t_idx increments of 1; same_label allows windows "
            "over available same-label samples, useful for sparse diagnostic labels."
        ),
    )
    parser.add_argument(
        "--temporal-allow-class-drop",
        action="store_true",
        help="Allow temporal aggregation to drop classes. Use for diagnostics only.",
    )
    parser.add_argument(
        "--temporal-allow-rewindow",
        action="store_true",
        help="Allow a second temporal aggregation pass on already-windowed NPZ inputs. Diagnostics only.",
    )
    parser.add_argument(
        "--exclude-labels",
        default="neutral",
        help="Comma-separated class labels to exclude before training. Default excludes neutral.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        train_npz=str(args.train_npz),
        output_model=str(args.output_model),
        metrics_json=str(args.metrics_json),
        model_type=args.model_type,
        cv_splits=args.cv_splits,
        max_iter=args.max_iter,
        alpha=args.alpha,
        c_value=args.c_value,
        random_state=args.random_state,
        max_samples=args.max_samples,
        no_cv=args.no_cv,
        n_jobs=args.n_jobs,
        scale=not args.no_scale,
        feature_mode=args.feature_mode,
        vertex_csv=str(args.vertex_csv),
        roi_reducers=args.roi_reducer or args.roi_reducers,
        atlas_manifest=str(args.atlas_manifest) if args.atlas_manifest else "",
        temporal_window_trs=args.temporal_window_trs,
        temporal_stride_trs=args.temporal_stride_trs,
        temporal_reducer=args.temporal_reducer,
        temporal_contiguity=args.temporal_contiguity,
        temporal_allow_class_drop=args.temporal_allow_class_drop,
        temporal_allow_rewindow=args.temporal_allow_rewindow,
        exclude_labels=args.exclude_labels,
    )


def main() -> None:
    cfg = _config_from_args(build_arg_parser().parse_args())
    artifact, metrics = train_model_from_npz(Path(cfg.train_npz), cfg)
    save_artifacts(artifact, metrics, cfg)


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal
except ImportError:  # pragma: no cover - local training can run without Modal installed.
    modal = None


if modal is not None:
    app = modal.App("neuroemo-emotion-training")
    train_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "numpy>=1.26.4,<2.1.0",
            "scikit-learn>=1.6,<1.7",
            "joblib>=1.4",
            "pyyaml>=6",
        )
    )

    @app.function(image=train_image, cpu=8, memory=16384, timeout=7200)
    def _train_remote(
        train_npz_bytes: bytes,
        cfg_payload: dict[str, Any],
        vertex_csv_text: str | None = None,
        atlas_manifest_text: str | None = None,
    ) -> dict[str, bytes | str]:
        if vertex_csv_text is not None:
            vertex_csv_path = Path(tempfile.gettempdir()) / "vertex_regions.csv"
            vertex_csv_path.write_text(vertex_csv_text, encoding="utf-8")
            cfg_payload = {**cfg_payload, "vertex_csv": str(vertex_csv_path)}
        if atlas_manifest_text is not None:
            manifest_path = Path(tempfile.gettempdir()) / "parcellation_manifest.yaml"
            manifest_path.write_text(atlas_manifest_text, encoding="utf-8")
            cfg_payload = {**cfg_payload, "atlas_manifest": str(manifest_path)}
        cfg = TrainConfig(**cfg_payload)
        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "neuroemo_tribev2_train.npz"
            train_path.write_bytes(train_npz_bytes)
            artifact, metrics = train_model_from_npz(train_path, cfg)

            model_buf = io.BytesIO()
            joblib.dump(artifact, model_buf)
            metrics_text = json.dumps(metrics, indent=2)
            return {
                "model_bytes": model_buf.getvalue(),
                "metrics_json": metrics_text,
            }

    @app.local_entrypoint()
    def train_modal(
        train_npz: str = str(DEFAULT_TRAIN_NPZ),
        output_model: str = str(DEFAULT_MODEL_PATH),
        metrics_json: str = str(DEFAULT_METRICS_PATH),
        model_type: str = "sgd_logistic",
        cv_splits: int = 5,
        max_iter: int = 3000,
        alpha: float = 0.0001,
        c_value: float = 1.0,
        random_state: int = 13,
        max_samples: int = 0,
        no_cv: bool = False,
        no_scale: bool = False,
        feature_mode: str = "roi",
        vertex_csv: str = str(DEFAULT_VERTEX_CSV),
        roi_reducers: str = "mean",
        atlas_manifest: str = str(DEFAULT_ATLAS_MANIFEST),
        temporal_window_trs: int = 1,
        temporal_stride_trs: int = 1,
        temporal_reducer: str = "mean",
        temporal_contiguity: str = "contiguous",
        temporal_allow_class_drop: bool = False,
        temporal_allow_rewindow: bool = False,
        exclude_labels: str = "neutral",
    ) -> None:
        """Train NeuroEmo classifier remotely on Modal CPU and save artifacts locally."""
        train_path = Path(train_npz).expanduser()
        if not train_path.is_absolute():
            train_path = PROJECT_ROOT / train_path
        if not train_path.is_file():
            raise SystemExit(f"Training NPZ not found: {train_path}")

        cfg = TrainConfig(
            train_npz=str(train_path),
            output_model=output_model,
            metrics_json=metrics_json,
            model_type=model_type,
            cv_splits=cv_splits,
            max_iter=max_iter,
            alpha=alpha,
            c_value=c_value,
            random_state=random_state,
            max_samples=max_samples,
            no_cv=no_cv,
            n_jobs=-1,
            scale=not no_scale,
            feature_mode=feature_mode,
            vertex_csv=vertex_csv,
            roi_reducers=roi_reducers,
            atlas_manifest=atlas_manifest,
            temporal_window_trs=temporal_window_trs,
            temporal_stride_trs=temporal_stride_trs,
            temporal_reducer=temporal_reducer,
            temporal_contiguity=temporal_contiguity,
            temporal_allow_class_drop=temporal_allow_class_drop,
            temporal_allow_rewindow=temporal_allow_rewindow,
            exclude_labels=exclude_labels,
        )
        vertex_csv_text = None
        atlas_manifest_text = None
        if feature_mode == "roi":
            vertex_csv_path = Path(vertex_csv).expanduser()
            if not vertex_csv_path.is_absolute():
                vertex_csv_path = PROJECT_ROOT / vertex_csv_path
            if not vertex_csv_path.is_file():
                raise SystemExit(f"ROI vertex CSV not found: {vertex_csv_path}")
            vertex_csv_text = vertex_csv_path.read_text(encoding="utf-8")
            manifest_path = Path(atlas_manifest).expanduser() if atlas_manifest else None
            if manifest_path is not None:
                if not manifest_path.is_absolute():
                    manifest_path = PROJECT_ROOT / manifest_path
                if not manifest_path.is_file():
                    raise SystemExit(f"Parcellation manifest not found: {manifest_path}")
                atlas_manifest_text = manifest_path.read_text(encoding="utf-8")

        print(f"Uploading training NPZ to Modal: {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)")
        result = _train_remote.remote(
            train_path.read_bytes(),
            asdict(cfg),
            vertex_csv_text,
            atlas_manifest_text,
        )

        model_path = Path(output_model).expanduser()
        metrics_path = Path(metrics_json).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not metrics_path.is_absolute():
            metrics_path = PROJECT_ROOT / metrics_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(result["model_bytes"])
        metrics_path.write_text(str(result["metrics_json"]), encoding="utf-8")
        print(f"Model:   {model_path}")
        print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
