#!/usr/bin/env python3
"""Train a small MLP NeuroEmo classifier for TribeV2-style outputs.

This is an experimental neural-network baseline. It intentionally keeps the
network small and regularized because the clean block-level NeuroEmo training
set is small relative to typical deep-learning needs.

Default behavior:
    - Excludes the derived neutral class.
    - Uses Schaefer ROI features via scripts/train_neuroemo_emotion_model.py.
    - Aggregates 2 contiguous TRs per sample before ROI reduction.
    - Uses subject-held-out cross-validation.

Usage:
    python scripts/train_neuroemo_mlp_model.py
    python scripts/train_neuroemo_mlp_model.py --temporal-window-trs 10
    python scripts/train_neuroemo_mlp_model.py --hidden-layers 128,64 --alpha 0.01
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scripts.train_neuroemo_emotion_model import (
    DEFAULT_ATLAS_MANIFEST,
    DEFAULT_TRAIN_NPZ,
    DEFAULT_VERTEX_CSV,
    PROJECT_ROOT,
    TEMPORAL_REDUCER_COMPONENTS,
    TrainConfig,
    _class_counts,
    _json_ready,
    _load_training_data,
    _predict_probability_matrix,
)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "scout_data" / "neuroemo" / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "neuroemo_mlp_emotion_model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_MODEL_DIR / "neuroemo_mlp_emotion_metrics.json"


@dataclass(frozen=True)
class MlpTrainConfig:
    train_npz: str = str(DEFAULT_TRAIN_NPZ)
    output_model: str = str(DEFAULT_MODEL_PATH)
    metrics_json: str = str(DEFAULT_METRICS_PATH)
    hidden_layers: str = "64"
    activation: str = "relu"
    alpha: float = 0.01
    batch_size: int = 64
    learning_rate_init: float = 0.001
    max_iter: int = 500
    early_stopping: bool = True
    validation_fraction: float = 0.15
    n_iter_no_change: int = 25
    epoch_metrics: bool = True
    cv_splits: int = 5
    random_state: int = 13
    max_samples: int = 0
    no_cv: bool = False
    scale: bool = True
    feature_mode: str = "roi"
    vertex_csv: str = str(DEFAULT_VERTEX_CSV)
    roi_reducers: str = "mean"
    atlas_manifest: str = str(DEFAULT_ATLAS_MANIFEST)
    temporal_window_trs: int = 2
    temporal_stride_trs: int = 1
    temporal_reducer: str = "mean"
    temporal_contiguity: str = "contiguous"
    temporal_allow_class_drop: bool = False
    exclude_labels: str = "neutral"
    label_merge_preset: str = "none"
    label_merge: str = ""


def _parse_hidden_layers(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--hidden-layers must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("--hidden-layers values must be positive")
    return values


def _base_train_config(cfg: MlpTrainConfig) -> TrainConfig:
    return TrainConfig(
        train_npz=cfg.train_npz,
        output_model=cfg.output_model,
        metrics_json=cfg.metrics_json,
        model_type="sgd_logistic",
        cv_splits=cfg.cv_splits,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        max_samples=cfg.max_samples,
        no_cv=cfg.no_cv,
        scale=cfg.scale,
        feature_mode=cfg.feature_mode,
        vertex_csv=cfg.vertex_csv,
        roi_reducers=cfg.roi_reducers,
        atlas_manifest=cfg.atlas_manifest,
        temporal_window_trs=cfg.temporal_window_trs,
        temporal_stride_trs=cfg.temporal_stride_trs,
        temporal_reducer=cfg.temporal_reducer,
        temporal_contiguity=cfg.temporal_contiguity,
        temporal_allow_class_drop=cfg.temporal_allow_class_drop,
        exclude_labels=cfg.exclude_labels,
    )


def _parse_label_merge_spec(raw: str) -> dict[str, list[str]]:
    """Parse 'positive=delighted,excited;negative=afraid,depressed;calm=calm'."""
    mapping: dict[str, list[str]] = {}
    if not raw.strip():
        return mapping
    for group in raw.split(";"):
        item = group.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--label-merge groups must look like merged=old1,old2; "
                f"got {item!r}"
            )
        new_label, old_labels_raw = item.split("=", maxsplit=1)
        new_label = new_label.strip()
        old_labels = [label.strip() for label in old_labels_raw.replace("|", ",").split(",") if label.strip()]
        if not new_label or not old_labels:
            raise ValueError(f"Invalid --label-merge group: {item!r}")
        mapping[new_label] = old_labels
    return mapping


def _label_merge_mapping(cfg: MlpTrainConfig) -> dict[str, list[str]]:
    if cfg.label_merge.strip():
        return _parse_label_merge_spec(cfg.label_merge)
    if cfg.label_merge_preset == "none":
        return {}
    if cfg.label_merge_preset == "valence3":
        return {
            "calm": ["calm"],
            "negative": ["afraid", "depressed"],
            "positive": ["delighted", "excited"],
        }
    if cfg.label_merge_preset == "arousal3":
        return {
            "low_arousal": ["calm", "depressed"],
            "threat": ["afraid"],
            "high_positive": ["delighted", "excited"],
        }
    raise ValueError(f"Unknown --label-merge-preset {cfg.label_merge_preset!r}")


def _apply_label_merge(data: Any, cfg: MlpTrainConfig) -> Any:
    mapping = _label_merge_mapping(cfg)
    if not mapping:
        return data

    old_labels = np.asarray(data.labels).astype(str)
    old_name_to_id = {name: i for i, name in enumerate(old_labels.tolist())}
    old_to_new: dict[int, int] = {}
    new_labels: list[str] = []
    for new_id, (new_label, old_names) in enumerate(mapping.items()):
        new_labels.append(new_label)
        for old_name in old_names:
            if old_name not in old_name_to_id:
                raise ValueError(
                    f"--label-merge references {old_name!r}, but available labels are "
                    f"{old_labels.tolist()}"
                )
            old_id = old_name_to_id[old_name]
            if old_id in old_to_new:
                raise ValueError(f"Label {old_name!r} appears in more than one merge group.")
            old_to_new[old_id] = new_id

    missing = [name for name, old_id in old_name_to_id.items() if old_id not in old_to_new]
    if missing:
        raise ValueError(
            f"--label-merge does not assign all available labels. Missing: {missing}. "
            "Either include them in a group or exclude them first."
        )

    y_merged = np.asarray([old_to_new[int(class_id)] for class_id in data.y], dtype=np.int64)
    metadata = {
        **data.metadata,
        "label_merge": {
            "preset": cfg.label_merge_preset,
            "spec": cfg.label_merge,
            "mapping": mapping,
            "original_labels": old_labels.tolist(),
            "merged_labels": new_labels,
            "class_counts": _class_counts(y_merged, np.asarray(new_labels)),
        },
    }
    return replace(data, y=y_merged, labels=np.asarray(new_labels), metadata=metadata)


def _make_estimator(cfg: MlpTrainConfig):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[tuple[str, Any]] = []
    if cfg.scale:
        steps.append(("scale", StandardScaler()))

    clf = MLPClassifier(
        hidden_layer_sizes=_parse_hidden_layers(cfg.hidden_layers),
        activation=cfg.activation,
        solver="adam",
        alpha=cfg.alpha,
        batch_size=cfg.batch_size,
        learning_rate="adaptive",
        learning_rate_init=cfg.learning_rate_init,
        max_iter=cfg.max_iter,
        early_stopping=cfg.early_stopping,
        validation_fraction=cfg.validation_fraction,
        n_iter_no_change=cfg.n_iter_no_change,
        verbose=cfg.epoch_metrics,
        random_state=cfg.random_state,
    )
    steps.append(("clf", clf))
    return Pipeline(steps)


def _make_mlp_classifier(cfg: MlpTrainConfig, *, max_iter: int = 1):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=_parse_hidden_layers(cfg.hidden_layers),
        activation=cfg.activation,
        solver="adam",
        alpha=cfg.alpha,
        batch_size=cfg.batch_size,
        learning_rate="adaptive",
        learning_rate_init=cfg.learning_rate_init,
        max_iter=max_iter,
        early_stopping=False,
        validation_fraction=cfg.validation_fraction,
        n_iter_no_change=cfg.n_iter_no_change,
        verbose=False,
        random_state=cfg.random_state,
    )


def _subject_validation_split(
    y: np.ndarray,
    subject: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return train/validation indices where validation contains whole subjects."""
    unique_subjects = np.unique(subject.astype(str))
    if unique_subjects.shape[0] < 2:
        idx = np.arange(y.shape[0], dtype=np.int64)
        return idx, np.empty((0,), dtype=np.int64), []

    rng = np.random.default_rng(random_state)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(float(validation_fraction) * len(shuffled))))
    n_val = min(n_val, len(shuffled) - 1)
    all_classes = set(np.unique(y).tolist())

    best: tuple[np.ndarray, np.ndarray, list[str]] | None = None
    for start in range(len(shuffled)):
        rolled = np.roll(shuffled, start)
        val_subjects = sorted(rolled[:n_val].tolist())
        val_mask = np.isin(subject, val_subjects)
        train_idx = np.flatnonzero(~val_mask)
        val_idx = np.flatnonzero(val_mask)
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        if set(np.unique(y[train_idx]).tolist()) == all_classes:
            return train_idx.astype(np.int64), val_idx.astype(np.int64), val_subjects
        if best is None:
            best = (train_idx.astype(np.int64), val_idx.astype(np.int64), val_subjects)

    if best is not None:
        return best
    idx = np.arange(y.shape[0], dtype=np.int64)
    return idx, np.empty((0,), dtype=np.int64), []


def _pipeline_from_parts(scaler: Any | None, clf: Any):
    from sklearn.pipeline import Pipeline

    steps: list[tuple[str, Any]] = []
    if scaler is not None:
        steps.append(("scale", scaler))
    steps.append(("clf", clf))
    return Pipeline(steps)


def _fit_estimator(
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    cfg: MlpTrainConfig,
    *,
    log_prefix: str = "",
) -> tuple[Any, dict[str, Any]]:
    """Fit an MLP with subject-level validation for early stopping."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler

    if not cfg.early_stopping:
        estimator = _make_estimator(MlpTrainConfig(**{**asdict(cfg), "early_stopping": False}))
        estimator.fit(X, y)
        clf = estimator.named_steps["clf"]
        return estimator, {
            "mode": "sklearn_no_early_stopping",
            "epochs_run": int(getattr(clf, "n_iter_", cfg.max_iter)),
            "loss_curve": list(getattr(clf, "loss_curve_", [])),
            "validation_scores": [],
            "validation_subjects": [],
        }

    train_idx, val_idx, val_subjects = _subject_validation_split(
        y,
        subject,
        validation_fraction=cfg.validation_fraction,
        random_state=cfg.random_state,
    )
    if val_idx.size == 0:
        estimator = _make_estimator(MlpTrainConfig(**{**asdict(cfg), "early_stopping": False}))
        estimator.fit(X, y)
        clf = estimator.named_steps["clf"]
        return estimator, {
            "mode": "sklearn_no_validation_subjects",
            "epochs_run": int(getattr(clf, "n_iter_", cfg.max_iter)),
            "loss_curve": list(getattr(clf, "loss_curve_", [])),
            "validation_scores": [],
            "validation_subjects": [],
        }

    scaler = StandardScaler() if cfg.scale else None
    X_train = X[train_idx]
    X_val = X[val_idx]
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    y_train = y[train_idx]
    y_val = y[val_idx]
    classes = np.arange(len(np.unique(y)), dtype=np.int64)
    clf = _make_mlp_classifier(cfg, max_iter=1)

    best_score = -np.inf
    best_state: dict[str, Any] | None = None
    no_improve = 0
    loss_curve: list[float] = []
    validation_scores: list[float] = []

    for epoch in range(1, int(cfg.max_iter) + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.partial_fit(X_train, y_train, classes=classes)
        loss = float(getattr(clf, "loss_", np.nan))
        val_score = float(clf.score(X_val, y_val))
        loss_curve.append(loss)
        validation_scores.append(val_score)

        if cfg.epoch_metrics:
            prefix = f"{log_prefix} " if log_prefix else ""
            print(f"{prefix}Epoch {epoch}, loss = {loss:.8f}")
            print(f"{prefix}Subject-validation score: {val_score:.6f}")

        if val_score > best_score + 1e-8:
            best_score = val_score
            no_improve = 0
            best_state = {
                "coefs_": copy.deepcopy(clf.coefs_),
                "intercepts_": copy.deepcopy(clf.intercepts_),
                "n_iter_": int(epoch),
                "loss_": loss,
                "classes_": copy.deepcopy(clf.classes_),
                "out_activation_": getattr(clf, "out_activation_", None),
                "n_layers_": getattr(clf, "n_layers_", None),
                "n_outputs_": getattr(clf, "n_outputs_", None),
            }
        else:
            no_improve += 1
            if no_improve >= int(cfg.n_iter_no_change):
                if cfg.epoch_metrics:
                    prefix = f"{log_prefix} " if log_prefix else ""
                    print(
                        f"{prefix}Stopping early after {epoch} epochs; "
                        f"best subject-validation score = {best_score:.6f}"
                    )
                break

    if best_state is not None:
        clf.coefs_ = best_state["coefs_"]
        clf.intercepts_ = best_state["intercepts_"]
        clf.n_iter_ = best_state["n_iter_"]
        clf.loss_ = best_state["loss_"]
        clf.classes_ = best_state["classes_"]
        if best_state["out_activation_"] is not None:
            clf.out_activation_ = best_state["out_activation_"]
        if best_state["n_layers_"] is not None:
            clf.n_layers_ = best_state["n_layers_"]
        if best_state["n_outputs_"] is not None:
            clf.n_outputs_ = best_state["n_outputs_"]

    clf.loss_curve_ = loss_curve
    clf.validation_scores_ = validation_scores
    clf.group_validation_scores_ = validation_scores
    clf.group_validation_subjects_ = val_subjects
    estimator = _pipeline_from_parts(scaler, clf)
    history = {
        "mode": "manual_subject_grouped_early_stopping",
        "epochs_run": int(len(loss_curve)),
        "best_validation_score": float(best_score),
        "best_epoch": int(np.argmax(validation_scores) + 1) if validation_scores else 0,
        "loss_curve": loss_curve,
        "validation_scores": validation_scores,
        "validation_subjects": val_subjects,
        "n_train": int(train_idx.size),
        "n_validation": int(val_idx.size),
    }
    return estimator, history


def _evaluate_group_cv(data: Any, cfg: MlpTrainConfig) -> dict[str, Any]:
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
        return {"skipped": True, "reason": "Need at least two subjects for subject-held-out CV."}

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
        estimator, fit_history = _fit_estimator(
            data.X[train_idx],
            data.y[train_idx],
            data.subject[train_idx],
            cfg,
            log_prefix=f"[fold {fold_idx}]",
        )
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
            "fit_history": fit_history,
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
        "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])),
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


def train_model_from_npz(train_npz: Path, cfg: MlpTrainConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    base_cfg = _base_train_config(cfg)
    data = _load_training_data(train_npz, base_cfg)
    data = _apply_label_merge(data, cfg)

    print(f"Loaded {train_npz}")
    print(f"  input feature shape: {data.input_shape}")
    print(f"  model feature mode: {data.feature_mode}")
    print(f"  model feature shape: {data.source_shape}")
    print(f"  X flattened shape: {data.X.shape}")
    if data.metadata.get("label_filter", {}).get("excluded_labels"):
        print(f"  excluded labels: {', '.join(data.metadata['label_filter']['excluded_labels'])}")
    if data.metadata.get("label_merge"):
        merge = data.metadata["label_merge"]
        print(f"  label merge: {merge['merged_labels']} from {merge['mapping']}")
    if data.metadata.get("temporal_aggregation", {}).get("enabled"):
        temporal = data.metadata["temporal_aggregation"]
        print(
            "  temporal aggregation: "
            f"{temporal['window_trs']} TRs, "
            f"stride {temporal['stride_trs']}, "
            f"{temporal['reducer']}, "
            f"{temporal['contiguity']}"
        )
    if data.roi_spec is not None:
        print(
            f"  ROI parcels: {data.roi_spec.n_rois}; "
            f"reducers: {', '.join(data.roi_spec.reducers)}; "
            f"atlas: {data.roi_spec.atlas_id or 'unversioned'}"
        )
    print(f"  MLP hidden layers: {_parse_hidden_layers(cfg.hidden_layers)}")
    print(f"  subjects: {len(np.unique(data.subject))}")
    print(f"  class counts: {_class_counts(data.y, data.labels)}")

    if cfg.no_cv:
        cv_metrics: dict[str, Any] = {"skipped": True, "reason": "--no-cv"}
    else:
        print(f"Running subject-held-out CV ({cfg.cv_splits} requested split(s)) ...")
        cv_metrics = _evaluate_group_cv(data, cfg)

    print("Training final MLP on all samples ...")
    start = time.time()
    estimator, final_fit_history = _fit_estimator(
        data.X,
        data.y,
        data.subject,
        cfg,
        log_prefix="[final]",
    )
    elapsed = time.time() - start

    artifact = {
        "model": estimator,
        "labels": data.labels.tolist(),
        "input_feature_shape": list(data.input_shape),
        "feature_shape": list(data.source_shape),
        "flattened_n_features": int(data.X.shape[1]),
        "feature_mode": data.feature_mode,
        "model_type": "mlp",
        "config": asdict(cfg),
        "base_training_config": asdict(base_cfg),
        "fit_history": final_fit_history,
        "training_metadata": data.metadata,
    }
    if data.roi_spec is not None:
        from scout_core.roi_features import summarise_roi_feature_spec

        artifact["roi_reducers"] = list(data.roi_spec.reducers)
        artifact["roi_spec"] = summarise_roi_feature_spec(data.roi_spec, include_vertex_map=True)

    metrics = {
        "created_at_unix_ms": int(time.time() * 1000),
        "elapsed_final_fit_seconds": round(elapsed, 3),
        "model_type": "mlp",
        "hidden_layers": list(_parse_hidden_layers(cfg.hidden_layers)),
        "activation": cfg.activation,
        "alpha": cfg.alpha,
        "early_stopping": cfg.early_stopping,
        "early_stopping_mode": final_fit_history.get("mode"),
        "fit_history": final_fit_history,
        "n_samples": int(data.X.shape[0]),
        "n_features": int(data.X.shape[1]),
        "input_feature_shape": list(data.input_shape),
        "feature_shape": list(data.source_shape),
        "feature_mode": data.feature_mode,
        "labels": data.labels.tolist(),
        "subjects": sorted(np.unique(data.subject).tolist()),
        "class_counts": _class_counts(data.y, data.labels),
        "label_filter": data.metadata.get("label_filter"),
        "label_merge": data.metadata.get("label_merge"),
        "temporal_aggregation": data.metadata.get("temporal_aggregation"),
        "config": asdict(cfg),
        "base_training_config": asdict(base_cfg),
        "cross_validation": cv_metrics,
        "notes": [
            "Experimental MLP baseline; compare against the linear ROI model using the same subject-held-out CV.",
            "Default temporal window is 2 contiguous TRs to increase samples while limiting long-window overfitting.",
            "Neutral is excluded by default because it is derived from white-noise control blocks, not a raw emotion condition.",
        ],
    }
    if data.roi_spec is not None:
        from scout_core.roi_features import summarise_roi_feature_spec

        metrics["roi_reducers"] = list(data.roi_spec.reducers)
        metrics["roi_spec"] = summarise_roi_feature_spec(data.roi_spec)

    return artifact, _json_ready(metrics)


def save_artifacts(artifact: dict[str, Any], metrics: dict[str, Any], cfg: MlpTrainConfig) -> None:
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
    parser.add_argument("--hidden-layers", default="64", help="Comma-separated hidden layer sizes, e.g. 64 or 128,64")
    parser.add_argument("--activation", choices=("identity", "logistic", "tanh", "relu"), default="relu")
    parser.add_argument("--alpha", type=float, default=0.01, help="L2 regularization strength")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate-init", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--n-iter-no-change", type=int, default=25)
    parser.add_argument(
        "--no-epoch-metrics",
        action="store_true",
        help="Disable live MLP epoch/iteration loss and validation-score logging.",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--feature-mode", choices=("roi", "vertices"), default="roi")
    parser.add_argument("--vertex-csv", type=Path, default=DEFAULT_VERTEX_CSV)
    parser.add_argument("--roi-reducers", default="mean")
    parser.add_argument("--atlas-manifest", type=Path, default=DEFAULT_ATLAS_MANIFEST)
    parser.add_argument("--temporal-window-trs", type=int, default=2)
    parser.add_argument("--temporal-stride-trs", type=int, default=1)
    parser.add_argument("--temporal-reducer", choices=tuple(TEMPORAL_REDUCER_COMPONENTS), default="mean")
    parser.add_argument("--temporal-contiguity", choices=("contiguous", "same_label"), default="contiguous")
    parser.add_argument("--temporal-allow-class-drop", action="store_true")
    parser.add_argument("--exclude-labels", default="neutral")
    parser.add_argument(
        "--label-merge-preset",
        choices=("none", "valence3", "arousal3"),
        default="none",
        help="Optional built-in merged-label target.",
    )
    parser.add_argument(
        "--label-merge",
        default="",
        help=(
            "Custom merge spec, e.g. "
            "'positive=delighted,excited;negative=afraid,depressed;calm=calm'. "
            "All remaining labels must be assigned."
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> MlpTrainConfig:
    return MlpTrainConfig(
        train_npz=str(args.train_npz),
        output_model=str(args.output_model),
        metrics_json=str(args.metrics_json),
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        early_stopping=not args.no_early_stopping,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
        epoch_metrics=not args.no_epoch_metrics,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        max_samples=args.max_samples,
        no_cv=args.no_cv,
        scale=not args.no_scale,
        feature_mode=args.feature_mode,
        vertex_csv=str(args.vertex_csv),
        roi_reducers=args.roi_reducers,
        atlas_manifest=str(args.atlas_manifest) if args.atlas_manifest else "",
        temporal_window_trs=args.temporal_window_trs,
        temporal_stride_trs=args.temporal_stride_trs,
        temporal_reducer=args.temporal_reducer,
        temporal_contiguity=args.temporal_contiguity,
        temporal_allow_class_drop=args.temporal_allow_class_drop,
        exclude_labels=args.exclude_labels,
        label_merge_preset=args.label_merge_preset,
        label_merge=args.label_merge,
    )


def main() -> None:
    cfg = _config_from_args(build_arg_parser().parse_args())
    artifact, metrics = train_model_from_npz(Path(cfg.train_npz), cfg)
    save_artifacts(artifact, metrics, cfg)


if __name__ == "__main__":
    main()
