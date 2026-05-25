#!/usr/bin/env python3
"""Train one-vs-rest NeuroEmo specialist classifiers.

This trains one binary classifier per emotion:

    calm vs other emotions
    afraid vs other emotions
    delighted vs other emotions
    depressed vs other emotions
    excited vs other emotions

At inference/evaluation time, the positive probabilities from all specialists
are stacked and the highest-probability specialist becomes the discrete emotion
prediction. This lets each emotion have its own decision boundary instead of
forcing one shared multiclass boundary.

Default configuration is tuned for the current NeuroEmo experiments:

    - neutral excluded
    - 10 contiguous TRs per sample
    - Schaefer ROI mean,std,mean_abs features
    - class_weight="balanced" binary logistic specialists
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scripts.train_neuroemo_emotion_model import (
    DEFAULT_ATLAS_MANIFEST,
    DEFAULT_TRAIN_NPZ,
    DEFAULT_VERTEX_CSV,
    PROJECT_ROOT,
    TrainConfig,
    _class_counts,
    _json_ready,
    _load_training_data,
)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "scout_data" / "neuroemo" / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "neuroemo_specialist_emotion_model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_MODEL_DIR / "neuroemo_specialist_emotion_metrics.json"


@dataclass(frozen=True)
class SpecialistConfig:
    train_npz: str = str(DEFAULT_TRAIN_NPZ)
    output_model: str = str(DEFAULT_MODEL_PATH)
    metrics_json: str = str(DEFAULT_METRICS_PATH)
    model_type: str = "logistic"
    cv_splits: int = 5
    max_iter: int = 3000
    c_value: float = 1.0
    alpha: float = 0.0001
    random_state: int = 13
    max_samples: int = 0
    no_cv: bool = False
    n_jobs: int = -1
    scale: bool = True
    feature_mode: str = "roi"
    vertex_csv: str = str(DEFAULT_VERTEX_CSV)
    roi_reducers: str = "mean,std,mean_abs"
    atlas_manifest: str = str(DEFAULT_ATLAS_MANIFEST)
    temporal_window_trs: int = 10
    temporal_stride_trs: int = 1
    temporal_reducer: str = "mean"
    temporal_contiguity: str = "contiguous"
    temporal_allow_class_drop: bool = False
    exclude_labels: str = "neutral"
    specialists: str = ""
    calibration: str = "sigmoid"
    calibration_fraction: float = 0.2


def _base_train_config(cfg: SpecialistConfig) -> TrainConfig:
    return TrainConfig(
        train_npz=cfg.train_npz,
        output_model=cfg.output_model,
        metrics_json=cfg.metrics_json,
        model_type="sgd_logistic",
        cv_splits=cfg.cv_splits,
        max_iter=cfg.max_iter,
        alpha=cfg.alpha,
        c_value=cfg.c_value,
        random_state=cfg.random_state,
        max_samples=cfg.max_samples,
        no_cv=cfg.no_cv,
        n_jobs=cfg.n_jobs,
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


def _specialist_labels(all_labels: np.ndarray, raw: str) -> list[str]:
    available = all_labels.astype(str).tolist()
    if not raw.strip():
        return available
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unknown specialist labels {missing}; available labels are {available}")
    return selected


def _make_binary_estimator(cfg: SpecialistConfig):
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[tuple[str, Any]] = []
    if cfg.scale:
        steps.append(("scale", StandardScaler()))

    if cfg.model_type == "logistic":
        clf = LogisticRegression(
            solver="liblinear",
            C=cfg.c_value,
            class_weight="balanced",
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )
    elif cfg.model_type == "sgd_logistic":
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=cfg.alpha,
            class_weight="balanced",
            max_iter=cfg.max_iter,
            tol=1e-4,
            random_state=cfg.random_state,
        )
    else:
        raise ValueError(f"Unknown --model-type {cfg.model_type!r}")

    steps.append(("clf", clf))
    return Pipeline(steps)


def _subject_calibration_split(
    y_binary: np.ndarray,
    subject: np.ndarray,
    *,
    calibration_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Split binary data into fit/calibration rows using whole subjects."""
    unique_subjects = np.unique(subject.astype(str))
    if unique_subjects.shape[0] < 2:
        idx = np.arange(y_binary.shape[0], dtype=np.int64)
        return idx, np.empty((0,), dtype=np.int64), []

    rng = np.random.default_rng(random_state)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    n_cal = max(1, int(round(float(calibration_fraction) * len(shuffled))))
    n_cal = min(n_cal, len(shuffled) - 1)

    best: tuple[np.ndarray, np.ndarray, list[str]] | None = None
    for start in range(len(shuffled)):
        rolled = np.roll(shuffled, start)
        cal_subjects = sorted(rolled[:n_cal].tolist())
        cal_mask = np.isin(subject, cal_subjects)
        fit_idx = np.flatnonzero(~cal_mask).astype(np.int64)
        cal_idx = np.flatnonzero(cal_mask).astype(np.int64)
        if fit_idx.size == 0 or cal_idx.size == 0:
            continue
        fit_classes = set(np.unique(y_binary[fit_idx]).tolist())
        cal_classes = set(np.unique(y_binary[cal_idx]).tolist())
        if fit_classes == {0, 1} and cal_classes == {0, 1}:
            return fit_idx, cal_idx, cal_subjects
        if best is None and fit_classes == {0, 1}:
            best = (fit_idx, cal_idx, cal_subjects)

    if best is not None:
        return best
    idx = np.arange(y_binary.shape[0], dtype=np.int64)
    return idx, np.empty((0,), dtype=np.int64), []


def _raw_positive_score(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Return uncalibrated positive-class scores for sigmoid calibration."""
    clf = estimator.named_steps.get("clf") if hasattr(estimator, "named_steps") else estimator
    classes = getattr(clf, "classes_", np.asarray([0, 1]))
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=np.float64)
        if scores.ndim > 1:
            positive_cols = np.flatnonzero(classes == 1)
            if positive_cols.size:
                return scores[:, int(positive_cols[0])]
            return scores[:, -1]
        return scores
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        positive_cols = np.flatnonzero(classes == 1)
        if positive_cols.size:
            return np.asarray(proba[:, int(positive_cols[0])], dtype=np.float64)
        return np.zeros((X.shape[0],), dtype=np.float64)
    return np.asarray(estimator.predict(X), dtype=np.float64)


def _positive_probability(estimator: Any, X: np.ndarray) -> np.ndarray:
    if isinstance(estimator, dict):
        base_estimator = estimator["estimator"]
        calibrator = estimator.get("calibrator")
        if calibrator is not None:
            scores = _raw_positive_score(base_estimator, X).reshape(-1, 1)
            return np.asarray(calibrator.predict_proba(scores)[:, 1], dtype=np.float64)
        estimator = base_estimator

    clf = estimator.named_steps.get("clf") if hasattr(estimator, "named_steps") else estimator
    classes = getattr(clf, "classes_", np.asarray([0, 1]))
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        positive_cols = np.flatnonzero(classes == 1)
        if positive_cols.size:
            return np.asarray(proba[:, int(positive_cols[0])], dtype=np.float64)
        return np.zeros((X.shape[0],), dtype=np.float64)

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-scores))

    return np.asarray(estimator.predict(X), dtype=np.float64)


def _fit_specialist_estimator(
    X: np.ndarray,
    y_binary: np.ndarray,
    subject: np.ndarray,
    cfg: SpecialistConfig,
    *,
    specialist_name: str,
) -> tuple[Any, dict[str, Any]]:
    """Fit an emotion-vs-rest specialist with optional sigmoid calibration."""
    if cfg.calibration == "none":
        estimator = _make_binary_estimator(cfg)
        estimator.fit(X, y_binary)
        return estimator, {
            "calibration": "none",
            "positive_count": int(np.sum(y_binary == 1)),
            "negative_count": int(np.sum(y_binary == 0)),
        }
    if cfg.calibration != "sigmoid":
        raise ValueError(f"Unknown --calibration {cfg.calibration!r}")

    from sklearn.linear_model import LogisticRegression

    fit_idx, cal_idx, cal_subjects = _subject_calibration_split(
        y_binary,
        subject,
        calibration_fraction=cfg.calibration_fraction,
        random_state=cfg.random_state + abs(hash(specialist_name)) % 10000,
    )
    estimator = _make_binary_estimator(cfg)
    estimator.fit(X[fit_idx], y_binary[fit_idx])

    calibrator = None
    calibration_summary: dict[str, Any] = {
        "calibration": "sigmoid",
        "calibration_subjects": cal_subjects,
        "fit_count": int(fit_idx.size),
        "calibration_count": int(cal_idx.size),
        "fit_positive_count": int(np.sum(y_binary[fit_idx] == 1)),
        "fit_negative_count": int(np.sum(y_binary[fit_idx] == 0)),
        "calibration_positive_count": int(np.sum(y_binary[cal_idx] == 1)) if cal_idx.size else 0,
        "calibration_negative_count": int(np.sum(y_binary[cal_idx] == 0)) if cal_idx.size else 0,
        "positive_count": int(np.sum(y_binary == 1)),
        "negative_count": int(np.sum(y_binary == 0)),
    }
    if cal_idx.size and len(np.unique(y_binary[cal_idx])) == 2:
        scores = _raw_positive_score(estimator, X[cal_idx]).reshape(-1, 1)
        calibrator = LogisticRegression(solver="lbfgs", random_state=cfg.random_state)
        calibrator.fit(scores, y_binary[cal_idx])
        calibration_summary["calibrator_intercept"] = float(calibrator.intercept_[0])
        calibration_summary["calibrator_coef"] = float(calibrator.coef_[0, 0])
    else:
        calibration_summary["warning"] = (
            "Calibration split did not contain both binary classes; using raw probabilities."
        )

    return {"estimator": estimator, "calibrator": calibrator}, calibration_summary


def _train_specialists(
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    labels: np.ndarray,
    specialist_names: list[str],
    cfg: SpecialistConfig,
) -> dict[str, Any]:
    label_to_id = {str(label): i for i, label in enumerate(labels.astype(str).tolist())}
    models: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for name in specialist_names:
        class_id = label_to_id[name]
        y_binary = (y == class_id).astype(np.int64)
        estimator, summary = _fit_specialist_estimator(
            X,
            y_binary,
            subject,
            cfg,
            specialist_name=name,
        )
        models[name] = estimator
        summaries[name] = {"positive_label": name, **summary}
    return {"models": models, "summaries": summaries}


def _predict_specialist_matrix(models: dict[str, Any], specialist_names: list[str], X: np.ndarray) -> np.ndarray:
    columns = [_positive_probability(models[name], X) for name in specialist_names]
    return np.stack(columns, axis=1).astype(np.float64)


def _evaluate_group_cv(data: Any, specialist_names: list[str], cfg: SpecialistConfig) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold

    unique_subjects = np.unique(data.subject)
    n_splits = min(int(cfg.cv_splits), len(unique_subjects))
    if n_splits < 2:
        return {"skipped": True, "reason": "Need at least two subjects for subject-held-out CV."}

    label_to_id = {str(label): i for i, label in enumerate(data.labels.astype(str).tolist())}
    specialist_ids = np.asarray([label_to_id[name] for name in specialist_names], dtype=np.int64)
    if not np.array_equal(specialist_ids, np.arange(len(specialist_ids))):
        # Keep the argmax index space compact by requiring selected specialists
        # to match all remaining labels. This is the default and intended mode.
        raise ValueError(
            "Combined specialist evaluation currently expects specialists for every remaining label. "
            f"Got {specialist_names}; remaining labels are {data.labels.tolist()}"
        )

    splitter = GroupKFold(n_splits=n_splits)
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    score_all: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    specialist_binary_rows: dict[str, list[dict[str, float]]] = {name: [] for name in specialist_names}

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(data.X, data.y, groups=data.subject),
        start=1,
    ):
        fold_start = time.time()
        bundle = _train_specialists(
            data.X[train_idx],
            data.y[train_idx],
            data.subject[train_idx],
            data.labels,
            specialist_names,
            cfg,
        )
        scores = _predict_specialist_matrix(bundle["models"], specialist_names, data.X[test_idx])
        y_pred = np.argmax(scores, axis=1).astype(np.int64)
        y_true = data.y[test_idx]

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        score_all.append(scores)
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

        try:
            row_sums = scores.sum(axis=1, keepdims=True)
            proba = scores / np.clip(row_sums, 1e-12, None)
            fold_metric["log_loss"] = float(log_loss(y_true, proba, labels=np.arange(len(specialist_names))))
        except ValueError:
            pass

        for spec_idx, name in enumerate(specialist_names):
            target_id = label_to_id[name]
            y_bin_true = (y_true == target_id).astype(np.int64)
            y_bin_pred = (scores[:, spec_idx] >= 0.5).astype(np.int64)
            row = {
                "fold": float(fold_idx),
                "accuracy": float(accuracy_score(y_bin_true, y_bin_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_bin_true, y_bin_pred)),
                "f1": float(f1_score(y_bin_true, y_bin_pred, zero_division=0)),
            }
            if len(np.unique(y_bin_true)) == 2:
                row["roc_auc"] = float(roc_auc_score(y_bin_true, scores[:, spec_idx]))
            specialist_binary_rows[name].append(row)

        fold_rows.append(fold_metric)

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)
    score_concat = np.concatenate(score_all)
    proba_concat = score_concat / np.clip(score_concat.sum(axis=1, keepdims=True), 1e-12, None)
    labels = np.asarray(specialist_names)

    specialist_binary_summary: dict[str, Any] = {}
    for name, rows in specialist_binary_rows.items():
        specialist_binary_summary[name] = {
            "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in rows])),
            "mean_f1": float(np.mean([row["f1"] for row in rows])),
            "mean_roc_auc": float(np.mean([row["roc_auc"] for row in rows if "roc_auc" in row]))
            if any("roc_auc" in row for row in rows)
            else None,
            "folds": rows,
        }

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
        "pooled_log_loss": float(log_loss(y_true_concat, proba_concat, labels=np.arange(len(specialist_names)))),
        "confusion_matrix": confusion_matrix(
            y_true_concat,
            y_pred_concat,
            labels=np.arange(len(specialist_names)),
        ).tolist(),
        "classification_report": classification_report(
            y_true_concat,
            y_pred_concat,
            labels=np.arange(len(specialist_names)),
            target_names=labels.tolist(),
            output_dict=True,
            zero_division=0,
        ),
        "specialist_binary_metrics": specialist_binary_summary,
    }
    return metrics


def train_model_from_npz(train_npz: Path, cfg: SpecialistConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    from scout_core.roi_features import summarise_roi_feature_spec

    data = _load_training_data(train_npz, _base_train_config(cfg))
    specialist_names = _specialist_labels(data.labels, cfg.specialists)

    print(f"Loaded {train_npz}")
    print(f"  input feature shape: {data.input_shape}")
    print(f"  model feature mode: {data.feature_mode}")
    print(f"  model feature shape: {data.source_shape}")
    print(f"  X flattened shape: {data.X.shape}")
    if data.metadata.get("label_filter", {}).get("excluded_labels"):
        print(f"  excluded labels: {', '.join(data.metadata['label_filter']['excluded_labels'])}")
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
    print(f"  specialist labels: {', '.join(specialist_names)}")
    print(f"  subjects: {len(np.unique(data.subject))}")
    print(f"  class counts: {_class_counts(data.y, data.labels)}")

    if cfg.no_cv:
        cv_metrics: dict[str, Any] = {"skipped": True, "reason": "--no-cv"}
    else:
        print(f"Running subject-held-out specialist CV ({cfg.cv_splits} requested split(s)) ...")
        cv_metrics = _evaluate_group_cv(data, specialist_names, cfg)

    print("Training final specialist models on all samples ...")
    start = time.time()
    final_bundle = _train_specialists(data.X, data.y, data.subject, data.labels, specialist_names, cfg)
    elapsed = time.time() - start

    roi_spec = (
        summarise_roi_feature_spec(data.roi_spec, include_vertex_map=True)
        if data.roi_spec is not None
        else None
    )
    artifact = {
        "models": final_bundle["models"],
        "specialist_summaries": final_bundle["summaries"],
        "labels": data.labels.tolist(),
        "specialist_labels": specialist_names,
        "input_feature_shape": list(data.input_shape),
        "feature_shape": list(data.source_shape),
        "flattened_n_features": int(data.X.shape[1]),
        "feature_mode": data.feature_mode,
        "roi_reducers": list(data.roi_spec.reducers) if data.roi_spec is not None else None,
        "roi_spec": roi_spec,
        "model_type": f"specialist_{cfg.model_type}",
        "config": asdict(cfg),
        "base_training_config": asdict(_base_train_config(cfg)),
        "training_metadata": data.metadata,
    }

    metrics = {
        "created_at_unix_ms": int(time.time() * 1000),
        "elapsed_final_fit_seconds": round(elapsed, 3),
        "model_type": f"specialist_{cfg.model_type}",
        "calibration": cfg.calibration,
        "n_samples": int(data.X.shape[0]),
        "n_features": int(data.X.shape[1]),
        "input_feature_shape": list(data.input_shape),
        "feature_shape": list(data.source_shape),
        "feature_mode": data.feature_mode,
        "roi_reducers": list(data.roi_spec.reducers) if data.roi_spec is not None else None,
        "roi_spec": summarise_roi_feature_spec(data.roi_spec) if data.roi_spec is not None else None,
        "labels": data.labels.tolist(),
        "specialist_labels": specialist_names,
        "subjects": sorted(np.unique(data.subject).tolist()),
        "class_counts": _class_counts(data.y, data.labels),
        "label_filter": data.metadata.get("label_filter"),
        "temporal_aggregation": data.metadata.get("temporal_aggregation"),
        "config": asdict(cfg),
        "base_training_config": asdict(_base_train_config(cfg)),
        "cross_validation": cv_metrics,
        "notes": [
            "One binary classifier is trained per emotion-vs-rest target.",
            "Combined discrete emotion prediction is argmax over specialist positive probabilities.",
            "Binary specialist metrics use a 0.5 positive-probability threshold.",
        ],
    }
    return artifact, _json_ready(metrics)


def save_artifacts(artifact: dict[str, Any], metrics: dict[str, Any], cfg: SpecialistConfig) -> None:
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
    parser.add_argument("--model-type", choices=("logistic", "sgd_logistic"), default="logistic")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--feature-mode", choices=("roi", "vertices"), default="roi")
    parser.add_argument("--vertex-csv", type=Path, default=DEFAULT_VERTEX_CSV)
    parser.add_argument("--roi-reducers", default="mean,std,mean_abs")
    parser.add_argument("--atlas-manifest", type=Path, default=DEFAULT_ATLAS_MANIFEST)
    parser.add_argument("--temporal-window-trs", type=int, default=10)
    parser.add_argument("--temporal-stride-trs", type=int, default=1)
    parser.add_argument("--temporal-reducer", choices=("mean", "median", "last"), default="mean")
    parser.add_argument("--temporal-contiguity", choices=("contiguous", "same_label"), default="contiguous")
    parser.add_argument("--temporal-allow-class-drop", action="store_true")
    parser.add_argument("--exclude-labels", default="neutral")
    parser.add_argument("--calibration", choices=("sigmoid", "none"), default="sigmoid")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument(
        "--specialists",
        default="",
        help="Comma-separated labels to train specialists for. Default uses every remaining label.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> SpecialistConfig:
    return SpecialistConfig(
        train_npz=str(args.train_npz),
        output_model=str(args.output_model),
        metrics_json=str(args.metrics_json),
        model_type=args.model_type,
        cv_splits=args.cv_splits,
        max_iter=args.max_iter,
        c_value=args.c_value,
        alpha=args.alpha,
        random_state=args.random_state,
        max_samples=args.max_samples,
        no_cv=args.no_cv,
        n_jobs=args.n_jobs,
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
        calibration=args.calibration,
        calibration_fraction=args.calibration_fraction,
        specialists=args.specialists,
    )


def main() -> None:
    cfg = _config_from_args(build_arg_parser().parse_args())
    artifact, metrics = train_model_from_npz(Path(cfg.train_npz), cfg)
    save_artifacts(artifact, metrics, cfg)


if __name__ == "__main__":
    main()
