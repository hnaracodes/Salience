#!/usr/bin/env python3
"""Train a coarse-to-fine NeuroEmo emotion classifier.

The default hierarchy is valence-like:

    calm -> calm
    negative -> afraid, depressed
    positive -> delighted, excited

The model first predicts the coarse group, then runs a within-group refiner for
groups that contain multiple fine emotion labels. This keeps the upstream
NeuroEmo/Nilearn surface training contract identical to the flat classifiers
while testing whether the easier valence structure helps 5-way prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_neuroemo_emotion_model import (
    DEFAULT_ATLAS_MANIFEST,
    DEFAULT_TRAIN_NPZ,
    DEFAULT_VERTEX_CSV,
    TEMPORAL_REDUCER_COMPONENTS,
    TrainConfig,
    _class_counts,
    _json_ready,
    _load_training_data,
    _make_estimator,
    _predict_probability_matrix,
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "scout_data" / "neuroEmoCode" / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "neuroemo_hierarchical_emotion_model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_MODEL_DIR / "neuroemo_hierarchical_emotion_metrics.json"
DEFAULT_HIERARCHY = "calm=calm;negative=afraid,depressed;positive=delighted,excited"


@dataclass(frozen=True)
class HierarchicalConfig:
    train_npz: str = str(DEFAULT_TRAIN_NPZ)
    output_model: str = str(DEFAULT_MODEL_PATH)
    metrics_json: str = str(DEFAULT_METRICS_PATH)
    model_type: str = "logistic_saga"
    hierarchy: str = DEFAULT_HIERARCHY
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
    roi_reducers: str = "mean,std,mean_abs"
    atlas_manifest: str = str(DEFAULT_ATLAS_MANIFEST)
    temporal_window_trs: int = 10
    temporal_stride_trs: int = 1
    temporal_reducer: str = "mean"
    temporal_contiguity: str = "contiguous"
    temporal_allow_class_drop: bool = False
    temporal_allow_rewindow: bool = False
    exclude_labels: str = "neutral"
    allow_unverified_vertex_equivalence: bool = False


def _base_train_config(cfg: HierarchicalConfig) -> TrainConfig:
    return TrainConfig(
        train_npz=cfg.train_npz,
        output_model=cfg.output_model,
        metrics_json=cfg.metrics_json,
        model_type=cfg.model_type,
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
        temporal_allow_rewindow=cfg.temporal_allow_rewindow,
        exclude_labels=cfg.exclude_labels,
        allow_unverified_vertex_equivalence=cfg.allow_unverified_vertex_equivalence,
    )


def _parse_hierarchy(raw: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for group in raw.split(";"):
        item = group.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Hierarchy groups must look like group=label1,label2; got {item!r}")
        group_name, labels_raw = item.split("=", maxsplit=1)
        group_name = group_name.strip()
        labels = [label.strip() for label in labels_raw.replace("|", ",").split(",") if label.strip()]
        if not group_name or not labels:
            raise ValueError(f"Invalid hierarchy group: {item!r}")
        if group_name in mapping:
            raise ValueError(f"Duplicate hierarchy group: {group_name!r}")
        mapping[group_name] = labels
    if not mapping:
        raise ValueError("--hierarchy must define at least one group")
    return mapping


def _hierarchy_index(labels: np.ndarray, hierarchy: dict[str, list[str]]) -> dict[str, Any]:
    label_names = labels.astype(str).tolist()
    label_to_id = {label: i for i, label in enumerate(label_names)}
    old_to_group: dict[int, int] = {}
    group_to_label_ids: list[list[int]] = []
    group_names = list(hierarchy)

    for group_id, group_name in enumerate(group_names):
        ids: list[int] = []
        for label in hierarchy[group_name]:
            if label not in label_to_id:
                raise ValueError(f"Hierarchy references {label!r}, but available labels are {label_names}")
            label_id = label_to_id[label]
            if label_id in old_to_group:
                raise ValueError(f"Label {label!r} appears in more than one hierarchy group")
            old_to_group[label_id] = group_id
            ids.append(label_id)
        group_to_label_ids.append(ids)

    missing = [label for label, label_id in label_to_id.items() if label_id not in old_to_group]
    if missing:
        raise ValueError(f"Hierarchy does not assign all labels: {missing}")

    return {
        "group_names": group_names,
        "group_to_label_ids": group_to_label_ids,
        "old_to_group": old_to_group,
        "label_names": label_names,
    }


def _coarse_labels(y: np.ndarray, old_to_group: dict[int, int]) -> np.ndarray:
    return np.asarray([old_to_group[int(label_id)] for label_id in y], dtype=np.int64)


def _one_hot(pred: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((pred.shape[0], n_classes), dtype=np.float64)
    out[np.arange(pred.shape[0]), pred.astype(int)] = 1.0
    return out


def _fit_hierarchy(X: np.ndarray, y: np.ndarray, cfg: HierarchicalConfig, hidx: dict[str, Any]) -> dict[str, Any]:
    base_cfg = _base_train_config(cfg)
    group_y = _coarse_labels(y, hidx["old_to_group"])
    coarse = _make_estimator(base_cfg)
    coarse.fit(X, group_y)

    refiners: dict[str, Any] = {}
    refiner_summaries: dict[str, Any] = {}
    for group_id, group_name in enumerate(hidx["group_names"]):
        label_ids = hidx["group_to_label_ids"][group_id]
        if len(label_ids) == 1:
            refiner_summaries[group_name] = {
                "kind": "identity",
                "labels": [hidx["label_names"][label_ids[0]]],
                "n_samples": int(np.sum(group_y == group_id)),
            }
            continue

        idx = np.flatnonzero(group_y == group_id)
        local_by_global = {label_id: local_id for local_id, label_id in enumerate(label_ids)}
        y_local = np.asarray([local_by_global[int(label_id)] for label_id in y[idx]], dtype=np.int64)
        if len(np.unique(y_local)) < 2:
            refiner_summaries[group_name] = {
                "kind": "majority_fallback",
                "labels": [hidx["label_names"][label_id] for label_id in label_ids],
                "majority_label_id": int(label_ids[int(np.bincount(y_local).argmax())]),
                "n_samples": int(idx.size),
            }
            continue
        refiner = _make_estimator(base_cfg)
        refiner.fit(X[idx], y_local)
        refiners[group_name] = refiner
        refiner_summaries[group_name] = {
            "kind": "estimator",
            "labels": [hidx["label_names"][label_id] for label_id in label_ids],
            "n_samples": int(idx.size),
        }

    return {"coarse": coarse, "refiners": refiners, "refiner_summaries": refiner_summaries}


def _predict_hierarchy(bundle: dict[str, Any], X: np.ndarray, hidx: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    coarse = bundle["coarse"]
    group_names = hidx["group_names"]
    group_to_label_ids = hidx["group_to_label_ids"]
    n_labels = len(hidx["label_names"])

    group_pred = np.asarray(coarse.predict(X), dtype=np.int64)
    group_proba, _ = _predict_probability_matrix(coarse, X, len(group_names))
    if group_proba is None:
        group_proba = _one_hot(group_pred, len(group_names))

    y_pred = np.zeros((X.shape[0],), dtype=np.int64)
    proba = np.zeros((X.shape[0], n_labels), dtype=np.float64)
    for group_id, group_name in enumerate(group_names):
        sample_idx = np.flatnonzero(group_pred == group_id)
        label_ids = group_to_label_ids[group_id]
        group_mass = group_proba[:, group_id]
        if len(label_ids) == 1:
            y_pred[sample_idx] = int(label_ids[0])
            proba[:, int(label_ids[0])] = group_mass
            continue

        refiner = bundle["refiners"].get(group_name)
        if refiner is None:
            fallback_label = int(label_ids[0])
            y_pred[sample_idx] = fallback_label
            proba[:, fallback_label] = group_mass
            continue

        if sample_idx.size:
            local_pred = np.asarray(refiner.predict(X[sample_idx]), dtype=np.int64)
            y_pred[sample_idx] = np.asarray([label_ids[int(local_id)] for local_id in local_pred], dtype=np.int64)

        local_proba, _ = _predict_probability_matrix(refiner, X, len(label_ids))
        if local_proba is None:
            local_all_pred = np.asarray(refiner.predict(X), dtype=np.int64)
            local_proba = _one_hot(local_all_pred, len(label_ids))
        for local_id, label_id in enumerate(label_ids):
            proba[:, int(label_id)] = group_mass * local_proba[:, local_id]

    row_sums = proba.sum(axis=1, keepdims=True)
    empty = row_sums[:, 0] <= 1e-12
    if np.any(empty):
        proba[empty] = 1.0 / n_labels
        row_sums = proba.sum(axis=1, keepdims=True)
    return y_pred, proba / row_sums


def _evaluate_group_cv(data: Any, cfg: HierarchicalConfig, hidx: dict[str, Any]) -> dict[str, Any]:
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
    label_ids = np.arange(len(data.labels), dtype=np.int64)

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(data.X, data.y, groups=data.subject),
        start=1,
    ):
        fold_start = time.time()
        bundle = _fit_hierarchy(data.X[train_idx], data.y[train_idx], cfg, hidx)
        y_pred, proba = _predict_hierarchy(bundle, data.X[test_idx], hidx)
        y_true = data.y[test_idx]
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        proba_all.append(proba)
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_subjects": sorted(np.unique(data.subject[train_idx]).tolist()),
                "test_subjects": sorted(np.unique(data.subject[test_idx]).tolist()),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "log_loss": float(log_loss(y_true, proba, labels=label_ids)),
                "elapsed_seconds": round(time.time() - fold_start, 3),
            }
        )

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)
    proba_concat = np.concatenate(proba_all)
    return {
        "skipped": False,
        "n_splits": n_splits,
        "folds": fold_rows,
        "mean_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
        "std_accuracy": float(np.std([row["accuracy"] for row in fold_rows])),
        "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        "std_balanced_accuracy": float(np.std([row["balanced_accuracy"] for row in fold_rows])),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])),
        "std_macro_f1": float(np.std([row["macro_f1"] for row in fold_rows])),
        "mean_log_loss": float(np.mean([row["log_loss"] for row in fold_rows])),
        "pooled_accuracy": float(accuracy_score(y_true_concat, y_pred_concat)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true_concat, y_pred_concat)),
        "pooled_macro_f1": float(f1_score(y_true_concat, y_pred_concat, average="macro", zero_division=0)),
        "pooled_log_loss": float(log_loss(y_true_concat, proba_concat, labels=label_ids)),
        "confusion_matrix": confusion_matrix(y_true_concat, y_pred_concat, labels=label_ids).tolist(),
        "classification_report": classification_report(
            y_true_concat,
            y_pred_concat,
            labels=label_ids,
            target_names=data.labels.tolist(),
            output_dict=True,
            zero_division=0,
        ),
    }


def train_model_from_npz(train_npz: Path, cfg: HierarchicalConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    from scout_core.neuroEmoCode.roi_features import summarise_roi_feature_spec

    data = _load_training_data(train_npz, _base_train_config(cfg))
    hierarchy = _parse_hierarchy(cfg.hierarchy)
    hidx = _hierarchy_index(data.labels, hierarchy)

    print(f"Loaded {train_npz}")
    print(f"  hierarchy: {hierarchy}")
    print(f"  model type: {cfg.model_type}")
    print(f"  X flattened shape: {data.X.shape}")
    print(f"  class counts: {_class_counts(data.y, data.labels)}")

    cv_metrics = {"skipped": True, "reason": "--no-cv"} if cfg.no_cv else _evaluate_group_cv(data, cfg, hidx)

    print("Training final coarse-to-fine model on all samples ...")
    start = time.time()
    final_bundle = _fit_hierarchy(data.X, data.y, cfg, hidx)
    elapsed = time.time() - start

    roi_spec = (
        summarise_roi_feature_spec(data.roi_spec, include_vertex_map=True)
        if data.roi_spec is not None
        else None
    )
    artifact = {
        "model_type": "hierarchical",
        "base_model_type": cfg.model_type,
        "hierarchy": hierarchy,
        "labels": data.labels.tolist(),
        "input_feature_shape": list(data.input_shape),
        "feature_shape": list(data.source_shape),
        "flattened_n_features": int(data.X.shape[1]),
        "feature_mode": data.feature_mode,
        "roi_reducers": list(data.roi_spec.reducers) if data.roi_spec is not None else None,
        "roi_spec": roi_spec,
        "config": asdict(cfg),
        "base_training_config": asdict(_base_train_config(cfg)),
        "bundle": final_bundle,
        "training_metadata": data.metadata,
    }

    metrics = {
        "created_at_unix_ms": int(time.time() * 1000),
        "elapsed_final_fit_seconds": round(elapsed, 3),
        "model_type": "hierarchical",
        "base_model_type": cfg.model_type,
        "hierarchy": hierarchy,
        "refiner_summaries": final_bundle["refiner_summaries"],
        "n_samples": int(data.X.shape[0]),
        "n_features": int(data.X.shape[1]),
        "input_feature_shape": list(data.input_shape),
        "feature_shape": list(data.source_shape),
        "feature_mode": data.feature_mode,
        "roi_reducers": list(data.roi_spec.reducers) if data.roi_spec is not None else None,
        "roi_spec": summarise_roi_feature_spec(data.roi_spec) if data.roi_spec is not None else None,
        "labels": data.labels.tolist(),
        "subjects": sorted(np.unique(data.subject).tolist()),
        "class_counts": _class_counts(data.y, data.labels),
        "label_filter": data.metadata.get("label_filter"),
        "temporal_aggregation": data.metadata.get("temporal_aggregation"),
        "config": asdict(cfg),
        "base_training_config": asdict(_base_train_config(cfg)),
        "cross_validation": cv_metrics,
        "notes": [
            "Coarse-to-fine hierarchy predicts a broad affective group before within-group emotion refinement.",
            "The upstream NeuroEmo/Nilearn surface, ROI, temporal, and subject-held-out contracts match the flat baselines.",
        ],
    }
    return artifact, _json_ready(metrics)


def save_artifacts(artifact: dict[str, Any], metrics: dict[str, Any], cfg: HierarchicalConfig) -> None:
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
    parser.add_argument("--model-type", choices=("sgd_logistic", "logistic_saga", "linear_svc"), default="logistic_saga")
    parser.add_argument("--hierarchy", default=DEFAULT_HIERARCHY)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--c-value", type=float, default=1.0)
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
    parser.add_argument("--temporal-reducer", choices=tuple(TEMPORAL_REDUCER_COMPONENTS), default="mean")
    parser.add_argument("--temporal-contiguity", choices=("contiguous", "same_label"), default="contiguous")
    parser.add_argument("--temporal-allow-class-drop", action="store_true")
    parser.add_argument("--temporal-allow-rewindow", action="store_true")
    parser.add_argument("--exclude-labels", default="neutral")
    parser.add_argument("--allow-unverified-vertex-equivalence", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> HierarchicalConfig:
    return HierarchicalConfig(
        train_npz=str(args.train_npz),
        output_model=str(args.output_model),
        metrics_json=str(args.metrics_json),
        model_type=args.model_type,
        hierarchy=args.hierarchy,
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
        roi_reducers=args.roi_reducers,
        atlas_manifest=str(args.atlas_manifest) if args.atlas_manifest else "",
        temporal_window_trs=args.temporal_window_trs,
        temporal_stride_trs=args.temporal_stride_trs,
        temporal_reducer=args.temporal_reducer,
        temporal_contiguity=args.temporal_contiguity,
        temporal_allow_class_drop=args.temporal_allow_class_drop,
        temporal_allow_rewindow=args.temporal_allow_rewindow,
        exclude_labels=args.exclude_labels,
        allow_unverified_vertex_equivalence=args.allow_unverified_vertex_equivalence,
    )


def main() -> None:
    cfg = _config_from_args(build_arg_parser().parse_args())
    artifact, metrics = train_model_from_npz(Path(cfg.train_npz), cfg)
    save_artifacts(artifact, metrics, cfg)


if __name__ == "__main__":
    main()
