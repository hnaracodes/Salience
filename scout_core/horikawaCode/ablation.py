"""Horikawa feature x label ablation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scout_core.affect_features import ALL_FEATURE_SPECS, resolve_feature_spec
from scout_core.horikawaCode.cv import evaluate_ridge_cv, iter_metric_items_in_target_order
from scout_core.horikawaCode.labels import label_names_for_target
from scout_core.horikawaCode.prepare import build_train_npz_from_manifest

LABEL_FRAMINGS = ("product_8", "dimensions_14", "raw_34", "va_2")

ABLATION_FEATURE_SETS = (
    "cortical_mean400_v1",
    "cortical_mean_std800_v1",
    "subcortical_fake8_v1",
    "subcortical_real16_v1",
    "subcortical_pca32_v1",
    "fused_schaefer400_subcortical_v1",
    "fused_v2_real_subcortical",
)


def _name_per_target(per_target: dict[str, float], label_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, (k, v) in enumerate(iter_metric_items_in_target_order(per_target)):
        name = label_names[i] if i < len(label_names) else k
        out[name] = v
    return out


def run_ablation_cell(
    manifest_clips: list[dict],
    *,
    intermediates_dir: Path,
    labels_cache: Path,
    corpus: str,
    feature_spec: str,
    label_framing: str,
    alphas: list[float],
    n_permutations: int = 100,
    n_folds: int = 5,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    if feature_spec not in ALL_FEATURE_SPECS:
        raise ValueError(f"Unknown feature spec: {feature_spec}")
    fit_pca = feature_spec == "subcortical_pca32_v1"
    data = build_train_npz_from_manifest(
        manifest_clips,
        intermediates_dir=intermediates_dir,
        labels_cache=labels_cache,
        corpus=corpus,
        target=label_framing,
        feature_spec=feature_spec,
        fit_pca_on_corpus=fit_pca,
    )
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    groups = np.asarray(data["groups"], dtype=np.int32)
    label_names = label_names_for_target(label_framing)

    metrics = evaluate_ridge_cv(
        X, y, groups, alphas,
        n_permutations=n_permutations,
        n_folds=n_folds,
        n_bootstrap=n_bootstrap,
    )
    return {
        "feature_spec": feature_spec,
        "label_framing": label_framing,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "pca_corpus_fit_warning": feature_spec == "subcortical_pca32_v1",
        "lovo_mean_r": metrics["lovo_mean_r"],
        "group_kfold_mean_r": metrics["group_kfold_mean_r"],
        "per_target_r": _name_per_target(metrics["per_target_r"], label_names),
        "permutation_null_mean": metrics["permutation"]["permutation_null_mean"],
        "permutation_null_std": metrics["permutation"]["permutation_null_std"],
        "permutation_p_value": metrics["permutation"]["permutation_p_value"],
        "in_sample_mean_r2": metrics["in_sample_mean_r2"],
        "in_sample_r2_per_target": _name_per_target(metrics["in_sample_r2_per_target"], label_names),
        "bootstrap_ci_per_target": {
            (label_names[i] if i < len(label_names) else k): v
            for i, (k, v) in enumerate(iter_metric_items_in_target_order(metrics["bootstrap_ci_per_target"]))
        },
    }


def run_ablation_matrix(
    manifest_clips: list[dict],
    *,
    intermediates_dir: Path,
    labels_cache: Path,
    corpus: str,
    alphas: list[float],
    n_permutations: int = 100,
    n_folds: int = 5,
    n_bootstrap: int = 500,
    feature_sets: tuple[str, ...] = ABLATION_FEATURE_SETS,
    label_framings: tuple[str, ...] = LABEL_FRAMINGS,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for i, (feat, framing) in enumerate(
        [(f, l) for f in feature_sets for l in label_framings], start=1
    ):
        print(f"[{i}/{len(feature_sets) * len(label_framings)}] {feat} x {framing}", flush=True)
        cell = run_ablation_cell(
            manifest_clips,
            intermediates_dir=intermediates_dir,
            labels_cache=labels_cache,
            corpus=corpus,
            feature_spec=feat,
            label_framing=framing,
            alphas=alphas,
            n_permutations=n_permutations,
            n_folds=n_folds,
            n_bootstrap=n_bootstrap,
        )
        print(
            f"  -> lovo_r={cell['lovo_mean_r']:.4f} p={cell['permutation_p_value']:.4f}",
            flush=True,
        )
        cells.append(cell)

    best = max(cells, key=lambda c: c["lovo_mean_r"])
    sig = [c for c in cells if c["permutation_p_value"] < 0.05]
    return {
        "corpus": corpus,
        "n_clips": len(manifest_clips),
        "n_permutations": n_permutations,
        "cells": cells,
        "best_cell": {
            "feature_spec": best["feature_spec"],
            "label_framing": best["label_framing"],
            "lovo_mean_r": best["lovo_mean_r"],
            "permutation_p_value": best["permutation_p_value"],
        },
        "n_significant_p005": len(sig),
    }
