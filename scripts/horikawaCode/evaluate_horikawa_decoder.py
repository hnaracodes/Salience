"""Evaluate Horikawa decoder with ablation and permutation baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.affect_features import resolve_feature_spec, slice_fused_features
from scout_core.horikawaCode.cv import evaluate_ridge_cv, iter_metric_items_in_target_order, lovo_mean_r
from scout_core.horikawaCode.mlp import group_kfold_mean_r
from scout_core.mvpa_engine import load_decoder_bundle


def _identification_top1(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.ndim != 2 or y_true.shape[1] < 2:
        return None
    true_idx = np.argmax(y_true, axis=1)
    pred_idx = np.argmax(y_pred, axis=1)
    return float(np.mean(true_idx == pred_idx))


def _ridge_eval(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    *,
    n_permutations: int = 100,
    n_folds: int = 5,
    n_bootstrap: int = 500,
    feature_spec_name: str | None = None,
) -> dict:
    metrics = evaluate_ridge_cv(
        X, y, groups, alphas,
        n_permutations=n_permutations,
        n_folds=n_folds,
        n_bootstrap=n_bootstrap,
    )
    spec = resolve_feature_spec(feature_spec_name or "fused_schaefer400_subcortical_v1")
    if spec.include_cortical and spec.name != "fused_schaefer400_subcortical_v1":
        cortical_only = X
        cortical_r = metrics["lovo_mean_r"]
    elif spec.include_cortical:
        cortical_only = slice_fused_features(
            X, resolve_feature_spec("cortical_mean_std800_v1")
        )
        cortical_r = lovo_mean_r(cortical_only, y, groups, alphas)
    else:
        cortical_r = float("nan")
    fused_r = metrics["lovo_mean_r"]
    perm = metrics["permutation"]
    oof = metrics["lovo_oof_predictions"]
    top1 = _identification_top1(y, oof)
    return {
        "fused_lovo_mean_r": fused_r,
        "group_kfold_mean_r": metrics["group_kfold_mean_r"],
        "cortical_only_lovo_mean_r": cortical_r,
        "subcortical_lift": fused_r - cortical_r,
        "permutation_lovo_mean_r": perm["permutation_null_mean"],
        "permutation_null_std": perm["permutation_null_std"],
        "permutation_p_value": perm["permutation_p_value"],
        "per_target_r": metrics["per_target_r"],
        "bootstrap_ci_per_target": metrics["bootstrap_ci_per_target"],
        "in_sample_mean_r2": metrics["in_sample_mean_r2"],
        "identification_top1": top1,
        "pass_subcortical_lift": fused_r >= cortical_r,
        "pass_beats_permutation": perm["permutation_p_value"] < 0.05,
    }


def _mlp_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: dict) -> dict:
    mlp_cfg = cfg.get("mlp") or {}
    mlp_kw = {
        "hidden_layer_sizes": tuple(int(x) for x in mlp_cfg.get("hidden_layer_sizes", [64])),
        "alpha": float(mlp_cfg.get("alpha", 0.01)),
        "learning_rate_init": float(mlp_cfg.get("learning_rate_init", 0.001)),
        "max_iter": int(mlp_cfg.get("max_iter", 500)),
        "random_state": int(mlp_cfg.get("random_state", 42)),
    }
    cv_splits = int(mlp_cfg.get("cv_splits", 5))
    fused_r = group_kfold_mean_r(X, y, groups, n_splits=cv_splits, **mlp_kw)
    return {"group_kfold_mean_r": fused_r}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "horikawa_decoding.yaml",
    )
    parser.add_argument(
        "--compare-mlp",
        action="store_true",
        help="Also compute MLP group-kfold metrics on the same training NPZ",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic fallback when train.npz is missing (default: fail closed)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    model_id = args.model_id or str(cfg.get("model_id", "horikawa_ridge_v1"))
    alphas = [float(a) for a in cfg.get("ridge", {}).get("alphas", [0.1, 1.0, 10.0])]
    cv_cfg = cfg.get("cv") or {}
    n_permutations = int(cv_cfg.get("n_permutations", 100))
    n_bootstrap = int(cv_cfg.get("n_bootstrap", 500))
    train_raw = cfg.get("data", {}).get(
        "train_npz", "scout_data/horikawaCode/tribev2_fused/train.npz"
    )
    train_path = Path(train_raw)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path

    if not train_path.is_file():
        if not args.allow_synthetic:
            raise SystemExit(
                f"Missing training NPZ: {train_path}. "
                "Run prepare_horikawa_tribev2.py or pass --allow-synthetic for demo data."
            )
        from scout_core.horikawaCode.labels import synthetic_horikawa_dataset

        data = synthetic_horikawa_dataset(n_samples=40)
        data_source = "synthetic"
    else:
        raw = np.load(train_path, allow_pickle=True)
        data = {k: raw[k] for k in raw.files}
        data_source = str(train_path)

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    groups = np.asarray(data["groups"], dtype=np.int32)

    feature_spec_name = str(data["feature_spec"]) if "feature_spec" in data else cfg.get("feature_spec")

    ridge = _ridge_eval(
        X,
        y,
        groups,
        alphas,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        feature_spec_name=feature_spec_name,
    )

    label_names = (
        [str(x) for x in np.asarray(data.get("label_names", [])).tolist()]
        if "label_names" in data
        else []
    )
    per_target_named = {}
    for i, (k, v) in enumerate(iter_metric_items_in_target_order(ridge["per_target_r"])):
        name = label_names[i] if i < len(label_names) else k
        per_target_named[name] = v
    bootstrap_named = {}
    for i, (k, v) in enumerate(iter_metric_items_in_target_order(ridge["bootstrap_ci_per_target"])):
        name = label_names[i] if i < len(label_names) else k
        bootstrap_named[name] = v

    try:
        _, _, meta = load_decoder_bundle(model_id)
    except FileNotFoundError:
        meta = {}

    target_type = cfg.get("target_type", "product_8")
    report = {
        "model_id": model_id,
        "data_source": data_source,
        "fused_lovo_mean_r": ridge["fused_lovo_mean_r"],
        "group_kfold_mean_r": ridge["group_kfold_mean_r"],
        "cortical_only_lovo_mean_r": ridge["cortical_only_lovo_mean_r"],
        "subcortical_lift": ridge["subcortical_lift"],
        "permutation_lovo_mean_r": ridge["permutation_lovo_mean_r"],
        "permutation_null_std": ridge["permutation_null_std"],
        "permutation_p_value": ridge["permutation_p_value"],
        "in_sample_mean_r2": ridge["in_sample_mean_r2"],
        "per_target_r": per_target_named,
        "bootstrap_ci_per_target": bootstrap_named,
        "exported_cv_mean_r": meta.get("cv_mean_r"),
        "train_eval_delta": (
            float(meta.get("cv_mean_r", 0)) - ridge["fused_lovo_mean_r"]
            if meta.get("cv_mean_r") is not None
            else None
        ),
        "pass_subcortical_lift": ridge["pass_subcortical_lift"],
        "pass_beats_permutation": ridge["pass_beats_permutation"],
    }
    if target_type == "product_8" and ridge["identification_top1"] is not None:
        report["identification_top1"] = ridge["identification_top1"]

    if args.compare_mlp:
        mlp_id = str(cfg.get("model_id", "horikawa_ridge_v1")).replace("ridge", "mlp")
        mlp_metrics = _mlp_eval(X, y, groups, cfg)
        report["mlp"] = {"model_id": mlp_id, **mlp_metrics}
        try:
            _, _, mlp_meta = load_decoder_bundle(mlp_id)
            report["mlp"]["exported_cv_mean_r"] = mlp_meta.get("cv_mean_r")
        except FileNotFoundError:
            pass

    reports_dir = PROJECT_ROOT / cfg.get("data", {}).get("reports_dir", "scout_data/horikawaCode/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_name = cfg.get("data", {}).get("eval_report_name", "cv_report_v1.json")
    out_path = reports_dir / report_name
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
