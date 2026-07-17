"""Fast ablation subset using frozen ready_all NPZ (+ one fused_v2 rebuild).

Feature sets: cortical_mean_std800_v1, fused_schaefer400_subcortical_v1, fused_v2_real_subcortical
Label framings: va_2, dimensions_14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.affect_features import FEATURE_SPEC_FUSED_V2
from scout_core.horikawaCode.constants import HORIKAWA_14_DIMENSIONS
from scout_core.horikawaCode.cv import evaluate_ridge_cv, iter_metric_items_in_target_order
from scout_core.horikawaCode.labels import DEFAULT_LABEL_CACHE, label_names_for_target
from scout_core.horikawaCode.prepare import build_train_npz_from_manifest


def _name_per_target(per_target: dict, label_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, (k, v) in enumerate(iter_metric_items_in_target_order(per_target)):
        name = label_names[i] if i < len(label_names) else str(k)
        out[name] = float(v)
    return out


def _y_for_framing(y_dims: np.ndarray, framing: str) -> tuple[np.ndarray, list[str]]:
    if framing == "dimensions_14":
        return y_dims, list(HORIKAWA_14_DIMENSIONS)
    if framing == "va_2":
        idx_v = HORIKAWA_14_DIMENSIONS.index("valence")
        idx_a = HORIKAWA_14_DIMENSIONS.index("arousal")
        return y_dims[:, [idx_v, idx_a]], ["valence", "arousal"]
    raise ValueError(framing)


def _eval_cell(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    feature_spec: str,
    label_framing: str,
    alphas: list[float],
    n_permutations: int,
    n_bootstrap: int,
) -> dict:
    label_names = label_names_for_target(label_framing)
    metrics = evaluate_ridge_cv(
        X,
        y,
        groups,
        alphas,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
    )
    return {
        "feature_spec": feature_spec,
        "label_framing": label_framing,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "lovo_mean_r": metrics["lovo_mean_r"],
        "group_kfold_mean_r": metrics["group_kfold_mean_r"],
        "per_target_r": _name_per_target(metrics["per_target_r"], label_names),
        "permutation_null_mean": metrics["permutation"]["permutation_null_mean"],
        "permutation_null_std": metrics["permutation"]["permutation_null_std"],
        "permutation_p_value": metrics["permutation"]["permutation_p_value"],
        "in_sample_mean_r2": metrics["in_sample_mean_r2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-npz",
        type=Path,
        default=PROJECT_ROOT / "scout_data/horikawaCode/tribev2_fused/train.npz",
    )
    parser.add_argument(
        "--train-dims-npz",
        type=Path,
        default=PROJECT_ROOT / "scout_data/horikawaCode/tribev2_fused/train_dims.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "scout_data/horikawaCode/manifests/ready_all.json",
    )
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data/horikawaCode/intermediates_modal",
    )
    parser.add_argument("--labels-cache", type=Path, default=DEFAULT_LABEL_CACHE)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/horikawa_decoding.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scout_data/horikawaCode/reports/ablation_matrix.json",
    )
    parser.add_argument("--n-permutations", type=int, default=25)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument(
        "--fused-v2-cache",
        type=Path,
        default=PROJECT_ROOT / "scout_data/horikawaCode/tribev2_fused/train_fused_v2.npz",
    )
    parser.add_argument("--skip-fused-v2", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    alphas = [float(a) for a in cfg.get("ridge", {}).get("alphas", [0.1, 1.0, 10.0])]

    fused = np.load(args.train_npz, allow_pickle=True)
    dims = np.load(args.train_dims_npz, allow_pickle=True)
    X_fused = np.asarray(fused["X"], dtype=np.float32)
    y_dims = np.asarray(dims["y"], dtype=np.float32)
    groups = np.asarray(fused["groups"], dtype=np.int32)
    if X_fused.shape[0] != y_dims.shape[0]:
        raise SystemExit(
            f"Row mismatch: fused X={X_fused.shape[0]} dims y={y_dims.shape[0]}"
        )

    n_cort = 800
    feature_mats = {
        "cortical_mean_std800_v1": X_fused[:, :n_cort],
        "fused_schaefer400_subcortical_v1": X_fused,
    }

    if not args.skip_fused_v2:
        if args.fused_v2_cache.is_file():
            print(f"Loading fused_v2 cache {args.fused_v2_cache}", flush=True)
            X_v2 = np.asarray(np.load(args.fused_v2_cache, allow_pickle=True)["X"], dtype=np.float32)
        else:
            print("Building fused_v2 features from intermediates (one-time)...", flush=True)
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            clips = list(payload.get("clips") or [])
            data = build_train_npz_from_manifest(
                clips,
                intermediates_dir=args.intermediates_dir,
                labels_cache=args.labels_cache,
                corpus=str(payload.get("corpus", "ready_all")),
                target="dimensions_14",
                feature_spec=FEATURE_SPEC_FUSED_V2,
            )
            X_v2 = np.asarray(data["X"], dtype=np.float32)
            args.fused_v2_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.fused_v2_cache,
                X=X_v2,
                stimulus_id=data["stimulus_id"],
                feature_spec=np.asarray(FEATURE_SPEC_FUSED_V2),
                corpus=np.asarray(str(payload.get("corpus", "ready_all"))),
            )
            print(f"Cached fused_v2 -> {args.fused_v2_cache} X={X_v2.shape}", flush=True)
        if X_v2.shape[0] != X_fused.shape[0]:
            raise SystemExit(f"fused_v2 rows {X_v2.shape[0]} != fused rows {X_fused.shape[0]}")
        feature_mats["fused_v2_real_subcortical"] = X_v2

    framings = ("va_2", "dimensions_14")
    cells: list[dict] = []
    total = len(feature_mats) * len(framings)
    i = 0
    for feat_name, X in feature_mats.items():
        for framing in framings:
            i += 1
            y, _ = _y_for_framing(y_dims, framing)
            print(f"[{i}/{total}] {feat_name} x {framing} X={X.shape} y={y.shape}", flush=True)
            cell = _eval_cell(
                X,
                y,
                groups,
                feature_spec=feat_name,
                label_framing=framing,
                alphas=alphas,
                n_permutations=args.n_permutations,
                n_bootstrap=args.n_bootstrap,
            )
            cells.append(cell)
            print(
                f"  lovo_r={cell['lovo_mean_r']:.4f} p={cell['permutation_p_value']:.4f}",
                flush=True,
            )

    best = max(cells, key=lambda c: c["lovo_mean_r"])
    sig = [c for c in cells if c["permutation_p_value"] < 0.05]
    matrix = {
        "corpus": str(fused["corpus"]) if "corpus" in fused.files else "ready_all",
        "n_clips": int(X_fused.shape[0]),
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "mode": "frozen_npz_subset",
        "cells": cells,
        "best_cell": {
            "feature_spec": best["feature_spec"],
            "label_framing": best["label_framing"],
            "lovo_mean_r": best["lovo_mean_r"],
            "permutation_p_value": best["permutation_p_value"],
        },
        "n_significant_p005": len(sig),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(json.dumps({"best_cell": matrix["best_cell"], "n_significant_p005": matrix["n_significant_p005"]}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
