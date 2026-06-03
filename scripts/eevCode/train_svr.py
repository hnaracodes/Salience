#!/usr/bin/env python3
"""Train Multi-Output SVR on aligned EEV dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eevCode._config import load_eev_config, path_cfg  # noqa: E402
from scout_core.eevCode.align import pearson_mean  # noqa: E402
from scout_core.eevCode.constants import EEV_TARGET_LABELS  # noqa: E402
from scout_core.eevCode.features import feature_provenance  # noqa: E402


def load_aligned(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        Y = np.asarray(z["Y"], dtype=np.float32)
        video_id = np.asarray(z["video_id"])
    return X, Y, video_id


def make_pipeline(C: float, epsilon: float, kernel: str) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", MultiOutputRegressor(SVR(kernel=kernel, C=C, epsilon=epsilon))),
        ]
    )


def lovo_cv(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, pipeline: Pipeline) -> dict:
    logo = LeaveOneGroupOut()
    fold_rs = []
    fold_mse = []
    for train_idx, test_idx in logo.split(X, Y, groups):
        model = make_pipeline(
            C=pipeline.named_steps["svr"].estimator.C,
            epsilon=pipeline.named_steps["svr"].estimator.epsilon,
            kernel=pipeline.named_steps["svr"].estimator.kernel,
        )
        model.fit(X[train_idx], Y[train_idx])
        pred = np.clip(model.predict(X[test_idx]), 0.0, 1.0)
        fold_rs.append(pearson_mean(Y[test_idx], pred))
        fold_mse.append(float(mean_squared_error(Y[test_idx], pred)))
    return {
        "mean_pearson_r": float(np.mean(fold_rs)) if fold_rs else 0.0,
        "mean_mse": float(np.mean(fold_mse)) if fold_mse else 0.0,
        "n_folds": len(fold_rs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--aligned-npz", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_eev_config(args.config)
    aligned_dir = path_cfg(cfg, "aligned_dir")
    models_dir = path_cfg(cfg, "models_dir")
    train_cfg = cfg.get("training") or {}
    svr_cfg = train_cfg.get("svr") or {}

    aligned_path = args.aligned_npz or (aligned_dir / "eev_aligned_train_v1.npz")
    if not aligned_path.is_file():
        raise SystemExit(f"Missing aligned NPZ: {aligned_path}")

    model_name = args.model_name or train_cfg.get("model_name", "eev_svr_v1")
    model_path = models_dir / f"{model_name}.joblib"
    metrics_path = models_dir / f"{model_name}.metrics.json"
    manifest_path = models_dir / f"{model_name}.manifest.json"

    X, Y, video_id = load_aligned(aligned_path)
    valid = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X, Y, video_id = X[valid], Y[valid], video_id[valid]

    pipeline = make_pipeline(
        C=float(svr_cfg.get("C", 1.0)),
        epsilon=float(svr_cfg.get("epsilon", 0.05)),
        kernel=str(svr_cfg.get("kernel", "rbf")),
    )

    cv = lovo_cv(X, Y, video_id, pipeline)
    t0 = time.time()
    pipeline.fit(X, Y)
    fit_s = time.time() - t0

    prov = feature_provenance()
    lag_s = 0.0
    with np.load(aligned_path, allow_pickle=True) as z:
        if "lag_s" in z.files:
            lag_s = float(np.asarray(z["lag_s"]).item())

    manifest = {
        "model_name": model_name,
        "label_names": list(EEV_TARGET_LABELS),
        "feature_names": prov["feature_names"],
        "feature_contract": prov["feature_contract"],
        "lag_s": lag_s,
        "aligned_npz": str(aligned_path.relative_to(PROJECT_ROOT)),
        **prov,
    }
    metrics = {
        "created_at_unix_ms": int(time.time() * 1000),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cross_validation": cv,
        "elapsed_fit_seconds": fit_s,
        "per_target_pearson_r": {},
    }
    pred_train = np.clip(pipeline.predict(X), 0.0, 1.0)
    for j, name in enumerate(EEV_TARGET_LABELS):
        yt = Y[:, j]
        yp = pred_train[:, j]
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            metrics["per_target_pearson_r"][name] = float(np.corrcoef(yt, yp)[0, 1])
        else:
            metrics["per_target_pearson_r"][name] = 0.0

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"LOVO mean Pearson r: {cv['mean_pearson_r']:.4f}")


if __name__ == "__main__":
    main()
