#!/usr/bin/env python3
"""Evaluate EEV SVR offline (LOVO or session preds)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from activation_store import SESSIONS_DIR  # noqa: E402
from scripts.eevCode._config import load_eev_config, path_cfg  # noqa: E402
from scout_core.eevCode.align import align_video_features, parse_eev_csv, pearson_mean  # noqa: E402
from scout_core.eevCode.constants import EEV_TARGET_LABELS  # noqa: E402
from scout_core.eevCode.features import load_features_npz  # noqa: E402
from scout_core.eevCode.inference import load_model_bundle, predict_eev_track  # noqa: E402
from scripts.eevCode.train_svr import load_aligned  # noqa: E402


def evaluate_lovo(aligned_path: Path, model_path: Path) -> dict:
    X, Y, video_id = load_aligned(aligned_path)
    pipeline, _manifest = load_model_bundle(model_path)
    logo = LeaveOneGroupOut()
    fold_rs, fold_mse = [], []
    for train_idx, test_idx in logo.split(X, Y, video_id):
        from sklearn.base import clone

        fold_model = clone(pipeline)
        fold_model.fit(X[train_idx], Y[train_idx])
        pred = np.clip(fold_model.predict(X[test_idx]), 0.0, 1.0)
        fold_rs.append(pearson_mean(Y[test_idx], pred))
        fold_mse.append(float(mean_squared_error(Y[test_idx], pred)))
    return {
        "mean_pearson_r": float(np.mean(fold_rs)) if fold_rs else 0.0,
        "mean_mse": float(np.mean(fold_mse)) if fold_mse else 0.0,
        "n_folds": len(fold_rs),
    }


def evaluate_val(
    csv_path: Path,
    intermediates_dir: Path,
    model_path: Path,
    *,
    lag_s: float = 0.0,
) -> dict:
    df = parse_eev_csv(csv_path)
    pipeline, _manifest = load_model_bundle(model_path)
    rs, mses = [], []
    for npz_path in sorted(intermediates_dir.glob("*_features.npz")):
        vid = npz_path.name.replace("_features.npz", "")
        df_v = df[df["video_id"] == vid]
        if df_v.empty:
            continue
        feat = load_features_npz(npz_path)
        X, Y, _t = align_video_features(feat, df_v, lag_s=lag_s)
        if X.size == 0:
            continue
        pred = np.clip(pipeline.predict(X), 0.0, 1.0)
        rs.append(pearson_mean(Y, pred))
        mses.append(float(mean_squared_error(Y, pred)))
    return {
        "mean_pearson_r": float(np.mean(rs)) if rs else 0.0,
        "mean_mse": float(np.mean(mses)) if mses else 0.0,
        "n_videos": len(rs),
    }


def evaluate_session(session_id: str, model_path: Path, eval_dir: Path) -> Path:
    preds_path = SESSIONS_DIR / session_id / "preds.npz"
    if not preds_path.is_file():
        raise FileNotFoundError(preds_path)
    with np.load(preds_path) as z:
        preds = np.asarray(z["preds"], dtype=np.float32)
    track = predict_eev_track(preds, model_path)
    scores = np.asarray(track["scores"], dtype=np.float32)
    out_path = eval_dir / f"{session_id}_eev_scores.npz"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        scores=scores,
        label_names=np.array(track["label_names"]),
        mode=np.array(track["mode"]),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", choices=("lovo", "val", "session"), default="lovo")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--aligned-npz", type=Path, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_eev_config(args.config)
    aligned_dir = path_cfg(cfg, "aligned_dir")
    models_dir = path_cfg(cfg, "models_dir")
    csv_dir = path_cfg(cfg, "csv_dir")
    intermediates_dir = path_cfg(cfg, "intermediates_dir")
    eval_dir = path_cfg(cfg, "eval_dir")

    train_cfg = cfg.get("training") or {}
    model_name = train_cfg.get("model_name", "eev_svr_v1")
    model_path = args.model_path or (models_dir / f"{model_name}.joblib")

    if args.mode == "lovo":
        aligned_path = args.aligned_npz or (aligned_dir / "eev_aligned_train_v1.npz")
        result = evaluate_lovo(aligned_path, model_path)
    elif args.mode == "val":
        val_csv = csv_dir / (cfg.get("eev") or {}).get("csv_files", {}).get("val", "val.csv")
        lag_s = 0.0
        lag_manifest = models_dir / "lag_manifest.json"
        if lag_manifest.is_file():
            lag_s = float(json.loads(lag_manifest.read_text(encoding="utf-8")).get("best_lag_s", 0.0))
        result = evaluate_val(val_csv, intermediates_dir, model_path, lag_s=lag_s)
    else:
        if not args.session_id:
            raise SystemExit("--session-id required for mode=session")
        out_path = evaluate_session(args.session_id, model_path, eval_dir)
        result = {"session_id": args.session_id, "output": str(out_path)}

    report_path = args.report or (models_dir / "pilot_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EEV SVR evaluation report",
        "",
        f"- mode: `{args.mode}`",
        f"- model: `{model_path}`",
        "",
        "```json",
        json.dumps(result, indent=2),
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
