#!/usr/bin/env python3
"""Align EEV labels with cached TRIBE feature NPZs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eevCode._config import load_eev_config, path_cfg  # noqa: E402
from scout_core.eevCode.align import (  # noqa: E402
    align_video_features,
    build_combined_matrix,
    parse_eev_csv,
    pearson_mean,
)
from scout_core.eevCode.constants import EEV_TARGET_LABELS  # noqa: E402
from scout_core.eevCode.features import load_features_npz  # noqa: E402


def _list_feature_npzs(intermediates_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in intermediates_dir.glob("*_features.npz"):
        vid = p.name.replace("_features.npz", "")
        out[vid] = p
    return out


def align_split(
    csv_path: Path,
    intermediates_dir: Path,
    *,
    lag_s: float = 0.0,
    video_ids: list[str] | None = None,
) -> dict:
    df = parse_eev_csv(csv_path)
    feature_map = _list_feature_npzs(intermediates_dir)
    if video_ids:
        feature_map = {k: v for k, v in feature_map.items() if k in set(video_ids)}

    pairs = []
    for vid, npz_path in sorted(feature_map.items()):
        df_v = df[df["video_id"] == vid]
        if df_v.empty:
            continue
        feat = load_features_npz(npz_path)
        X, Y, t_sec = align_video_features(feat, df_v, lag_s=lag_s)
        if X.shape[0] == 0:
            continue
        pairs.append((vid, X, Y, t_sec))

    combined = build_combined_matrix(pairs)
    combined["label_names"] = list(EEV_TARGET_LABELS)
    combined["lag_s"] = float(lag_s)
    return combined


def search_best_lag(
    csv_path: Path,
    intermediates_dir: Path,
    lag_grid: list[float],
    *,
    video_ids: list[str] | None = None,
) -> tuple[float, list[dict]]:
    """LOVO-ish lag search: for each lag, leave-one-video-out Pearson on mean target."""
    df = parse_eev_csv(csv_path)
    feature_map = _list_feature_npzs(intermediates_dir)
    if video_ids:
        feature_map = {k: v for k, v in feature_map.items() if k in set(video_ids)}

    per_video = {}
    for vid, npz_path in feature_map.items():
        df_v = df[df["video_id"] == vid]
        if df_v.empty:
            continue
        feat = load_features_npz(npz_path)
        per_video[vid] = (feat, df_v)

    results: list[dict] = []
    best_lag = 0.0
    best_score = -1.0
    for lag in lag_grid:
        fold_scores = []
        for holdout in per_video:
            preds_blocks = []
            true_blocks = []
            for vid, (feat, df_v) in per_video.items():
                X, Y, _t = align_video_features(feat, df_v, lag_s=lag)
                if vid == holdout:
                    continue
                if X.size:
                    preds_blocks.append(X)
                    true_blocks.append(Y)
            if not preds_blocks:
                continue
            # Lag search uses feature-label correlation proxy without model.
            X_all = np.vstack(preds_blocks)
            Y_all = np.vstack(true_blocks)
            fold_scores.append(pearson_mean(Y_all, X_all[:, : Y_all.shape[1]]))
        mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
        results.append({"lag_s": lag, "mean_proxy_r": mean_score})
        if mean_score > best_score:
            best_score = mean_score
            best_lag = lag
    return best_lag, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--lag-s", type=float, default=None)
    parser.add_argument("--search-lag", action="store_true")
    parser.add_argument("--video-ids-file", type=Path, default=None)
    parser.add_argument("--intermediates-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_eev_config(args.config)
    csv_dir = path_cfg(cfg, "csv_dir")
    intermediates_dir = args.intermediates_dir or path_cfg(cfg, "intermediates_dir")
    aligned_dir = path_cfg(cfg, "aligned_dir")
    models_dir = path_cfg(cfg, "models_dir")

    eev_cfg = cfg.get("eev") or {}
    csv_name = (eev_cfg.get("csv_files") or {}).get(args.split, f"{args.split}.csv")
    csv_path = csv_dir / csv_name

    video_ids = None
    if args.video_ids_file and args.video_ids_file.is_file():
        payload = json.loads(args.video_ids_file.read_text(encoding="utf-8"))
        video_ids = payload.get("video_ids")

    lag_s = args.lag_s if args.lag_s is not None else 0.0
    lag_results = []
    if args.search_lag:
        lag_grid = [float(x) for x in (cfg.get("alignment") or {}).get("lag_grid_s", [0.0])]
        lag_s, lag_results = search_best_lag(csv_path, intermediates_dir, lag_grid, video_ids=video_ids)
        lag_manifest = models_dir / "lag_manifest.json"
        lag_manifest.parent.mkdir(parents=True, exist_ok=True)
        lag_manifest.write_text(
            json.dumps({"best_lag_s": lag_s, "grid_results": lag_results}, indent=2),
            encoding="utf-8",
        )
        print(f"Best lag: {lag_s}s → {lag_manifest}")

    combined = align_split(csv_path, intermediates_dir, lag_s=lag_s, video_ids=video_ids)
    out_path = args.output or (aligned_dir / f"eev_aligned_{args.split}_v1.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=combined["X"],
        Y=combined["Y"],
        video_id=combined["video_id"],
        t_sec=combined["t_sec"],
        label_names=np.array(combined["label_names"]),
        lag_s=np.float32(combined["lag_s"]),
        feature_contract=np.array("eev_features_v1"),
    )
    print(f"Aligned: {out_path}  X={combined['X'].shape}  Y={combined['Y'].shape}")


if __name__ == "__main__":
    main()
