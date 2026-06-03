"""Temporal alignment of EEV labels to TRIBE feature matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scout_core.eevCode.constants import EEV_ALL_EXPRESSION_LABELS, EEV_TARGET_LABELS

# EEV public CSV uses microseconds in the official README.
TIMESTAMP_UNIT = "microseconds"
TIMESTAMP_SCALE = 1e6


def parse_eev_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    vid_col = (
        cols.get("video id")
        or cols.get("video_id")
        or cols.get("videoid")
        or cols.get("youtube id")
    )
    ts_col = cols.get("timestamp (microseconds)") or cols.get("timestamp (milliseconds)")
    if ts_col is None:
        for c in df.columns:
            if "timestamp" in c.lower():
                ts_col = c
                break
    if vid_col is None or ts_col is None:
        raise ValueError(f"Could not find video/timestamp columns in {path}")

    out = df.rename(columns={vid_col: "video_id", ts_col: "timestamp_raw"})
    out["video_id"] = out["video_id"].astype(str)
    unit = "microseconds" if "micro" in ts_col.lower() else "milliseconds"
    scale = 1e6 if unit == "microseconds" else 1e3
    out["t_sec"] = out["timestamp_raw"].astype(np.float64) / scale

    for label in EEV_ALL_EXPRESSION_LABELS:
        if label in out.columns:
            out[label] = pd.to_numeric(out[label], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out


def downsample_labels_to_grid(
    df_video: pd.DataFrame,
    t_grid: np.ndarray,
    label_names: tuple[str, ...] = EEV_TARGET_LABELS,
) -> np.ndarray:
    """Interpolate EEV 6 Hz labels onto 1 Hz TRIBE grid. Returns (len(t_grid), n_labels)."""
    t_grid = np.asarray(t_grid, dtype=np.float64)
    src_t = df_video["t_sec"].to_numpy(dtype=np.float64)
    if src_t.size == 0:
        return np.zeros((t_grid.size, len(label_names)), dtype=np.float32)
    order = np.argsort(src_t)
    src_t = src_t[order]
    Y = np.zeros((t_grid.size, len(label_names)), dtype=np.float32)
    for j, name in enumerate(label_names):
        if name not in df_video.columns:
            continue
        src_y = df_video[name].to_numpy(dtype=np.float64)[order]
        Y[:, j] = np.interp(t_grid, src_t, src_y, left=src_y[0], right=src_y[-1]).astype(np.float32)
    return np.clip(Y, 0.0, 1.0)


def apply_lag_features(X: np.ndarray, Y: np.ndarray, lag_s: float, tr_hz: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Shift features earlier relative to labels: pair X[t] with Y[t + lag]."""
    lag_trs = int(round(float(lag_s) * tr_hz))
    if lag_trs <= 0:
        return X, Y
    if lag_trs >= X.shape[0]:
        raise ValueError(f"lag {lag_s}s exceeds series length")
    return X[:-lag_trs], Y[lag_trs:]


def align_video_features(
    feature_npz: dict[str, Any],
    df_video: pd.DataFrame,
    *,
    label_names: tuple[str, ...] = EEV_TARGET_LABELS,
    lag_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, t_sec aligned for one video."""
    X_full = np.asarray(feature_npz["X_feat"], dtype=np.float32)
    t_sec = np.asarray(feature_npz["t_sec"], dtype=np.float32)
    tr_hz = float(np.asarray(feature_npz.get("tr_hz", 1.0)).item())

    # Drop rows with NaN (coherence warmup).
    valid = np.all(np.isfinite(X_full), axis=1)
    X = X_full[valid]
    t_sec = t_sec[valid]
    if X.size == 0:
        return X, np.zeros((0, len(label_names)), dtype=np.float32), t_sec

    Y = downsample_labels_to_grid(df_video, t_sec, label_names=label_names)
    if lag_s:
        X, Y = apply_lag_features(X, Y, lag_s, tr_hz=tr_hz)
        t_sec = t_sec[: X.shape[0]]
    return X, Y, t_sec


def build_combined_matrix(
    pairs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Stack per-video aligned blocks."""
    if not pairs:
        raise ValueError("No aligned video pairs")
    X_parts, Y_parts, vid_parts, t_parts = [], [], [], []
    for vid, X, Y, t_sec in pairs:
        X_parts.append(X)
        Y_parts.append(Y)
        vid_parts.extend([vid] * X.shape[0])
        t_parts.append(t_sec)
    return {
        "X": np.vstack(X_parts).astype(np.float32),
        "Y": np.vstack(Y_parts).astype(np.float32),
        "video_id": np.array(vid_parts, dtype=object),
        "t_sec": np.concatenate(t_parts).astype(np.float32),
    }


def pearson_mean(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    rs = []
    for j in range(Y_true.shape[1]):
        a = Y_true[:, j]
        b = Y_pred[:, j]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rs)) if rs else 0.0
