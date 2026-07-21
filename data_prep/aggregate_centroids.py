from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def zscore_subject_parcel(ts: np.ndarray) -> np.ndarray:
    """Z-score each parcel time series for one subject."""
    ts = np.asarray(ts, dtype=np.float32)
    mu = ts.mean(axis=0, keepdims=True)
    sd = ts.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((ts - mu) / sd).astype(np.float32)


def aggregate_cluster_centroid(
    subject_arrays: list[np.ndarray],
    *,
    smooth_window: int = 0,
) -> np.ndarray:
    stack = np.stack([zscore_subject_parcel(a) for a in subject_arrays], axis=0)
    centroid = stack.mean(axis=0).astype(np.float32)
    if smooth_window > 1:
        centroid = uniform_filter1d(centroid, size=smooth_window, axis=0, mode="nearest").astype(
            np.float32
        )
    return centroid


def build_video_centroids(
    aligned_dir: Path,
    cluster_members: pd.DataFrame,
    video_id: str,
    *,
    target_key: str = "parcel_ts",
) -> dict[str, Any]:
    """Build y_target [K, T, P] for one video from aligned subject npz files."""
    members = cluster_members[~cluster_members["suppressed"].astype(bool)]
    if members.empty:
        return {"y_target": np.zeros((0, 0, 0), dtype=np.float32), "cluster_ids": np.array([], dtype=np.int64)}

    cluster_ids_sorted = sorted(members["cluster_id"].unique().tolist())
    centroids: list[np.ndarray] = []
    built_cluster_ids: list[int] = []
    provenance: list[dict[str, Any]] = []

    for cid in cluster_ids_sorted:
        subj_ids = members.loc[members["cluster_id"] == cid, "subject_id"].tolist()
        arrs: list[np.ndarray] = []
        datasets: set[str] = set()
        for sid in subj_ids:
            matches = list(aligned_dir.glob(f"*/{video_id}/{sid}.npz"))
            if not matches:
                continue
            data = np.load(matches[0])
            arrs.append(np.asarray(data[target_key], dtype=np.float32))
            datasets.add(matches[0].parts[-3])
        if not arrs:
            continue
        centroids.append(aggregate_cluster_centroid(arrs))
        built_cluster_ids.append(int(cid))
        provenance.append(
            {
                "cluster_id": int(cid),
                "n_subjects": len(arrs),
                "dataset_ids": sorted(datasets),
            }
        )

    if not centroids:
        return {"y_target": np.zeros((0, 0, 0), dtype=np.float32), "cluster_ids": np.array([], dtype=np.int64)}

    y_target = np.stack(centroids, axis=0).astype(np.float32)
    cluster_ids = np.asarray(built_cluster_ids, dtype=np.int64)
    return {
        "y_target": y_target,
        "cluster_ids": cluster_ids,
        "fps": np.float32(1.0),
        "provenance_json": json.dumps(provenance),
    }


def write_centroid_npz(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
