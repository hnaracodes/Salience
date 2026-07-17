from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def frame_parcel_chunk(
    parcel_ts: np.ndarray,
    cluster_start: int,
    cluster_end: int,
) -> dict[str, Any]:
    chunk = parcel_ts[cluster_start:cluster_end]
    return {
        "cluster_start": cluster_start,
        "cluster_end": cluster_end,
        "shape": list(chunk.shape),
        "dtype": "float32",
        "array": chunk,
    }


def build_stream_payload(
    parcel_ts: np.ndarray,
    cluster_ids: list[int],
    *,
    provenance: dict[str, Any],
    k_chunk: int = 8,
) -> dict[str, Any]:
    K, T, P = parcel_ts.shape
    chunks = []
    for start in range(0, K, k_chunk):
        end = min(start + k_chunk, K)
        chunks.append(
            {
                "cluster_start": start,
                "cluster_end": end,
                "shape": [end - start, T, P],
                "dtype": "float32",
            }
        )
    return {
        "schema_version": 2,
        "mesh": "fsaverage5",
        "target_space": "parcel",
        "n_parcels": P,
        "fps": 1.0,
        "cluster_ids": cluster_ids,
        "provenance": provenance,
        "chunks": chunks,
    }


def save_mux_parcel_npz(path: Path, parcel_ts: np.ndarray, cluster_ids: list[int], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        parcel_ts=parcel_ts.astype(np.float32),
        cluster_ids=np.asarray(cluster_ids, dtype=np.int64),
        meta_json=json.dumps(meta),
    )
