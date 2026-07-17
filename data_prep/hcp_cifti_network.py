from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_prep.viability_partition import NET_COLUMN_PREFIX
from scout_core.aggregate import network_timeseries, parcel_timeseries
from scout_core.constants import NETWORK_ID_TO_NAME, YEO7_NAMES


def load_parcellation_artifact(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    prov_raw = data["provenance_json"]
    if isinstance(prov_raw, np.ndarray):
        prov_raw = prov_raw.item() if prov_raw.ndim == 0 else str(prov_raw[0])
    return {
        "brain_axis_size": int(data["brain_axis_size"]),
        "cortical_mask": np.asarray(data["cortical_mask"], dtype=bool),
        "parcel_id": np.asarray(data["parcel_id"], dtype=np.int16),
        "parcel_ids": np.asarray(data["parcel_ids"], dtype=np.int16),
        "parcel_to_net_id": np.asarray(data["parcel_to_net_id"], dtype=np.int8),
        "provenance": json.loads(str(prov_raw)),
    }


def _parcel_to_net_map(artifact: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for pid, nid in zip(
        artifact["parcel_ids"].tolist(),
        artifact["parcel_to_net_id"].tolist(),
        strict=True,
    ):
        out[int(pid)] = int(nid)
    return out


def dtseries_array_to_network_features(
    ts: np.ndarray,
    artifact: dict[str, Any],
    *,
    reducer: str = "mean_abs",
) -> dict[str, float]:
    """
    Convert CIFTI dtseries [T, brain_axis] to subject-level net_* scalars.

    Matches extract_network_features convention: mean absolute network amplitude.
    """
    ts = np.asarray(ts, dtype=np.float32)
    if ts.ndim != 2:
        raise ValueError(f"dtseries must be 2D [T, brain], got {ts.shape}")

    brain_n = int(artifact["brain_axis_size"])
    if ts.shape[1] != brain_n:
        raise ValueError(f"brain axis {ts.shape[1]} != artifact {brain_n}")

    cortical_mask = artifact["cortical_mask"]
    parcel_id_full = artifact["parcel_id"]
    vertex_ts = ts[:, cortical_mask]
    vertex_parcel = parcel_id_full[cortical_mask].astype(np.int64)

    parcel_ts, parcel_ids = parcel_timeseries(vertex_ts, vertex_parcel, reducer=reducer)
    p2n = _parcel_to_net_map(artifact)
    net_ts, net_ids = network_timeseries(parcel_ts, parcel_ids, p2n, reducer=reducer)

    id_to_name = {i + 1: name for i, name in enumerate(YEO7_NAMES)}
    out: dict[str, float] = {}
    for j, nid in enumerate(net_ids):
        name = id_to_name.get(int(nid), NETWORK_ID_TO_NAME.get(int(nid), f"net_{nid}"))
        out[f"{NET_COLUMN_PREFIX}{name}"] = float(np.mean(np.abs(net_ts[:, j])))
    return out


def load_dtseries_from_bytes(data: bytes) -> np.ndarray:
    import os
    import tempfile

    import nibabel as nib

    fd, path = tempfile.mkstemp(suffix=".dtseries.nii")
    try:
        os.write(fd, data)
        os.close(fd)
        img = nib.load(path)
        arr = np.asarray(img.get_fdata(dtype=np.float32))
    finally:
        if os.path.exists(path):
            os.unlink(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"dtseries must be 2D after load, got {arr.shape}")
    # HCP CIFTI dtseries is (n_grayordinates, n_timepoints).
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr.astype(np.float32)


def aggregate_run_features(run_feats: list[dict[str, float]]) -> dict[str, float]:
    """Mean net_* across successful movie runs for one subject."""
    if not run_feats:
        return {}
    keys = run_feats[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([f[k] for f in run_feats if k in f]))
    return out


def extract_subject_features_from_s3_runs(
    run_data_list: list[bytes],
    artifact: dict[str, Any],
) -> dict[str, float]:
    run_feats: list[dict[str, float]] = []
    for blob in run_data_list:
        ts = load_dtseries_from_bytes(blob)
        run_feats.append(dtseries_array_to_network_features(ts, artifact))
    return aggregate_run_features(run_feats)
