from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from scout_core.aggregate import network_timeseries, parcel_timeseries
from scout_core.parcellation import load_vertex_table, parcel_to_network_map

HEMO_LAG_S = 5.0

DATASET_TR = {
    "hcp_7t": 1.0,
    "cneuromod": 1.49,
    "nndb": 2.0,
    "camcan": 0.74,
    "forrest": 2.0,
}


def resample_to_1hz(
    y: np.ndarray,
    *,
    source_tr: float,
    video_duration_s: float | None = None,
    hemodynamic_lag_s: float = HEMO_LAG_S,
) -> np.ndarray:
    """Linear interpolation to 1 Hz with optional hemodynamic shift."""
    y = np.asarray(y, dtype=np.float32)
    n_source = y.shape[0]
    source_t = np.arange(n_source, dtype=np.float64) * source_tr
    if video_duration_s is None:
        video_duration_s = float(source_t[-1]) if n_source else 0.0
    target_t = np.arange(0.0, max(0.0, video_duration_s - hemodynamic_lag_s), 1.0)
    if target_t.size == 0 or n_source < 2:
        return np.zeros((0, y.shape[1]), dtype=np.float32)
    interp = interp1d(
        source_t,
        y,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interp(target_t + hemodynamic_lag_s).astype(np.float32)


def parcellate_vertices(vertex_ts: np.ndarray, vertex_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    table = load_vertex_table(vertex_csv)
    parcel_ts, parcel_ids = parcel_timeseries(vertex_ts, table.parcel_id)
    return parcel_ts, parcel_ids


def derive_network_ts(
    parcel_ts: np.ndarray,
    parcel_ids: np.ndarray,
    parcel_to_net: dict[int, int],
) -> tuple[np.ndarray, list[int]]:
    return network_timeseries(parcel_ts, parcel_ids, parcel_to_net)


def align_subject_npz(
    vertex_ts: np.ndarray,
    *,
    dataset: str,
    video_duration_s: float | None,
    project_root: Path,
) -> dict[str, np.ndarray]:
    tr = DATASET_TR.get(dataset, 1.0)
    y_1hz = resample_to_1hz(vertex_ts, source_tr=tr, video_duration_s=video_duration_s)
    if y_1hz.size == 0:
        return {
            "parcel_ts": np.zeros((0, 400), dtype=np.float32),
            "network_ts": np.zeros((0, 7), dtype=np.float32),
            "vertex_ts": y_1hz,
        }

    parcel_ts, parcel_ids = parcellate_vertices(y_1hz, project_root / "configs" / "vertex_regions.csv")
    table = load_vertex_table(project_root / "configs" / "vertex_regions.csv")
    p2n = parcel_to_network_map(table)
    net_ts, net_ids = derive_network_ts(parcel_ts, parcel_ids, p2n)
    return {
        "parcel_ts": parcel_ts.astype(np.float32),
        "network_ts": net_ts.astype(np.float32),
        "vertex_ts": y_1hz.astype(np.float32),
        "parcel_ids": parcel_ids.astype(np.int64),
        "network_ids": np.asarray(net_ids, dtype=np.int64),
    }


def write_aligned_subject(
    output_path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
