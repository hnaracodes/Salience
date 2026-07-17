from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_prep.viability_partition import NET_COLUMN_PREFIX
from scout_core.aggregate import network_timeseries, parcel_timeseries
from scout_core.constants import YEO7_NAMES
from scout_core.parcellation import load_vertex_table, parcel_to_network_map


def subject_network_features_from_vertex_ts(
    vertex_ts: np.ndarray,
    vertex_csv: Path,
    *,
    reducer: str = "mean_abs",
) -> dict[str, float]:
    """
    Aggregate one subject's vertex time series to scalar network features.

    Returns dict with keys net_Vis, net_Default, ... (session mean amplitude).
    """
    table = load_vertex_table(vertex_csv)
    parcel_ts, parcel_ids = parcel_timeseries(vertex_ts, table.parcel_id, reducer=reducer)
    p2n = parcel_to_network_map(table)
    net_ts, net_ids = network_timeseries(parcel_ts, parcel_ids, p2n, reducer=reducer)
    id_to_name = {i + 1: name for i, name in enumerate(YEO7_NAMES)}
    out: dict[str, float] = {}
    for j, nid in enumerate(net_ids):
        name = id_to_name.get(int(nid), f"net_{nid}")
        out[f"{NET_COLUMN_PREFIX}{name}"] = float(np.mean(np.abs(net_ts[:, j])))
    return out


def build_subjects_csv_from_aligned_npz(
    aligned_root: Path,
    metadata_parquet: Path,
    output_csv: Path,
    *,
    vertex_csv: Path,
    target_key: str = "network_ts",
) -> pd.DataFrame:
    """
    Build M0 subjects CSV from aligned subject npz files + harmonized metadata.

    aligned_root layout:
      <dataset>/<video_id>/<subject_id>.npz
        network_ts: [T, 7]  (preferred)
        or parcel_ts / vertex_ts
    """
    meta = pd.read_parquet(metadata_parquet)
    rows: list[dict] = []

    for npz_path in sorted(aligned_root.glob("*/*/*.npz")):
        subject_id = npz_path.stem
        sub_meta = meta[meta["subject_id"] == subject_id]
        if sub_meta.empty:
            continue
        row = sub_meta.iloc[0].to_dict()
        data = np.load(npz_path)

        if target_key in data:
            ts = np.asarray(data[target_key], dtype=np.float32)
            if ts.ndim != 2:
                continue
            if target_key == "network_ts":
                for j, name in enumerate(YEO7_NAMES[: ts.shape[1]]):
                    row[f"{NET_COLUMN_PREFIX}{name}"] = float(np.mean(np.abs(ts[:, j])))
            elif target_key == "parcel_ts":
                table = load_vertex_table(vertex_csv)
                p2n = parcel_to_network_map(table)
                parcel_ids = np.asarray(data.get("parcel_ids", []), dtype=np.int64)
                if parcel_ids.size == 0:
                    continue
                net_ts, net_ids = network_timeseries(ts, parcel_ids, p2n)
                id_to_name = {i + 1: name for i, name in enumerate(YEO7_NAMES)}
                for j, nid in enumerate(net_ids):
                    name = id_to_name.get(int(nid), f"net_{nid}")
                    row[f"{NET_COLUMN_PREFIX}{name}"] = float(np.mean(np.abs(net_ts[:, j])))
            else:
                feats = subject_network_features_from_vertex_ts(ts, vertex_csv)
                row.update(feats)
        elif "vertex_ts" in data:
            feats = subject_network_features_from_vertex_ts(
                np.asarray(data["vertex_ts"], dtype=np.float32),
                vertex_csv,
            )
            row.update(feats)
        else:
            continue
        rows.append(row)

    if not rows:
        raise ValueError(f"No aligned npz files matched metadata under {aligned_root}")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def build_subjects_csv_from_preprocessed_rows(
    records: Iterable[dict],
    output_csv: Path,
) -> pd.DataFrame:
    """Write pre-built subject rows (must include metadata + net_* columns)."""
    df = pd.DataFrame(list(records))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
