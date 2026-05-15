from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def norms_table_for_parcels(parcel_ids: np.ndarray, norms: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Align norms df rows to parcel_ids order; return mean/std vectors."""
    means = []
    stds = []
    index_map = norms.set_index("parcel_id")
    for pid in parcel_ids.tolist():
        pid = int(pid)
        if pid not in index_map.index:
            raise KeyError(f"parcel_id {pid} missing from norms table")
        row = index_map.loc[pid]
        means.append(float(row["mean"]))
        std = float(row["std"])
        stds.append(max(std, 1e-8))
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def zscore_roi(parcel_ts: np.ndarray, norms: pd.DataFrame, parcel_ids: np.ndarray) -> np.ndarray:
    """Z-score each parcel column vs norm bundle."""
    m, s = norms_table_for_parcels(parcel_ids, norms)
    return (parcel_ts.astype(np.float32) - m) / s


def zscore_network(net_ts: np.ndarray, norms_net: pd.DataFrame, network_ids: list[int]) -> np.ndarray:
    means = []
    stds = []
    index_map = norms_net.set_index("yeo_network_id")
    for nid in network_ids:
        nid = int(nid)
        if nid not in index_map.index:
            raise KeyError(f"network id {nid} missing from network norms")
        row = index_map.loc[nid]
        means.append(float(row["mean"]))
        std = float(row["std"])
        stds.append(max(std, 1e-8))
    m = np.asarray(means, dtype=np.float32)
    s = np.asarray(stds, dtype=np.float32)
    return (net_ts.astype(np.float32) - m) / s


def quantile_flags(
    parcel_ts: np.ndarray,
    norms: pd.DataFrame,
    parcel_ids: np.ndarray,
    *,
    hi_q: float = 0.95,
) -> np.ndarray:
    """Boolean mask T×P: value exceeds norm bundle high quantile when column q95 exists."""
    if "q95" not in norms.columns:
        return np.zeros_like(parcel_ts, dtype=bool)
    q95s = []
    index_map = norms.set_index("parcel_id")
    for pid in parcel_ids.tolist():
        pid = int(pid)
        q95s.append(float(index_map.loc[pid]["q95"]))
    q = np.asarray(q95s, dtype=np.float32)
    return parcel_ts > q
