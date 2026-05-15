from __future__ import annotations

import numpy as np


def parcel_timeseries(
    preds: np.ndarray,
    vertex_parcel: np.ndarray,
    *,
    reducer: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate vertex predictions within each parcel.

    Returns
    -------
    parcel_ts : (T, P) float32
    parcel_ids : (P,) int64 unique sorted parcel ids
    """
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D, got {preds.shape}")
    T, V = preds.shape
    if vertex_parcel.shape != (V,):
        raise ValueError(f"vertex_parcel shape {vertex_parcel.shape} != ({V},)")

    parcel_ids = np.unique(vertex_parcel.astype(np.int64, copy=False))
    out = np.zeros((T, parcel_ids.size), dtype=np.float32)

    for j, pid in enumerate(parcel_ids):
        mask = vertex_parcel == pid
        block = preds[:, mask]
        if block.size == 0:
            continue
        if reducer == "mean":
            out[:, j] = block.mean(axis=1)
        elif reducer == "mean_abs":
            out[:, j] = np.abs(block).mean(axis=1)
        elif reducer == "median":
            out[:, j] = np.median(block, axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unknown reducer {reducer!r}")

    return out, parcel_ids


def network_timeseries(
    parcel_ts: np.ndarray,
    parcel_ids: np.ndarray,
    parcel_to_net: dict[int, int],
    *,
    reducer: str = "mean",
) -> tuple[np.ndarray, list[int]]:
    """
    Collapse parcels into Yeo (or other) networks using parcel_to_net.

    Parcels without mapping are skipped.
    """
    if parcel_ts.ndim != 2:
        raise ValueError(f"parcel_ts must be 2D, got {parcel_ts.shape}")

    net_ids_sorted = sorted({parcel_to_net[int(p)] for p in parcel_ids.tolist() if int(p) in parcel_to_net})
    if not net_ids_sorted:
        return np.zeros((parcel_ts.shape[0], 0), dtype=np.float32), []

    T = parcel_ts.shape[0]
    out = np.zeros((T, len(net_ids_sorted)), dtype=np.float32)

    for j, nid in enumerate(net_ids_sorted):
        cols = [
            k
            for k in range(parcel_ids.size)
            if int(parcel_ids[k]) in parcel_to_net and parcel_to_net[int(parcel_ids[k])] == nid
        ]
        if not cols:
            continue
        blk = parcel_ts[:, cols]
        if reducer == "mean":
            out[:, j] = blk.mean(axis=1)
        elif reducer == "mean_abs":
            out[:, j] = np.abs(blk).mean(axis=1)
        elif reducer == "median":
            out[:, j] = np.median(blk, axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unknown reducer {reducer!r}")

    return out, net_ids_sorted
