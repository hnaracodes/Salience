from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DivergenceWindow:
    cluster_a: int
    cluster_b: int
    t_start: int
    t_end: int
    network_id: int
    network_name: str
    max_delta_z: float


def pairwise_network_delta(
    net_ts_a: np.ndarray,
    net_ts_b: np.ndarray,
    net_ids: list[int],
) -> np.ndarray:
    """Absolute difference between cluster network traces [T, N]."""
    if net_ts_a.shape != net_ts_b.shape:
        raise ValueError(f"shape mismatch {net_ts_a.shape} vs {net_ts_b.shape}")
    return np.abs(net_ts_a - net_ts_b).astype(np.float32)


def detect_divergence_windows(
    net_ts_by_cluster: dict[int, np.ndarray],
    net_ids: list[int],
    net_names: list[str],
    *,
    z_threshold: float = 1.5,
    min_duration_tr: int = 2,
) -> list[DivergenceWindow]:
    """
    Find temporal windows where cluster pairs diverge in network z-scored space.

    Input traces should already be z-scored per network.
    """
    cluster_ids = sorted(net_ts_by_cluster.keys())
    windows: list[DivergenceWindow] = []

    for i, ca in enumerate(cluster_ids):
        for cb in cluster_ids[i + 1 :]:
            delta = pairwise_network_delta(net_ts_by_cluster[ca], net_ts_by_cluster[cb], net_ids)
            T, N = delta.shape
            for j in range(N):
                col = delta[:, j]
                above = col >= z_threshold
                start: int | None = None
                for t in range(T):
                    if above[t] and start is None:
                        start = t
                    if (not above[t] or t == T - 1) and start is not None:
                        end = t if not above[t] else t
                        if end - start + 1 >= min_duration_tr:
                            windows.append(
                                DivergenceWindow(
                                    cluster_a=ca,
                                    cluster_b=cb,
                                    t_start=start,
                                    t_end=end,
                                    network_id=net_ids[j],
                                    network_name=net_names[j] if j < len(net_names) else f"net_{net_ids[j]}",
                                    max_delta_z=float(col[start : end + 1].max()),
                                )
                            )
                        start = None
    return windows
