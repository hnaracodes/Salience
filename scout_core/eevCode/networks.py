"""Network vertex masks for EEV feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scout_core.dual_track import _network_name_mask
from scout_core.eevCode.constants import EEV_NETWORK_NAMES
from scout_core.parcellation import load_vertex_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERTEX_CSV = PROJECT_ROOT / "configs" / "vertex_regions.csv"


@dataclass(frozen=True)
class NetworkMasks:
    names: tuple[str, ...]
    indices: tuple[np.ndarray, ...]

    def amplitudes(self, preds: np.ndarray) -> np.ndarray:
        """Per-TR mean absolute activation per network. Shape (T, n_networks)."""
        preds = np.asarray(preds, dtype=np.float32)
        if preds.ndim != 2:
            raise ValueError(f"preds must be 2D, got {preds.shape}")
        out = np.zeros((preds.shape[0], len(self.indices)), dtype=np.float32)
        for j, idx in enumerate(self.indices):
            if idx.size == 0:
                continue
            out[:, j] = np.abs(preds[:, idx]).mean(axis=1)
        return out


def load_network_masks(
    network_names: tuple[str, ...] = EEV_NETWORK_NAMES,
    *,
    vertex_csv: Path | None = None,
) -> NetworkMasks:
    path = vertex_csv or DEFAULT_VERTEX_CSV
    table = load_vertex_table(path)
    names_arr = np.asarray(table.yeo_network_name).astype(str)
    indices: list[np.ndarray] = []
    for name in network_names:
        mask = _network_name_mask(names_arr, name)
        indices.append(table.vertex_index[mask].astype(np.int64, copy=False))
    return NetworkMasks(names=network_names, indices=tuple(indices))
