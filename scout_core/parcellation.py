from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VertexParcellationTable:
    """Vertex-indexed parcellation aligned with Tribe preds[:, vertex_index]."""

    vertex_index: np.ndarray  # (V,) int
    parcel_id: np.ndarray  # (V,) int
    parcel_label: np.ndarray  # (V,) object str
    yeo_network_id: np.ndarray  # (V,) int, 0 if unknown
    yeo_network_name: np.ndarray  # (V,) object str
    hemisphere: np.ndarray  # (V,) object str

    @property
    def n_vertices(self) -> int:
        return int(self.vertex_index.shape[0])


def load_vertex_table(path: Path) -> VertexParcellationTable:
    """Load extended CSV or legacy two-column vertex_index,region_name."""
    vertices: list[int] = []
    parcel_ids: list[int] = []
    parcel_labels: list[str] = []
    yeo_ids: list[int] = []
    yeo_names: list[str] = []
    hemis: list[str] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        fields = {x.strip() for x in reader.fieldnames}

        for row in reader:
            vi = int(row["vertex_index"])
            vertices.append(vi)
            if "parcel_id" in fields and row.get("parcel_id") not in (None, ""):
                parcel_ids.append(int(row["parcel_id"]))
                plabel = (row.get("parcel_label") or f"p_{row['parcel_id']}").strip()
                parcel_labels.append(plabel)
                yn_raw = row.get("yeo_network_id") or ""
                yeo_ids.append(int(yn_raw) if str(yn_raw).strip() != "" else 0)
                yeo_names.append((row.get("yeo_network_name") or "").strip())
                hemis.append((row.get("hemisphere") or "unknown").strip())
            else:
                region = (row.get("region_name") or row.get("parcel_label") or "unknown").strip()
                pid = hash(region) % (2**31 - 1) or 1
                parcel_ids.append(abs(pid))
                parcel_labels.append(region)
                yeo_ids.append(0)
                yeo_names.append("")
                hemis.append("unknown")

    order = np.argsort(np.asarray(vertices, dtype=np.int64))
    vi = np.asarray(vertices, dtype=np.int64)[order]
    pid = np.asarray(parcel_ids, dtype=np.int64)[order]
    plab = np.asarray(parcel_labels, dtype=object)[order]
    yn = np.asarray(yeo_ids, dtype=np.int64)[order]
    yn_name = np.asarray(yeo_names, dtype=object)[order]
    hemi = np.asarray(hemis, dtype=object)[order]

    expected = np.arange(vi.shape[0], dtype=np.int64)
    if not np.array_equal(vi, expected):
        raise ValueError(
            f"{path}: vertex_index must be contiguous 0..{vi.shape[0] - 1} after sort"
        )

    return VertexParcellationTable(
        vertex_index=vi,
        parcel_id=pid,
        parcel_label=plab,
        yeo_network_id=yn,
        yeo_network_name=yn_name,
        hemisphere=hemi,
    )


def dense_parcel_labels(table: VertexParcellationTable, n_vertices: int) -> np.ndarray:
    if table.n_vertices != n_vertices:
        raise ValueError(
            f"Parcellation has V={table.n_vertices} but preds require V={n_vertices}"
        )
    return table.parcel_id.copy()


def parcel_to_network_map(table: VertexParcellationTable) -> dict[int, int]:
    """First row wins: parcel_id -> yeo_network_id (non-zero)."""
    out: dict[int, int] = {}
    for pid, yn in zip(table.parcel_id.tolist(), table.yeo_network_id.tolist(), strict=True):
        if int(pid) not in out and int(yn) > 0:
            out[int(pid)] = int(yn)
    return out
