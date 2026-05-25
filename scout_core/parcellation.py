from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


EXPECTED_FSAVERAGE5_VERTICES = 20484
EXPECTED_FSAVERAGE5_HEMI_VERTICES = EXPECTED_FSAVERAGE5_VERTICES // 2


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

    @property
    def parcel_ids(self) -> np.ndarray:
        return np.asarray(list(dict.fromkeys(self.parcel_id.tolist())), dtype=np.int64)

    @property
    def n_parcels(self) -> int:
        return int(self.parcel_ids.shape[0])

    @property
    def parcel_labels_by_id(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for parcel_id, label in zip(self.parcel_id.tolist(), self.parcel_label.tolist(), strict=True):
            out.setdefault(int(parcel_id), str(label))
        return out

    @property
    def network_by_parcel_id(self) -> dict[int, tuple[int, str]]:
        out: dict[int, tuple[int, str]] = {}
        for parcel_id, network_id, network_name in zip(
            self.parcel_id.tolist(),
            self.yeo_network_id.tolist(),
            self.yeo_network_name.tolist(),
            strict=True,
        ):
            out.setdefault(int(parcel_id), (int(network_id), str(network_name)))
        return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parcellation_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Parcellation manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


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


def validate_vertex_table(
    table: VertexParcellationTable,
    *,
    n_vertices: int = EXPECTED_FSAVERAGE5_VERTICES,
    expected_n_parcels: int | None = None,
    require_hemispheres: bool = True,
) -> dict[str, Any]:
    """Validate the TribeV2 vertex-to-parcel table and return a compact summary."""
    if table.n_vertices != n_vertices:
        raise ValueError(f"Expected {n_vertices} vertices, got {table.n_vertices}")

    expected = np.arange(n_vertices, dtype=np.int64)
    if not np.array_equal(table.vertex_index, expected):
        raise ValueError(f"vertex_index must be contiguous 0..{n_vertices - 1}")

    if np.any(table.parcel_id <= 0):
        raise ValueError("parcel_id values must be positive integers")

    labels = np.asarray(table.parcel_label).astype(str)
    if np.any(np.char.str_len(labels) == 0):
        raise ValueError("parcel_label values must be non-empty")

    parcel_ids = table.parcel_ids
    if expected_n_parcels is not None and len(parcel_ids) != int(expected_n_parcels):
        raise ValueError(f"Expected {expected_n_parcels} parcels, got {len(parcel_ids)}")

    hemi = np.asarray(table.hemisphere).astype(str)
    hemi_counts = {name: int(np.sum(hemi == name)) for name in sorted(set(hemi.tolist()))}
    if require_hemispheres:
        if hemi_counts.get("lh", 0) != EXPECTED_FSAVERAGE5_HEMI_VERTICES:
            raise ValueError(f"Expected 10242 lh vertices, got {hemi_counts.get('lh', 0)}")
        if hemi_counts.get("rh", 0) != EXPECTED_FSAVERAGE5_HEMI_VERTICES:
            raise ValueError(f"Expected 10242 rh vertices, got {hemi_counts.get('rh', 0)}")

    network_names = sorted(
        name for name in set(np.asarray(table.yeo_network_name).astype(str).tolist()) if name
    )
    parcel_sizes = np.asarray([int(np.sum(table.parcel_id == pid)) for pid in parcel_ids], dtype=np.int64)
    return {
        "n_vertices": int(table.n_vertices),
        "n_parcels": int(len(parcel_ids)),
        "hemisphere_counts": hemi_counts,
        "network_names": network_names,
        "parcel_size_min": int(parcel_sizes.min()) if parcel_sizes.size else 0,
        "parcel_size_max": int(parcel_sizes.max()) if parcel_sizes.size else 0,
        "parcel_size_mean": float(parcel_sizes.mean()) if parcel_sizes.size else 0.0,
    }


def validate_manifest_matches_table(
    manifest: dict[str, Any],
    table: VertexParcellationTable,
    *,
    vertex_csv_path: Path | None = None,
) -> dict[str, Any]:
    if not manifest:
        return {}

    expected_vertices = int(manifest.get("n_vertices_expected") or EXPECTED_FSAVERAGE5_VERTICES)
    expected_parcels = manifest.get("n_parcels")
    summary = validate_vertex_table(
        table,
        n_vertices=expected_vertices,
        expected_n_parcels=int(expected_parcels) if expected_parcels is not None else None,
    )

    if vertex_csv_path is not None:
        expected_hash = (
            manifest.get("artifact_sha256", {}).get("vertex_regions_csv")
            if isinstance(manifest.get("artifact_sha256"), dict)
            else None
        )
        if expected_hash:
            actual_hash = file_sha256(vertex_csv_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"vertex_regions_csv sha256 mismatch: manifest={expected_hash} actual={actual_hash}"
                )
            summary["vertex_regions_csv_sha256"] = actual_hash

    summary["atlas_id"] = manifest.get("atlas_id")
    summary["mesh"] = manifest.get("mesh")
    summary["tribe_vertex_order"] = manifest.get("tribe_vertex_order")
    return summary


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
