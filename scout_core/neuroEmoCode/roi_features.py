from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scout_core.parcellation import (
    VertexParcellationTable,
    file_sha256,
    load_parcellation_manifest,
    load_vertex_table,
    validate_manifest_matches_table,
    validate_vertex_table,
)


SUPPORTED_REDUCERS = ("mean", "std", "mean_abs", "max_abs")


@dataclass(frozen=True)
class RoiFeatureSpec:
    vertex_csv: str
    vertex_csv_sha256: str
    manifest_path: str | None
    manifest_sha256: str | None
    atlas_id: str | None
    mesh: str | None
    vertex_order: str | None
    n_vertices: int
    parcel_ids: np.ndarray
    parcel_labels: np.ndarray
    parcel_id_by_vertex: np.ndarray
    reducers: tuple[str, ...]
    feature_names: tuple[str, ...]
    validation_summary: dict[str, Any]
    source_metadata: dict[str, Any]

    @property
    def n_rois(self) -> int:
        return int(self.parcel_ids.shape[0])

    @property
    def n_features(self) -> int:
        return int(len(self.feature_names))


def parse_reducers(raw: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",")]
    else:
        parts = [str(item).strip() for item in raw]
    reducers = tuple(item for item in parts if item)
    if not reducers:
        raise ValueError("At least one ROI reducer is required.")
    unknown = sorted(set(reducers) - set(SUPPORTED_REDUCERS))
    if unknown:
        raise ValueError(f"Unsupported ROI reducer(s): {unknown}. Supported: {SUPPORTED_REDUCERS}")
    return reducers


def _ordered_parcel_metadata(table: VertexParcellationTable) -> tuple[np.ndarray, np.ndarray]:
    parcel_ids = table.parcel_ids
    labels_by_id = table.parcel_labels_by_id
    labels = np.asarray([labels_by_id[int(pid)] for pid in parcel_ids.tolist()])
    return parcel_ids.astype(np.int64), labels.astype(str)


def load_roi_feature_spec(
    vertex_csv: Path,
    *,
    n_vertices: int,
    reducers: str | tuple[str, ...] | list[str] = ("mean",),
    manifest_path: Path | None = None,
    allow_legacy_atlas: bool = False,
    require_surface_native: bool = True,
) -> RoiFeatureSpec:
    if not vertex_csv.is_file():
        raise FileNotFoundError(f"ROI vertex CSV not found: {vertex_csv}")

    reducer_tuple = parse_reducers(reducers)
    table = load_vertex_table(vertex_csv)
    manifest = load_parcellation_manifest(manifest_path)
    if manifest:
        validation_summary = validate_manifest_matches_table(
            manifest,
            table,
            vertex_csv_path=vertex_csv,
            allow_legacy_atlas=allow_legacy_atlas,
            require_surface_native=require_surface_native,
        )
    else:
        validation_summary = validate_vertex_table(table, n_vertices=n_vertices)

    if int(validation_summary["n_vertices"]) != int(n_vertices):
        raise ValueError(
            f"ROI table has V={validation_summary['n_vertices']} but feature matrix requires V={n_vertices}"
        )

    parcel_ids, parcel_labels = _ordered_parcel_metadata(table)
    feature_names = tuple(
        f"{label}__{reducer}"
        for label in parcel_labels.astype(str).tolist()
        for reducer in reducer_tuple
    )
    manifest_sha = file_sha256(manifest_path) if manifest_path is not None and manifest_path.is_file() else None

    return RoiFeatureSpec(
        vertex_csv=str(vertex_csv),
        vertex_csv_sha256=file_sha256(vertex_csv),
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        manifest_sha256=manifest_sha,
        atlas_id=str(manifest.get("atlas_id")) if manifest.get("atlas_id") else None,
        mesh=str(manifest.get("mesh")) if manifest.get("mesh") else None,
        vertex_order=(
            str(validation_summary.get("tribe_vertex_order"))
            if validation_summary.get("tribe_vertex_order")
            else None
        ),
        n_vertices=int(table.n_vertices),
        parcel_ids=parcel_ids,
        parcel_labels=parcel_labels,
        parcel_id_by_vertex=table.parcel_id.astype(np.int64),
        reducers=reducer_tuple,
        feature_names=feature_names,
        validation_summary=validation_summary,
        source_metadata=manifest.get("source", {}) if isinstance(manifest.get("source"), dict) else {},
    )


def reduce_vertices_to_rois(X: np.ndarray, spec: RoiFeatureSpec) -> np.ndarray:
    """Aggregate vertex features to ROI features, preserving leading dimensions."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"ROI reduction expects X shape (N,V) or (N,W,V), got {arr.shape}")
    if arr.shape[-1] != spec.n_vertices:
        raise ValueError(f"ROI map expects V={spec.n_vertices}, got X shape {arr.shape}")

    leading_shape = arr.shape[:-1]
    flat = arr.reshape(-1, spec.n_vertices)
    out = np.empty((flat.shape[0], spec.n_features), dtype=np.float32)

    col = 0
    for parcel_id in spec.parcel_ids.tolist():
        idx = np.flatnonzero(spec.parcel_id_by_vertex == int(parcel_id))
        if idx.size == 0:
            out[:, col : col + len(spec.reducers)] = 0.0
            col += len(spec.reducers)
            continue
        values = flat[:, idx]
        for reducer in spec.reducers:
            if reducer == "mean":
                out[:, col] = values.mean(axis=1)
            elif reducer == "std":
                out[:, col] = values.std(axis=1)
            elif reducer == "mean_abs":
                out[:, col] = np.abs(values).mean(axis=1)
            elif reducer == "max_abs":
                out[:, col] = np.abs(values).max(axis=1)
            else:  # parse_reducers should prevent this.
                raise ValueError(f"Unknown ROI reducer: {reducer}")
            col += 1

    return out.reshape(*leading_shape, spec.n_features).astype(np.float32)


def summarise_roi_feature_spec(spec: RoiFeatureSpec, *, include_vertex_map: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source": spec.vertex_csv,
        "vertex_csv_sha256": spec.vertex_csv_sha256,
        "manifest_path": spec.manifest_path,
        "manifest_sha256": spec.manifest_sha256,
        "atlas_id": spec.atlas_id,
        "mesh": spec.mesh,
        "vertex_order": spec.vertex_order,
        "n_vertices": int(spec.n_vertices),
        "n_rois": int(spec.n_rois),
        "reducers": list(spec.reducers),
        "n_features": int(spec.n_features),
        "parcel_ids": spec.parcel_ids.astype(np.int64).tolist(),
        "parcel_labels": spec.parcel_labels.astype(str).tolist(),
        "feature_names": list(spec.feature_names),
        "validation_summary": spec.validation_summary,
        "source_metadata": spec.source_metadata,
    }
    if include_vertex_map:
        summary["parcel_id_by_vertex"] = spec.parcel_id_by_vertex.astype(np.int64).tolist()
    return summary
