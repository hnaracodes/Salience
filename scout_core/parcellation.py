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
CANONICAL_VERTEX_ORDER = "lh_then_rh_fsaverage5"
CANONICAL_HEMISPHERE_ORDER = "lh_then_rh"
VERTEX_ORDER_ALIASES = {
    CANONICAL_VERTEX_ORDER: CANONICAL_VERTEX_ORDER,
    "facebook/tribev2_lh_then_rh": CANONICAL_VERTEX_ORDER,
    "lh_then_rh_nilearn_fsaverage5": CANONICAL_VERTEX_ORDER,
}
HEMISPHERE_ORDER_ALIASES = {
    CANONICAL_HEMISPHERE_ORDER: CANONICAL_HEMISPHERE_ORDER,
    "left_then_right": CANONICAL_HEMISPHERE_ORDER,
}

SURFACE_NATIVE_LABEL_KINDS = frozenset({"surface_annot_cbig", "surface_label_gifti"})
ALLOWED_VERTEX_ORDER_PROOF_STATUSES = frozenset(
    {
        "annot_matches_nilearn_projection",
        "annot_reordered_by_coordinate_map",
        "surface_label_gifti_no_projection_check",
        "skipped_for_tests",
    }
)
MAX_FILLED_UNASSIGNED_SURFACE = 500
MAX_FILLED_UNASSIGNED_SURFACE_ANNOT = 2000


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


def normalize_vertex_order(value: Any) -> str | None:
    raw = str(value).strip() if value not in (None, "") else ""
    if not raw:
        return None
    return VERTEX_ORDER_ALIASES.get(raw, raw)


def normalize_hemisphere_order(value: Any) -> str | None:
    raw = str(value).strip() if value not in (None, "") else ""
    if not raw:
        return None
    return HEMISPHERE_ORDER_ALIASES.get(raw, raw)


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
                parcel_ids.append(0)
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

    legacy_mask = pid <= 0
    if np.any(legacy_mask):
        existing_max = int(pid[~legacy_mask].max()) if np.any(~legacy_mask) else 0
        legacy_labels = plab[legacy_mask].astype(str).tolist()
        stable_ids = {
            label: existing_max + idx + 1
            for idx, label in enumerate(sorted(set(legacy_labels)))
        }
        pid[legacy_mask] = np.asarray([stable_ids[label] for label in legacy_labels], dtype=np.int64)

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

    label_by_parcel: dict[int, str] = {}
    network_by_parcel: dict[int, tuple[int, str]] = {}
    for parcel_id, label, network_id, network_name in zip(
        table.parcel_id.tolist(),
        labels.tolist(),
        table.yeo_network_id.tolist(),
        np.asarray(table.yeo_network_name).astype(str).tolist(),
        strict=True,
    ):
        parcel_key = int(parcel_id)
        label_value = str(label)
        network_value = (int(network_id), str(network_name))
        prev_label = label_by_parcel.setdefault(parcel_key, label_value)
        if prev_label != label_value:
            raise ValueError(
                f"parcel_id {parcel_key} maps to conflicting labels: {prev_label!r} vs {label_value!r}"
            )
        prev_network = network_by_parcel.setdefault(parcel_key, network_value)
        if prev_network != network_value:
            raise ValueError(
                f"parcel_id {parcel_key} maps to conflicting networks: {prev_network!r} vs {network_value!r}"
            )

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
        if not np.all(hemi[:EXPECTED_FSAVERAGE5_HEMI_VERTICES] == "lh"):
            raise ValueError("Expected first 10242 vertices to be labeled as lh")
        if not np.all(hemi[EXPECTED_FSAVERAGE5_HEMI_VERTICES:] == "rh"):
            raise ValueError("Expected last 10242 vertices to be labeled as rh")

    network_names = sorted(
        name for name in set(np.asarray(table.yeo_network_name).astype(str).tolist()) if name
    )
    parcel_sizes = np.asarray([int(np.sum(table.parcel_id == pid)) for pid in parcel_ids], dtype=np.int64)
    return {
        "n_vertices": int(table.n_vertices),
        "n_parcels": int(len(parcel_ids)),
        "hemisphere_counts": hemi_counts,
        "hemisphere_order": CANONICAL_HEMISPHERE_ORDER if require_hemispheres else None,
        "tribe_vertex_order": CANONICAL_VERTEX_ORDER if require_hemispheres else None,
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
    allow_legacy_atlas: bool = False,
    require_surface_native: bool = True,
    require_hemispheres: bool = True,
) -> dict[str, Any]:
    if not manifest:
        return {}

    expected_vertices = int(manifest.get("n_vertices_expected") or EXPECTED_FSAVERAGE5_VERTICES)
    expected_parcels = manifest.get("n_parcels")
    summary = validate_vertex_table(
        table,
        n_vertices=expected_vertices,
        expected_n_parcels=int(expected_parcels) if expected_parcels is not None else None,
        require_hemispheres=require_hemispheres,
    )

    manifest_mesh = str(manifest.get("mesh") or "").strip()
    if manifest_mesh and manifest_mesh != "fsaverage5":
        raise ValueError(f"Unsupported parcellation mesh {manifest_mesh!r}; expected 'fsaverage5'")

    manifest_hemi_order = normalize_hemisphere_order(manifest.get("hemisphere_order"))
    if manifest_hemi_order not in (None, CANONICAL_HEMISPHERE_ORDER):
        raise ValueError(
            f"Unsupported hemisphere order {manifest.get('hemisphere_order')!r}; "
            f"expected {CANONICAL_HEMISPHERE_ORDER!r}"
        )

    manifest_vertex_order = normalize_vertex_order(manifest.get("tribe_vertex_order"))
    if manifest_vertex_order not in (None, CANONICAL_VERTEX_ORDER):
        raise ValueError(
            f"Unsupported vertex order {manifest.get('tribe_vertex_order')!r}; "
            f"expected one of {sorted(VERTEX_ORDER_ALIASES)}"
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

    vertex_equivalence = manifest.get("vertex_equivalence")
    if vertex_equivalence is not None:
        from scout_core.vertex_equivalence import validate_vertex_equivalence_summary

        validated_vertex_equivalence = validate_vertex_equivalence_summary(
            vertex_equivalence if isinstance(vertex_equivalence, dict) else {},
            allow_contract_only=True,
        )
        expected_report_hash = (
            manifest.get("artifact_sha256", {}).get("vertex_equivalence_report_json")
            if isinstance(manifest.get("artifact_sha256"), dict)
            else None
        )
        if expected_report_hash and validated_vertex_equivalence.get("report_sha256") != expected_report_hash:
            raise ValueError(
                "vertex_equivalence report sha256 mismatch: "
                f"manifest={expected_report_hash} "
                f"reported={validated_vertex_equivalence.get('report_sha256')}"
            )
        summary["vertex_equivalence"] = validated_vertex_equivalence
    mesh_fingerprints = manifest.get("mesh_fingerprints")
    if isinstance(mesh_fingerprints, dict):
        summary["mesh_fingerprints"] = mesh_fingerprints

    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    label_source_kind = str(source.get("label_source_kind") or "").strip()
    summary["label_source_kind"] = label_source_kind or None

    if label_source_kind == "volume_projection_fallback":
        if allow_legacy_atlas:
            summary["legacy_atlas_warning"] = (
                "Training with deprecated volume_projection_fallback atlas; retrain after surface-native regen."
            )
        else:
            raise ValueError(
                "Parcellation manifest uses deprecated volume_projection_fallback labels. "
                "Regenerate atlas with scripts/build_schaefer_vertex_regions.py and CBIG .annot files, "
                "or pass allow_legacy_atlas=True for diagnostics only."
            )

    if require_surface_native and label_source_kind in SURFACE_NATIVE_LABEL_KINDS:
        filled = source.get("filled_unassigned_vertices")
        fill_limit = (
            MAX_FILLED_UNASSIGNED_SURFACE_ANNOT
            if label_source_kind == "surface_annot_cbig"
            else MAX_FILLED_UNASSIGNED_SURFACE
        )
        if filled is not None and int(filled) > fill_limit:
            raise ValueError(
                f"Surface-native atlas filled {int(filled)} unassigned vertices "
                f"(limit {fill_limit}). Check annot quality."
            )

    vertex_order_proof = manifest.get("vertex_order_proof")
    if isinstance(vertex_order_proof, dict):
        summary["vertex_order_proof"] = vertex_order_proof
        if require_surface_native and label_source_kind == "surface_annot_cbig":
            status = str(vertex_order_proof.get("status") or "").strip()
            if status not in ALLOWED_VERTEX_ORDER_PROOF_STATUSES:
                raise ValueError(
                    f"Invalid vertex_order_proof.status {status!r}; expected one of "
                    f"{sorted(ALLOWED_VERTEX_ORDER_PROOF_STATUSES)}"
                )
    elif require_surface_native and label_source_kind == "surface_annot_cbig":
        raise ValueError(
            "Surface annot atlas manifest is missing vertex_order_proof metadata. "
            "Regenerate configs/parcellation_manifest.yaml with the updated builder."
        )

    summary["atlas_id"] = manifest.get("atlas_id")
    summary["mesh"] = manifest_mesh or "fsaverage5"
    summary["tribe_vertex_order"] = manifest_vertex_order or CANONICAL_VERTEX_ORDER
    summary["source_vertex_order"] = manifest.get("tribe_vertex_order")
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
