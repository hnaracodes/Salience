from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scout_core.parcellation import (
    ALLOWED_VERTEX_ORDER_PROOF_STATUSES,
    VertexParcellationTable,
    validate_manifest_matches_table,
    validate_vertex_table,
)
from scout_core.parcellation import EXPECTED_FSAVERAGE5_HEMI_VERTICES
from scout_core.schaefer_surface_labels import (
    remap_to_contiguous_parcel_ids,
    reorder_hemisphere_labels_by_coords,
)


def _synthetic_annot_labels(n_vertices: int = EXPECTED_FSAVERAGE5_HEMI_VERTICES) -> tuple[np.ndarray, dict[int, str]]:
    raw = np.zeros(n_vertices, dtype=np.int64)
    label_map: dict[int, str] = {}
    for idx in range(1, 201):
        raw[idx - 1] = idx
        label_map[idx] = f"7Networks_LH_Vis_{idx}"
    return raw, label_map


def test_remap_to_contiguous_parcel_ids_lh_rh_400():
    lh_raw, lh_map = _synthetic_annot_labels()
    rh_raw, rh_map = _synthetic_annot_labels()
    for idx in range(1, 201):
        rh_raw[idx - 1] = idx
        rh_map[idx] = f"7Networks_RH_Vis_{idx}"

    lh_labels, rh_labels, parcel_id_to_name, ordered = remap_to_contiguous_parcel_ids(
        lh_raw,
        rh_raw,
        lh_map,
        rh_map,
        n_rois=400,
    )
    assert lh_labels.shape == (EXPECTED_FSAVERAGE5_HEMI_VERTICES,)
    assert rh_labels.shape == (EXPECTED_FSAVERAGE5_HEMI_VERTICES,)
    assert len(ordered) == 400
    assert len(parcel_id_to_name) == 400
    assert set(np.unique(lh_labels[lh_labels > 0])) <= set(range(1, 401))
    assert set(np.unique(rh_labels[rh_labels > 0])) <= set(range(1, 401))


def test_remap_rejects_wrong_parcel_count():
    lh_raw, lh_map = _synthetic_annot_labels()
    rh_raw, rh_map = _synthetic_annot_labels()
    with pytest.raises(ValueError, match="Expected 100 unique Schaefer parcels"):
        remap_to_contiguous_parcel_ids(lh_raw, rh_raw, lh_map, rh_map, n_rois=100)


def test_validate_manifest_rejects_volume_projection_by_default():
    table = VertexParcellationTable(
        vertex_index=np.arange(4, dtype=np.int64),
        parcel_id=np.asarray([1, 1, 2, 2], dtype=np.int64),
        parcel_label=np.asarray(["a", "a", "b", "b"], dtype=object),
        yeo_network_id=np.zeros(4, dtype=np.int64),
        yeo_network_name=np.asarray([""] * 4, dtype=object),
        hemisphere=np.asarray(["lh", "lh", "rh", "rh"], dtype=object),
    )
    manifest = {
        "mesh": "fsaverage5",
        "tribe_vertex_order": "lh_then_rh_fsaverage5",
        "n_vertices_expected": 4,
        "n_parcels": 2,
        "source": {"label_source_kind": "volume_projection_fallback"},
    }
    with pytest.raises(ValueError, match="volume_projection_fallback"):
        validate_manifest_matches_table(
            manifest,
            table,
            require_surface_native=True,
            require_hemispheres=False,
        )


def test_validate_manifest_accepts_surface_annot_with_proof():
    table = VertexParcellationTable(
        vertex_index=np.arange(4, dtype=np.int64),
        parcel_id=np.asarray([1, 1, 2, 2], dtype=np.int64),
        parcel_label=np.asarray(["a", "a", "b", "b"], dtype=object),
        yeo_network_id=np.zeros(4, dtype=np.int64),
        yeo_network_name=np.asarray([""] * 4, dtype=object),
        hemisphere=np.asarray(["lh", "lh", "rh", "rh"], dtype=object),
    )
    manifest = {
        "mesh": "fsaverage5",
        "tribe_vertex_order": "lh_then_rh_fsaverage5",
        "n_vertices_expected": 4,
        "n_parcels": 2,
        "source": {
            "label_source_kind": "surface_annot_cbig",
            "filled_unassigned_vertices": 10,
        },
        "vertex_order_proof": {
            "status": "annot_matches_nilearn_projection",
            "reorder_applied": False,
        },
    }
    summary = validate_manifest_matches_table(
        manifest,
        table,
        require_surface_native=True,
        require_hemispheres=False,
    )
    assert summary["label_source_kind"] == "surface_annot_cbig"
    assert summary["vertex_order_proof"]["status"] in ALLOWED_VERTEX_ORDER_PROOF_STATUSES


def test_reorder_hemisphere_labels_by_coords_is_bijective():
    rng = np.random.default_rng(0)
    n = 64
    target_coords = rng.standard_normal((n, 3)).astype(np.float32)
    perm = rng.permutation(n)
    source_coords = target_coords[perm]
    labels = np.arange(1, n + 1, dtype=np.int64)
    reordered, meta = reorder_hemisphere_labels_by_coords(
        labels, source_coords=source_coords, target_coords=target_coords, atol=1e-1
    )
    assert meta["reorder_applied"] is True
    assert reordered.shape == labels.shape
    assert len(np.unique(reordered[reordered > 0])) == n


def test_validate_manifest_legacy_volume_allowed_with_flag():
    table = VertexParcellationTable(
        vertex_index=np.arange(4, dtype=np.int64),
        parcel_id=np.asarray([1, 1, 2, 2], dtype=np.int64),
        parcel_label=np.asarray(["a", "a", "b", "b"], dtype=object),
        yeo_network_id=np.zeros(4, dtype=np.int64),
        yeo_network_name=np.asarray([""] * 4, dtype=object),
        hemisphere=np.asarray(["lh", "lh", "rh", "rh"], dtype=object),
    )
    manifest = {
        "mesh": "fsaverage5",
        "tribe_vertex_order": "lh_then_rh_fsaverage5",
        "n_vertices_expected": 4,
        "n_parcels": 2,
        "source": {"label_source_kind": "volume_projection_fallback"},
    }
    summary = validate_manifest_matches_table(
        manifest,
        table,
        allow_legacy_atlas=True,
        require_surface_native=False,
        require_hemispheres=False,
    )
    assert "legacy_atlas_warning" in summary
