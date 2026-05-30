from __future__ import annotations

import numpy as np
import pytest

from scout_core.parcellation import CANONICAL_VERTEX_ORDER
from scout_core.vertex_equivalence import (
    canonical_vertex_equivalence_report_path,
    ReferenceMeshes,
    build_vertex_equivalence_report,
    mesh_fingerprint,
    validate_vertex_equivalence_summary,
    vertex_equivalence_reference,
)


def _dummy_mesh(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(
        [
            [0.0 + offset, 0.0, 0.0],
            [1.0 + offset, 0.0, 0.0],
            [0.0 + offset, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    return coords, faces


def test_mesh_fingerprint_changes_when_faces_change():
    coords, faces = _dummy_mesh()
    first = mesh_fingerprint((coords, faces))
    second = mesh_fingerprint((coords, np.asarray([[0, 2, 1]], dtype=np.int32)))
    assert first["coords_sha256"] == second["coords_sha256"]
    assert first["faces_sha256"] != second["faces_sha256"]


def test_validate_vertex_equivalence_summary_rejects_contract_only_by_default():
    with pytest.raises(ValueError, match="contract_only"):
        validate_vertex_equivalence_summary(
            {
                "proof_status": "contract_only",
                "mesh": "fsaverage5",
                "vertex_order": CANONICAL_VERTEX_ORDER,
            },
            allow_contract_only=False,
        )


def test_build_vertex_equivalence_report_marks_mesh_identity_verified(monkeypatch):
    mesh_map = {
        "pial_left": _dummy_mesh(0.0),
        "pial_right": _dummy_mesh(1.0),
        "white_left": _dummy_mesh(2.0),
        "white_right": _dummy_mesh(3.0),
    }

    monkeypatch.setattr(
        "scout_core.vertex_equivalence.load_nilearn_reference_meshes",
        lambda mesh="fsaverage5": mesh_map,
    )
    monkeypatch.setattr(
        "scout_core.vertex_equivalence.load_tribe_reference_meshes",
        lambda mesh="fsaverage5": ReferenceMeshes(
            meshes=mesh_map,
            reference_kind="upstream_source_mirror",
            reference_detail="synthetic",
            source_url="https://example.test",
        ),
    )

    report = build_vertex_equivalence_report(mesh="fsaverage5")
    assert report["proof_status"] == "mesh_identity_verified"
    assert report["reference_kind"] == "upstream_source_mirror"
    assert report["projection_comparison"]["available"] is False


def test_vertex_equivalence_reference_contains_report_hash(tmp_path):
    report = {
        "proof_status": "mesh_identity_verified",
        "mesh": "fsaverage5",
        "vertex_order": CANONICAL_VERTEX_ORDER,
        "report_sha256": "deadbeef",
        "schema_version": 1,
    }
    report_path = canonical_vertex_equivalence_report_path(tmp_path / "vertex_report.json")
    ref = vertex_equivalence_reference(report, report_path=report_path)
    assert ref["report_sha256"] == "deadbeef"
    assert ref["report_path"] == str(report_path)
