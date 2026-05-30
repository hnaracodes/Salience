from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scout_core.parcellation import (
    CANONICAL_VERTEX_ORDER,
    EXPECTED_FSAVERAGE5_HEMI_VERTICES,
    EXPECTED_FSAVERAGE5_VERTICES,
)

TRIBEV2_UTILS_FMRI_URL = "https://raw.githubusercontent.com/facebookresearch/tribev2/main/tribev2/utils_fmri.py"
TRIBEV2_PLOTTING_URL = "https://raw.githubusercontent.com/facebookresearch/tribev2/main/tribev2/plotting/cortical_pv.py"
VERIFIED_PROOF_STATUSES = ("mesh_identity_verified", "projection_equivalence_verified")
ALL_PROOF_STATUSES = (*VERIFIED_PROOF_STATUSES, "contract_only")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERTEX_EQUIVALENCE_REPORT = (
    PROJECT_ROOT / "scout_data" / "neuroEmoCode" / "vertex_equivalence_report.json"
)


@dataclass(frozen=True)
class ReferenceMeshes:
    meshes: dict[str, Any]
    reference_kind: str
    reference_detail: str
    source_url: str | None


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_vertex_equivalence_report_path(path: Path | None = None) -> Path:
    return (path or DEFAULT_VERTEX_EQUIVALENCE_REPORT).expanduser().resolve()


def load_vertex_equivalence_report(path: Path | None = None) -> dict[str, Any]:
    report_path = canonical_vertex_equivalence_report_path(path)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Vertex equivalence report not found: {report_path}. "
            "Generate it first with scripts/verify_tribe_vertex_equivalence.py."
        )
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Vertex equivalence report is not a JSON object: {report_path}")
    return raw


def vertex_equivalence_reference(report: dict[str, Any], *, report_path: Path) -> dict[str, Any]:
    return {
        "proof_status": str(report.get("proof_status") or "contract_only"),
        "mesh": str(report.get("mesh") or "fsaverage5"),
        "vertex_order": str(report.get("vertex_order") or CANONICAL_VERTEX_ORDER),
        "reference_kind": report.get("reference_kind"),
        "report_path": str(report_path),
        "report_sha256": str(report.get("report_sha256") or ""),
        "schema_version": int(report.get("schema_version") or 1),
        "mesh_fingerprints": report.get("mesh_fingerprints"),
        "projection_comparison": report.get("projection_comparison"),
    }


def _as_mesh_arrays(mesh_source: Any) -> tuple[np.ndarray, np.ndarray]:
    from nilearn.surface import InMemoryMesh

    if isinstance(mesh_source, InMemoryMesh):
        coords = np.asarray(mesh_source.coordinates, dtype=np.float32)
        faces = np.asarray(mesh_source.faces, dtype=np.int32)
        return coords, faces
    if isinstance(mesh_source, tuple) and len(mesh_source) == 2:
        coords = np.asarray(mesh_source[0], dtype=np.float32)
        faces = np.asarray(mesh_source[1], dtype=np.int32)
        return coords, faces
    if isinstance(mesh_source, (str, Path)):
        import nibabel as nib

        img = nib.load(str(mesh_source))
        if hasattr(img, "agg_data"):
            coords = np.asarray(img.darrays[0].data, dtype=np.float32)
            faces = np.asarray(img.darrays[1].data, dtype=np.int32)
            return coords, faces
    raise TypeError(f"Unsupported mesh source type: {type(mesh_source)!r}")


def mesh_fingerprint(mesh_source: Any) -> dict[str, Any]:
    coords, faces = _as_mesh_arrays(mesh_source)
    return {
        "n_vertices": int(coords.shape[0]),
        "n_faces": int(faces.shape[0]),
        "coords_shape": [int(x) for x in coords.shape],
        "faces_shape": [int(x) for x in faces.shape],
        "coords_sha256": hashlib.sha256(np.ascontiguousarray(coords).tobytes()).hexdigest(),
        "faces_sha256": hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest(),
    }


def compare_mesh_fingerprints(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_vertices_match": int(left.get("n_vertices", -1)) == int(right.get("n_vertices", -2)),
        "n_faces_match": int(left.get("n_faces", -1)) == int(right.get("n_faces", -2)),
        "coords_sha256_match": str(left.get("coords_sha256", "")) == str(right.get("coords_sha256", "")),
        "faces_sha256_match": str(left.get("faces_sha256", "")) == str(right.get("faces_sha256", "")),
    }


def surface_to_time_by_vertex(surface: np.ndarray, *, expected_vertices: int) -> np.ndarray:
    arr = np.asarray(surface, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] == expected_vertices:
        arr = arr.T
    elif arr.shape[1] == expected_vertices:
        pass
    else:
        raise ValueError(
            f"Unexpected surface shape {arr.shape}; expected one axis to be {expected_vertices}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_nilearn_reference_meshes(mesh: str = "fsaverage5") -> dict[str, Any]:
    from nilearn import datasets

    return datasets.fetch_surf_fsaverage(mesh)


def load_tribe_reference_meshes(mesh: str = "fsaverage5") -> ReferenceMeshes:
    try:
        from tribev2.utils_fmri import TribeSurfaceProjector

        projector = TribeSurfaceProjector(mesh=mesh)
        return ReferenceMeshes(
            meshes=projector.get_mesh(),
            reference_kind="local_tribev2_package",
            reference_detail="Loaded via tribev2.utils_fmri.TribeSurfaceProjector.get_mesh().",
            source_url=None,
        )
    except Exception:
        return ReferenceMeshes(
            meshes=load_nilearn_reference_meshes(mesh),
            reference_kind="upstream_source_mirror",
            reference_detail=(
                "Local tribev2 package unavailable; mirrored the published TribeSurfaceProjector "
                "mesh-loading contract, which fetches nilearn fsaverage meshes."
            ),
            source_url=TRIBEV2_UTILS_FMRI_URL,
        )


def project_bold_to_fsaverage5_tribe_equivalent(
    bold_path: Path,
    *,
    mesh: str = "fsaverage5",
    radius: float = 3.0,
    interpolation: str = "linear",
) -> np.ndarray:
    import nibabel as nib
    from nilearn.surface import vol_to_surf

    img = nib.load(str(bold_path))
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D BOLD image, got shape {img.shape}: {bold_path}")

    meshes = load_nilearn_reference_meshes(mesh)
    hemis = []
    for hemi in ("left", "right"):
        surf = vol_to_surf(
            img,
            surf_mesh=meshes[f"pial_{hemi}"],
            inner_mesh=meshes[f"white_{hemi}"],
            radius=radius,
            interpolation=interpolation,
            kind="auto",
            n_samples=None,
            mask_img=None,
            depth=None,
        )
        hemis.append(
            surface_to_time_by_vertex(surf, expected_vertices=EXPECTED_FSAVERAGE5_HEMI_VERTICES)
        )
    if hemis[0].shape[0] != hemis[1].shape[0]:
        raise ValueError(f"Left/right hemisphere TR mismatch: {hemis[0].shape} vs {hemis[1].shape}")
    surface = np.concatenate(hemis, axis=1)
    if surface.shape[1] != EXPECTED_FSAVERAGE5_VERTICES:
        raise ValueError(
            f"Projected surface has V={surface.shape[1]}, expected {EXPECTED_FSAVERAGE5_VERTICES}"
        )
    return surface.astype(np.float32)


def _project_with_local_tribev2_projector(
    bold_path: Path,
    *,
    mesh: str = "fsaverage5",
    radius: float = 3.0,
    interpolation: str = "linear",
) -> np.ndarray:
    import nibabel as nib
    from tribev2.utils_fmri import TribeSurfaceProjector

    img = nib.load(str(bold_path))
    projector = TribeSurfaceProjector(mesh=mesh, radius=radius, interpolation=interpolation)
    surf = np.asarray(projector.apply(img), dtype=np.float32)
    return surface_to_time_by_vertex(surf, expected_vertices=EXPECTED_FSAVERAGE5_VERTICES)


def validate_vertex_equivalence_summary(
    summary: dict[str, Any] | None,
    *,
    allow_contract_only: bool,
) -> dict[str, Any]:
    if not summary:
        if not allow_contract_only:
            raise ValueError("Missing vertex_equivalence proof metadata.")
        summary = {
            "proof_status": "contract_only",
            "mesh": "fsaverage5",
            "vertex_order": CANONICAL_VERTEX_ORDER,
            "reference_kind": "legacy_opt_in",
            "reference_detail": "Legacy artifact loaded with allow_contract_only override.",
        }

    proof_status = str(summary.get("proof_status") or "").strip()
    if proof_status not in ALL_PROOF_STATUSES:
        raise ValueError(
            f"Unsupported vertex_equivalence proof_status {proof_status!r}; "
            f"expected one of {ALL_PROOF_STATUSES}"
        )
    if proof_status == "contract_only" and not allow_contract_only:
        raise ValueError(
            "vertex_equivalence proof_status='contract_only'. "
            "Pass the explicit legacy override only if you intentionally want to train on unverified data."
        )
    if str(summary.get("mesh") or "fsaverage5") != "fsaverage5":
        raise ValueError("vertex_equivalence metadata must record mesh='fsaverage5'.")
    vertex_order = str(summary.get("vertex_order") or CANONICAL_VERTEX_ORDER)
    if vertex_order != CANONICAL_VERTEX_ORDER:
        raise ValueError(
            f"vertex_equivalence metadata must record vertex_order={CANONICAL_VERTEX_ORDER!r}, "
            f"got {vertex_order!r}"
        )
    return {
        "proof_status": proof_status,
        "reference_kind": summary.get("reference_kind"),
        "mesh": "fsaverage5",
        "vertex_order": CANONICAL_VERTEX_ORDER,
        "reference_detail": summary.get("reference_detail"),
        "source_url": summary.get("source_url"),
        "mesh_comparison": summary.get("mesh_comparison"),
        "projection_comparison": summary.get("projection_comparison"),
        "mesh_fingerprints": summary.get("mesh_fingerprints"),
        "report_sha256": summary.get("report_sha256"),
    }


def build_vertex_equivalence_report(
    *,
    bold_path: Path | None = None,
    mesh: str = "fsaverage5",
    radius: float = 3.0,
    interpolation: str = "linear",
    allclose_atol: float = 1e-5,
    runtime_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nilearn_meshes = load_nilearn_reference_meshes(mesh)
    reference = load_tribe_reference_meshes(mesh)

    mesh_fingerprints = {
        "nilearn": {
            hemi_name: mesh_fingerprint(nilearn_meshes[hemi_name])
            for hemi_name in ("pial_left", "pial_right", "white_left", "white_right")
        },
        "tribe_reference": {
            hemi_name: mesh_fingerprint(reference.meshes[hemi_name])
            for hemi_name in ("pial_left", "pial_right", "white_left", "white_right")
        },
    }
    mesh_comparison = {
        hemi_name: compare_mesh_fingerprints(
            mesh_fingerprints["nilearn"][hemi_name],
            mesh_fingerprints["tribe_reference"][hemi_name],
        )
        for hemi_name in ("pial_left", "pial_right", "white_left", "white_right")
    }
    mesh_identity_verified = all(
        all(bool(v) for v in hemi_result.values())
        for hemi_result in mesh_comparison.values()
    )

    projection_comparison: dict[str, Any] = {
        "available": False,
        "verified": False,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "allclose_atol": float(allclose_atol),
    }
    if (
        bold_path is not None
        and reference.reference_kind == "local_tribev2_package"
        and interpolation in {"linear", "nearest"}
    ):
        prep_surface = project_bold_to_fsaverage5_tribe_equivalent(
            bold_path,
            mesh=mesh,
            radius=radius,
            interpolation=interpolation,
        )
        tribe_surface = _project_with_local_tribev2_projector(
            bold_path,
            mesh=mesh,
            radius=radius,
            interpolation=interpolation,
        )
        diff = np.asarray(prep_surface - tribe_surface, dtype=np.float64)
        max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
        mean_abs_diff = float(np.mean(np.abs(diff))) if diff.size else 0.0
        projection_comparison = {
            "available": True,
            "verified": bool(np.allclose(prep_surface, tribe_surface, atol=allclose_atol)),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "allclose_atol": float(allclose_atol),
        }

    proof_status = "contract_only"
    if projection_comparison["available"] and projection_comparison["verified"]:
        proof_status = "projection_equivalence_verified"
    elif mesh_identity_verified:
        proof_status = "mesh_identity_verified"

    report = {
        "schema_version": 1,
        "mesh": mesh,
        "vertex_order": CANONICAL_VERTEX_ORDER,
        "proof_status": proof_status,
        "reference_kind": reference.reference_kind,
        "reference_detail": reference.reference_detail,
        "source_url": reference.source_url,
        "mesh_comparison": mesh_comparison,
        "projection_comparison": projection_comparison,
        "mesh_fingerprints": mesh_fingerprints,
        "bold_path": str(bold_path) if bold_path is not None else None,
        "projection_params": {
            "radius": float(radius),
            "interpolation": interpolation,
        },
        "generated_at_unix_ms": int(time.time() * 1000),
        "runtime_provenance": runtime_provenance or {},
    }
    report["report_sha256"] = json_sha256(report)
    return report


def write_vertex_equivalence_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
