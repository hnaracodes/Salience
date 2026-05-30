"""Load CBIG Schaefer surface .annot labels and verify Nilearn vertex order."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from scout_core.parcellation import EXPECTED_FSAVERAGE5_HEMI_VERTICES, EXPECTED_FSAVERAGE5_VERTICES

PROJECTION_AGREEMENT_THRESHOLD = 0.85
COORD_MATCH_ATOL = 1e-2
MAX_FILLED_UNASSIGNED_SURFACE = 500
# CBIG .annot leaves medial wall unassigned (~1.7k vertices on fsaverage5); nearest fill is expected.
MAX_FILLED_UNASSIGNED_SURFACE_ANNOT = 2000

ALLOWED_VERTEX_ORDER_PROOF_STATUSES = (
    "annot_matches_nilearn_projection",
    "annot_reordered_by_coordinate_map",
)

DEFAULT_ALIGNMENT_REPORT = (
    Path(__file__).resolve().parents[2] / "scout_data" / "neuroEmoCode" / "schaefer_annot_alignment_report.json"
)

CBIG_FSAVERAGE5_SURF_BASE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage5/surf"
)


def _decode_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalise_label(label: str) -> str:
    label = label.strip()
    if label.startswith("b'") and label.endswith("'"):
        label = label[2:-1]
    return label


def network_name_from_label(label: str) -> str:
    parts = label.split("_")
    if len(parts) >= 4 and parts[0].endswith("Networks") and parts[1] in {"LH", "RH"}:
        return "_".join(parts[2:-1]).strip()
    return parts[-2].strip() if len(parts) >= 2 else ""


def load_freesurfer_annot(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    """Return per-vertex annotation indices and raw_key -> region name."""
    from nibabel.freesurfer import read_annot

    labels, _ctab, names = read_annot(str(path))
    arr = np.asarray(labels, dtype=np.int64)
    if arr.ndim != 1 or arr.shape[0] != EXPECTED_FSAVERAGE5_HEMI_VERTICES:
        raise ValueError(
            f"Expected {EXPECTED_FSAVERAGE5_HEMI_VERTICES} vertices in {path}, got shape {arr.shape}"
        )

    label_by_raw: dict[int, str] = {}
    for idx, name in enumerate(names):
        text = _normalise_label(_decode_label(name))
        if idx > 0 and text:
            label_by_raw[int(idx)] = text
    return arr, label_by_raw


def _canonical_sort_key(name: str) -> tuple[str, str, str]:
    parts = name.split("_")
    hemi = parts[1] if len(parts) > 1 else ""
    network = network_name_from_label(name)
    tail = parts[-1] if parts else name
    return (hemi, network, tail)


def remap_to_contiguous_parcel_ids(
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
    lh_label_by_raw: dict[int, str],
    rh_label_by_raw: dict[int, str],
    *,
    n_rois: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], list[str]]:
    """Map raw annot keys to global parcel ids 1..n_rois using stable Schaefer name order."""
    name_to_raw: dict[str, int] = {}
    for raw_labels, raw_map in ((lh_labels, lh_label_by_raw), (rh_labels, rh_label_by_raw)):
        for raw_key in np.unique(raw_labels):
            raw_int = int(raw_key)
            if raw_int <= 0:
                continue
            name = raw_map.get(raw_int)
            if not name:
                raise ValueError(f"Missing annot name for raw key {raw_int}")
            if name in name_to_raw and name_to_raw[name] != raw_int:
                raise ValueError(f"Conflicting raw keys for parcel name {name!r}")
            name_to_raw[name] = raw_int

    ordered_names = sorted(name_to_raw.keys(), key=_canonical_sort_key)
    if len(ordered_names) != n_rois:
        raise ValueError(
            f"Expected {n_rois} unique Schaefer parcels, found {len(ordered_names)} from annot files"
        )

    name_to_parcel_id = {name: idx + 1 for idx, name in enumerate(ordered_names)}
    parcel_id_to_name = {parcel_id: name for name, parcel_id in name_to_parcel_id.items()}

    def _remap_hemi(raw_labels: np.ndarray, raw_map: dict[int, str]) -> np.ndarray:
        out = np.zeros_like(raw_labels, dtype=np.int64)
        for vertex, raw_key in enumerate(raw_labels.tolist()):
            raw_int = int(raw_key)
            if raw_int <= 0:
                continue
            name = raw_map[raw_int]
            out[vertex] = name_to_parcel_id[name]
        return out

    return (
        _remap_hemi(lh_labels, lh_label_by_raw),
        _remap_hemi(rh_labels, rh_label_by_raw),
        parcel_id_to_name,
        ordered_names,
    )


def _cbig_pial_paths(data_dir: Path) -> tuple[Path, Path]:
    surf_dir = data_dir / "FreeSurfer5.3" / "fsaverage5" / "surf"
    return surf_dir / "lh.pial", surf_dir / "rh.pial"


def _download_cbig_pial_surfaces(data_dir: Path) -> None:
    import urllib.request

    lh_path, rh_path = _cbig_pial_paths(data_dir)
    lh_path.parent.mkdir(parents=True, exist_ok=True)
    for hemi, path in (("lh", lh_path), ("rh", rh_path)):
        if path.is_file():
            continue
        url = f"{CBIG_FSAVERAGE5_SURF_BASE}/{hemi}.pial"
        urllib.request.urlretrieve(url, path)


def load_cbig_fsaverage5_pial_coords(
    data_dir: Path,
    *,
    download_if_missing: bool = True,
) -> dict[str, np.ndarray]:
    """Pial coordinates in FreeSurfer vertex order (matches CBIG .annot indexing)."""
    from nibabel.freesurfer import read_geometry

    lh_path, rh_path = _cbig_pial_paths(data_dir)
    if download_if_missing and (not lh_path.is_file() or not rh_path.is_file()):
        _download_cbig_pial_surfaces(data_dir)
    if not lh_path.is_file() or not rh_path.is_file():
        raise FileNotFoundError(
            f"Missing CBIG fsaverage5 pial surfaces under {lh_path.parent}. "
            "Download lh.pial and rh.pial from CBIG FreeSurfer5.3/fsaverage5/surf "
            "(see docs/atlas-setup/cbig-schaefer2018-fsaverage5.md)."
        )

    def _load(path: Path) -> np.ndarray:
        coords, _faces = read_geometry(str(path))
        arr = np.asarray(coords, dtype=np.float32)
        if arr.shape[0] != EXPECTED_FSAVERAGE5_HEMI_VERTICES:
            raise ValueError(f"Expected {EXPECTED_FSAVERAGE5_HEMI_VERTICES} coords in {path}, got {arr.shape}")
        return arr

    return {"lh": _load(lh_path), "rh": _load(rh_path)}


def load_nilearn_fsaverage5_coords() -> dict[str, np.ndarray]:
    from nilearn import datasets

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    def _coords(mesh_path: str) -> np.ndarray:
        import nibabel as nib

        img = nib.load(mesh_path)
        coords = np.asarray(img.darrays[0].data, dtype=np.float32)
        if coords.shape[0] != EXPECTED_FSAVERAGE5_HEMI_VERTICES:
            raise ValueError(f"Expected {EXPECTED_FSAVERAGE5_HEMI_VERTICES} coords, got {coords.shape}")
        return coords

    return {
        "lh": _coords(fsaverage.pial_left),
        "rh": _coords(fsaverage.pial_right),
        "pial_left": fsaverage.pial_left,
        "pial_right": fsaverage.pial_right,
    }


def _mesh_coords_share_vertex_order(
    fs_coords: np.ndarray,
    nilearn_coords: np.ndarray,
    *,
    atol: float = COORD_MATCH_ATOL,
) -> dict[str, Any]:
    """True when FreeSurfer and Nilearn use the same per-index vertex order on this mesh."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(fs_coords, dtype=np.float64))
    distances, indices = tree.query(np.asarray(nilearn_coords, dtype=np.float64), k=1)
    indices = np.asarray(indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    expected = np.arange(indices.shape[0], dtype=np.int64)
    identity_order = bool(
        np.array_equal(indices, expected)
        and np.all(distances <= atol)
        and len(np.unique(indices)) == indices.shape[0]
    )
    return {
        "identity_order": identity_order,
        "max_distance": float(distances.max()) if distances.size else 0.0,
        "mean_distance": float(distances.mean()) if distances.size else 0.0,
    }


def _agreement_pct(
    annot_labels: np.ndarray,
    reference_labels: np.ndarray,
) -> float:
    mask = (annot_labels > 0) & (reference_labels > 0)
    if not np.any(mask):
        return 0.0
    matches = int(np.sum(annot_labels[mask] == reference_labels[mask]))
    return float(matches) / float(np.sum(mask))


def reorder_hemisphere_labels_by_coords(
    labels: np.ndarray,
    *,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    atol: float = COORD_MATCH_ATOL,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reorder labels from source vertex order to target (Nilearn) vertex order."""
    from scipy.spatial import cKDTree

    if labels.shape[0] != source_coords.shape[0] or target_coords.shape[0] != labels.shape[0]:
        raise ValueError("labels and coordinate arrays must share the same vertex count")

    tree = cKDTree(np.asarray(source_coords, dtype=np.float64))
    distances, indices = tree.query(np.asarray(target_coords, dtype=np.float64), k=1)
    indices = np.asarray(indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)

    if np.any(distances > atol):
        bad = int(np.sum(distances > atol))
        raise ValueError(
            f"Coordinate reorder failed: {bad} vertices exceed atol={atol} "
            f"(max distance {float(distances.max()):.4f})"
        )
    if len(np.unique(indices)) != indices.shape[0]:
        raise ValueError("Coordinate reorder is not bijective; annot and Nilearn meshes may differ")

    reordered = labels[indices].astype(np.int64, copy=True)
    return reordered, {
        "reorder_applied": True,
        "max_distance": float(distances.max()),
        "mean_distance": float(distances.mean()),
    }


def build_volume_projected_labels(
    *,
    n_rois: int,
    yeo_networks: int,
    resolution_mm: int,
    data_dir: Path | None,
    name_to_parcel_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project volumetric Schaefer to fsaverage5 and remap to the same global parcel ids as annot."""
    from nilearn import datasets
    from nilearn.surface import vol_to_surf

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm,
        data_dir=str(data_dir) if data_dir is not None else None,
    )
    decoded = [_normalise_label(_decode_label(item)) for item in atlas.labels]
    vol_index_to_name = {idx: label for idx, label in enumerate(decoded)}

    def _project(surf_mesh: str) -> np.ndarray:
        projected = vol_to_surf(
            atlas.maps,
            surf_mesh,
            interpolation="nearest_most_frequent",
            radius=3.0,
        )
        arr = np.asarray(projected)
        if arr.ndim != 1:
            arr = np.squeeze(arr)
        return np.rint(arr).astype(np.int64)

    def _remap_projected(raw: np.ndarray) -> np.ndarray:
        out = np.zeros_like(raw, dtype=np.int64)
        for vertex, vol_idx in enumerate(raw.tolist()):
            vol_int = int(vol_idx)
            if vol_int < 0:
                continue
            name = vol_index_to_name.get(vol_int)
            if not name:
                continue
            parcel_id = name_to_parcel_id.get(name)
            if parcel_id is not None:
                out[vertex] = int(parcel_id)
        return out

    lh = _remap_projected(_project(fsaverage.pial_left))
    rh = _remap_projected(_project(fsaverage.pial_right))
    return lh, rh


def verify_annot_vertex_order(
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
    *,
    lh_coords: np.ndarray,
    rh_coords: np.ndarray,
    n_rois: int,
    yeo_networks: int,
    resolution_mm: int,
    data_dir: Path | None,
    name_to_parcel_id: dict[str, int],
    skip_projection_check: bool = False,
    fs_lh_coords: np.ndarray | None = None,
    fs_rh_coords: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Verify or reorder annot labels to match Nilearn fsaverage5 vertex order."""
    if lh_labels.shape[0] != EXPECTED_FSAVERAGE5_HEMI_VERTICES:
        raise ValueError(f"Expected {EXPECTED_FSAVERAGE5_HEMI_VERTICES} LH labels")
    if rh_labels.shape[0] != EXPECTED_FSAVERAGE5_HEMI_VERTICES:
        raise ValueError(f"Expected {EXPECTED_FSAVERAGE5_HEMI_VERTICES} RH labels")

    lh_out = lh_labels.astype(np.int64, copy=True)
    rh_out = rh_labels.astype(np.int64, copy=True)
    report: dict[str, Any] = {
        "projection_agreement_threshold": PROJECTION_AGREEMENT_THRESHOLD,
        "coord_match_atol": COORD_MATCH_ATOL,
    }

    if not skip_projection_check:
        if fs_lh_coords is None or fs_rh_coords is None:
            if data_dir is None:
                raise ValueError(
                    "Vertex order proof needs CBIG fsaverage5 lh.pial/rh.pial under --data-dir."
                )
            fs_coords = load_cbig_fsaverage5_pial_coords(data_dir)
            fs_lh_coords = fs_coords["lh"]
            fs_rh_coords = fs_coords["rh"]

        lh_mesh_proof = _mesh_coords_share_vertex_order(fs_lh_coords, lh_coords)
        rh_mesh_proof = _mesh_coords_share_vertex_order(fs_rh_coords, rh_coords)
        report["lh_mesh_vertex_order"] = lh_mesh_proof
        report["rh_mesh_vertex_order"] = rh_mesh_proof
        mesh_order_verified = lh_mesh_proof["identity_order"] and rh_mesh_proof["identity_order"]

        proj_lh, proj_rh = build_volume_projected_labels(
            n_rois=n_rois,
            yeo_networks=yeo_networks,
            resolution_mm=resolution_mm,
            data_dir=data_dir,
            name_to_parcel_id=name_to_parcel_id,
        )
        lh_pct = _agreement_pct(lh_out, proj_lh)
        rh_pct = _agreement_pct(rh_out, proj_rh)
        report["lh_projection_agreement_pct"] = lh_pct
        report["rh_projection_agreement_pct"] = rh_pct
        report["mean_projection_agreement_pct"] = float((lh_pct + rh_pct) / 2.0)
        report["projection_agreement_note"] = (
            "Diagnostic only: surface .annot vs volumetric projection; low overlap is expected."
        )

        if mesh_order_verified:
            report["status"] = "annot_matches_nilearn_projection"
            report["reorder_applied"] = False
            return lh_out, rh_out, report

        lh_reordered, lh_meta = reorder_hemisphere_labels_by_coords(
            lh_out, source_coords=fs_lh_coords, target_coords=lh_coords
        )
        rh_reordered, rh_meta = reorder_hemisphere_labels_by_coords(
            rh_out, source_coords=fs_rh_coords, target_coords=rh_coords
        )
        report["lh_reorder"] = lh_meta
        report["rh_reorder"] = rh_meta
        lh_mesh2 = _mesh_coords_share_vertex_order(fs_lh_coords, lh_coords)
        rh_mesh2 = _mesh_coords_share_vertex_order(fs_rh_coords, rh_coords)
        if lh_mesh2["identity_order"] and rh_mesh2["identity_order"]:
            lh_pct2 = _agreement_pct(lh_reordered, proj_lh)
            rh_pct2 = _agreement_pct(rh_reordered, proj_rh)
            report["lh_projection_agreement_pct_after_reorder"] = lh_pct2
            report["rh_projection_agreement_pct_after_reorder"] = rh_pct2
            report["status"] = "annot_reordered_by_coordinate_map"
            report["reorder_applied"] = True
            return lh_reordered, rh_reordered, report

        raise ValueError(
            "Vertex order proof failed: CBIG FreeSurfer pial coords do not map 1:1 to Nilearn "
            f"fsaverage5 (LH identity={lh_mesh_proof['identity_order']}, "
            f"RH identity={rh_mesh_proof['identity_order']}). "
            "Verify CBIG .annot/.pial files and docs/atlas-setup/cbig-schaefer2018-fsaverage5.md."
        )

    report["status"] = "annot_matches_nilearn_projection"
    report["reorder_applied"] = False
    return lh_out, rh_out, report


def resolve_annot_paths(
    *,
    n_rois: int,
    yeo_networks: int,
    data_dir: Path | None,
    annot_lh: Path | None,
    annot_rh: Path | None,
) -> tuple[Path, Path]:
    if (annot_lh is None) ^ (annot_rh is None):
        raise ValueError("Provide both annot_lh and annot_rh together.")

    if annot_lh is not None and annot_rh is not None:
        if not annot_lh.is_file():
            raise FileNotFoundError(f"Left annot not found: {annot_lh}")
        if not annot_rh.is_file():
            raise FileNotFoundError(f"Right annot not found: {annot_rh}")
        return annot_lh, annot_rh

    if data_dir is None:
        raise FileNotFoundError(
            "Surface .annot files not found. Download the LH/RH pair per "
            "docs/atlas-setup/cbig-schaefer2018-fsaverage5.md or pass --annot-lh/--annot-rh."
        )

    pattern = re.compile(
        rf"(?i)schaefer2018_{n_rois}parcels_{yeo_networks}networks_order\.annot$"
    )
    lh_candidates: list[Path] = []
    rh_candidates: list[Path] = []
    if data_dir.exists():
        for path in data_dir.rglob("*.annot"):
            if not pattern.search(path.name):
                continue
            lower = path.name.lower()
            if lower.startswith("lh."):
                lh_candidates.append(path)
            elif lower.startswith("rh."):
                rh_candidates.append(path)

    if lh_candidates and rh_candidates:
        return sorted(lh_candidates)[0], sorted(rh_candidates)[0]

    raise FileNotFoundError(
        f"No Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order.annot pair under {data_dir}. "
        "See docs/atlas-setup/cbig-schaefer2018-fsaverage5.md."
    )


def report_sha256(report: dict[str, Any]) -> str:
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_alignment_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {**report, "report_sha256": report_sha256(report)}
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(path)
    return report
