#!/usr/bin/env python3
"""Build atlas-backed vertex_regions.csv from Schaefer 2018.

The TribeV2/NeuroEmo ROI training path expects a dense vertex table:

    vertex_index, parcel_id, parcel_label, yeo_network_id, yeo_network_name, hemisphere

Default: load CBIG FreeSurfer5.3 surface `.annot` files on fsaverage5 (via nibabel).
Fallback: volumetric projection only with --allow-volume-projection-fallback.

Usage:
    python scripts/neuroEmoCode/build_schaefer_vertex_regions.py --n-rois 400 --yeo-networks 7 \\
        --data-dir scout_data/atlases/cbig_schaefer2018
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from scout_core.neuroEmoCode.schaefer_surface_labels import (
    DEFAULT_ALIGNMENT_REPORT,
    MAX_FILLED_UNASSIGNED_SURFACE,
    MAX_FILLED_UNASSIGNED_SURFACE_ANNOT,
    load_freesurfer_annot,
    load_nilearn_fsaverage5_coords,
    network_name_from_label,
    remap_to_contiguous_parcel_ids,
    resolve_annot_paths,
    verify_annot_vertex_order,
    write_alignment_report,
)
from scout_core.vertex_equivalence import (
    canonical_vertex_equivalence_report_path,
    load_vertex_equivalence_report,
    vertex_equivalence_reference,
)

DEFAULT_VERTEX_CSV = PROJECT_ROOT / "configs" / "vertex_regions.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "parcellation_manifest.yaml"
DEFAULT_ATLAS_DATA_DIR = PROJECT_ROOT / "scout_data" / "atlases" / "cbig_schaefer2018"
EXPECTED_HEMI_VERTICES = 10242
EXPECTED_VERTICES = EXPECTED_HEMI_VERTICES * 2


def _decode_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalise_label(label: str) -> str:
    label = label.strip()
    if label.startswith("b'") and label.endswith("'"):
        label = label[2:-1]
    return label


def _network_id_map(labels: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for label in labels:
        name = network_name_from_label(label)
        if name and name not in mapping:
            mapping[name] = len(mapping) + 1
    return mapping


def _atlas_labels(labels: Any) -> dict[int, str]:
    decoded = [_normalise_label(_decode_label(item)) for item in labels]
    return {idx: label for idx, label in enumerate(decoded)}


def _project_hemi(labels_img: Any, surf_mesh: str) -> np.ndarray:
    from nilearn.surface import vol_to_surf

    projected = vol_to_surf(
        labels_img,
        surf_mesh,
        interpolation="nearest_most_frequent",
        radius=3.0,
    )
    arr = np.asarray(projected)
    if arr.ndim != 1:
        arr = np.squeeze(arr)
    if arr.shape[0] != EXPECTED_HEMI_VERTICES:
        raise ValueError(f"Expected {EXPECTED_HEMI_VERTICES} hemi vertices, got {arr.shape}")
    return np.rint(arr).astype(np.int64)


def _surface_coords(surf_mesh: str) -> np.ndarray:
    import nibabel as nib

    img = nib.load(surf_mesh)
    coords = np.asarray(img.darrays[0].data, dtype=np.float32)
    if coords.shape[0] != EXPECTED_HEMI_VERTICES:
        raise ValueError(f"Expected {EXPECTED_HEMI_VERTICES} surface coordinates, got {coords.shape}")
    return coords


def _fill_unassigned_nearest(labels: np.ndarray, surf_mesh: str) -> tuple[np.ndarray, int]:
    arr = np.asarray(labels, dtype=np.int64).copy()
    missing = np.flatnonzero(arr <= 0)
    if missing.size == 0:
        return arr, 0

    assigned = np.flatnonzero(arr > 0)
    if assigned.size == 0:
        raise ValueError("Cannot fill unassigned atlas labels because no assigned vertices exist.")

    from scipy.spatial import cKDTree

    coords = _surface_coords(surf_mesh)
    tree = cKDTree(coords[assigned])
    _, nearest = tree.query(coords[missing], k=1)
    arr[missing] = arr[assigned[nearest]]
    return arr, int(missing.size)


def _load_label_gifti(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    import nibabel as nib

    img = nib.load(str(path))
    if not getattr(img, "darrays", None):
        raise ValueError(f"GIFTI label file has no data arrays: {path}")
    arr = np.asarray(img.darrays[0].data, dtype=np.int64)
    if arr.ndim != 1 or arr.shape[0] != EXPECTED_HEMI_VERTICES:
        raise ValueError(
            f"Expected {EXPECTED_HEMI_VERTICES} vertices in {path}, got shape {arr.shape}"
        )

    label_by_id: dict[int, str] = {}
    label_table = getattr(img, "labeltable", None)
    if label_table is not None:
        for item in getattr(label_table, "labels", []) or []:
            key = int(getattr(item, "key", 0))
            label = _normalise_label(_decode_label(getattr(item, "label", "")))
            if key > 0 and label:
                label_by_id[key] = label
    return arr, label_by_id


def _resolve_label_gifti_paths(
    *,
    n_rois: int,
    yeo_networks: int,
    data_dir: Path | None,
    labels_gii_lh: Path | None,
    labels_gii_rh: Path | None,
) -> tuple[Path | None, Path | None]:
    if (labels_gii_lh is None) ^ (labels_gii_rh is None):
        raise ValueError("Provide both --labels-gii-lh and --labels-gii-rh together.")

    if labels_gii_lh is not None and labels_gii_rh is not None:
        if not labels_gii_lh.is_file():
            raise FileNotFoundError(f"Left hemisphere label GIFTI not found: {labels_gii_lh}")
        if not labels_gii_rh.is_file():
            raise FileNotFoundError(f"Right hemisphere label GIFTI not found: {labels_gii_rh}")
        return labels_gii_lh, labels_gii_rh

    if data_dir is None:
        return None, None

    roots = [data_dir, data_dir / "schaefer_2018", data_dir / "schaefer_2018" / "fsaverage5"]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.label.gii"))

    token = f"schaefer2018_{n_rois}parcels_{yeo_networks}networks"
    lh_candidates = sorted(
        {
            path
            for path in files
            if token in path.name.lower() and ("lh" in path.name.lower() or "left" in path.name.lower())
        }
    )
    rh_candidates = sorted(
        {
            path
            for path in files
            if token in path.name.lower() and ("rh" in path.name.lower() or "right" in path.name.lower())
        }
    )
    if lh_candidates and rh_candidates:
        return lh_candidates[0], rh_candidates[0]
    return None, None


def _write_vertex_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "vertex_index",
        "parcel_id",
        "parcel_label",
        "yeo_network_id",
        "yeo_network_name",
        "hemisphere",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    from scout_core.parcellation import file_sha256

    return file_sha256(path)


def _write_manifest(
    path: Path,
    *,
    vertex_csv: Path,
    n_rois: int,
    yeo_networks: int,
    labels: list[str],
    source_metadata: dict[str, Any],
    labels_gii_lh: Path | None,
    labels_gii_rh: Path | None,
    annot_lh: Path | None,
    annot_rh: Path | None,
    vertex_equivalence: dict[str, Any],
    vertex_equivalence_report_path: Path,
    vertex_order_proof: dict[str, Any],
    alignment_report_path: Path,
) -> None:
    from scout_core.parcellation import CANONICAL_HEMISPHERE_ORDER, CANONICAL_VERTEX_ORDER

    atlas_id = f"schaefer2018_{n_rois}_{yeo_networks}networks_fsaverage5"
    manifest = {
        "mesh": "fsaverage5",
        "tribe_vertex_order": CANONICAL_VERTEX_ORDER,
        "atlas_id": atlas_id,
        "yeo7_map_version": f"{yeo_networks}Networks",
        "n_vertices_expected": EXPECTED_VERTICES,
        "n_parcels": n_rois,
        "hemisphere_order": CANONICAL_HEMISPHERE_ORDER,
        "source": source_metadata,
        "vertex_equivalence": vertex_equivalence,
        "vertex_order_proof": vertex_order_proof,
        "mesh_fingerprints": vertex_equivalence.get("mesh_fingerprints", {}),
        "artifact_sha256": {
            "vertex_regions_csv": _sha256(vertex_csv),
            "labels_gii_lh": _sha256(labels_gii_lh) if labels_gii_lh is not None and labels_gii_lh.is_file() else "",
            "labels_gii_rh": _sha256(labels_gii_rh) if labels_gii_rh is not None and labels_gii_rh.is_file() else "",
            "labels_annot_lh": _sha256(annot_lh) if annot_lh is not None and annot_lh.is_file() else "",
            "labels_annot_rh": _sha256(annot_rh) if annot_rh is not None and annot_rh.is_file() else "",
            "vertex_equivalence_report_json": vertex_equivalence.get("report_sha256", ""),
            "schaefer_annot_alignment_report_json": vertex_order_proof.get("report_sha256", ""),
        },
        "inputs": {
            "labels_gii_lh": str(labels_gii_lh) if labels_gii_lh is not None else "",
            "labels_gii_rh": str(labels_gii_rh) if labels_gii_rh is not None else "",
            "labels_annot_lh": str(annot_lh) if annot_lh is not None else "",
            "labels_annot_rh": str(annot_rh) if annot_rh is not None else "",
            "nilearn_fetcher": source_metadata.get("nilearn_fetcher", ""),
            "vertex_equivalence_report": str(vertex_equivalence_report_path),
            "schaefer_annot_alignment_report": str(alignment_report_path),
        },
        "labels": labels,
        "created_at_unix_ms": int(time.time() * 1000),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def build_schaefer_vertex_regions(
    *,
    n_rois: int,
    yeo_networks: int,
    resolution_mm: int,
    output_csv: Path,
    manifest: Path,
    data_dir: Path | None,
    annot_lh: Path | None,
    annot_rh: Path | None,
    labels_gii_lh: Path | None,
    labels_gii_rh: Path | None,
    vertex_equivalence_report: Path | None,
    alignment_report: Path | None,
    allow_volume_projection_fallback: bool,
    skip_vertex_order_proof: bool,
) -> None:
    from nilearn import datasets

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    mesh_coords = load_nilearn_fsaverage5_coords()

    annot_lh_path: Path | None = None
    annot_rh_path: Path | None = None
    gifti_lh_path: Path | None = None
    gifti_rh_path: Path | None = None
    label_by_id: dict[int, str] = {}
    source_metadata: dict[str, Any]
    vertex_order_proof: dict[str, Any] = {}
    alignment_report_path = (alignment_report or DEFAULT_ALIGNMENT_REPORT).expanduser().resolve()

    try:
        annot_lh_path, annot_rh_path = resolve_annot_paths(
            n_rois=n_rois,
            yeo_networks=yeo_networks,
            data_dir=data_dir,
            annot_lh=annot_lh,
            annot_rh=annot_rh,
        )
    except FileNotFoundError:
        annot_lh_path = None
        annot_rh_path = None

    if annot_lh_path is None or annot_rh_path is None:
        gifti_lh_path, gifti_rh_path = _resolve_label_gifti_paths(
            n_rois=n_rois,
            yeo_networks=yeo_networks,
            data_dir=data_dir,
            labels_gii_lh=labels_gii_lh,
            labels_gii_rh=labels_gii_rh,
        )

    if annot_lh_path is not None and annot_rh_path is not None:
        lh_raw, lh_map = load_freesurfer_annot(annot_lh_path)
        rh_raw, rh_map = load_freesurfer_annot(annot_rh_path)
        lh_labels, rh_labels, label_by_id, ordered_labels = remap_to_contiguous_parcel_ids(
            lh_raw,
            rh_raw,
            lh_map,
            rh_map,
            n_rois=n_rois,
        )
        name_to_parcel_id = {name: pid for pid, name in label_by_id.items()}
        if not skip_vertex_order_proof:
            lh_labels, rh_labels, proof = verify_annot_vertex_order(
                lh_labels,
                rh_labels,
                lh_coords=mesh_coords["lh"],
                rh_coords=mesh_coords["rh"],
                n_rois=n_rois,
                yeo_networks=yeo_networks,
                resolution_mm=resolution_mm,
                data_dir=data_dir,
                name_to_parcel_id=name_to_parcel_id,
            )
            alignment_payload = write_alignment_report(
                alignment_report_path,
                {
                    "annot_lh": str(annot_lh_path),
                    "annot_rh": str(annot_rh_path),
                    "n_rois": n_rois,
                    "yeo_networks": yeo_networks,
                    "vertex_order_proof": proof,
                },
            )
            vertex_order_proof = {
                **proof,
                "report_path": alignment_payload["report_path"],
                "report_sha256": alignment_payload["report_sha256"],
            }
        else:
            vertex_order_proof = {
                "status": "skipped_for_tests",
                "reorder_applied": False,
                "report_path": str(alignment_report_path),
                "report_sha256": "",
            }

        lh_labels, n_lh_filled = _fill_unassigned_nearest(lh_labels, fsaverage.pial_left)
        rh_labels, n_rh_filled = _fill_unassigned_nearest(rh_labels, fsaverage.pial_right)
        filled = n_lh_filled + n_rh_filled
        if filled > MAX_FILLED_UNASSIGNED_SURFACE_ANNOT:
            raise ValueError(
                f"Surface annot fill count {filled} exceeds limit {MAX_FILLED_UNASSIGNED_SURFACE_ANNOT}. "
                "Check annot files or medial-wall handling."
            )
        source_metadata = {
            "name": "Schaefer2018",
            "parcels": n_rois,
            "networks": yeo_networks,
            "mesh": "fsaverage5",
            "source_space": "fsaverage5_surface_annot_cbig",
            "label_source_kind": "surface_annot_cbig",
            "projection": "",
            "unassigned_vertex_fill": "nearest_assigned_vertex_within_hemisphere",
            "filled_unassigned_vertices": filled,
            "annot_lh": str(annot_lh_path),
            "annot_rh": str(annot_rh_path),
            "cbig_release": "FreeSurfer5.3",
            "nilearn_fetcher": "",
        }
    elif gifti_lh_path is not None and gifti_rh_path is not None:
        lh_labels, lh_label_map = _load_label_gifti(gifti_lh_path)
        rh_labels, rh_label_map = _load_label_gifti(gifti_rh_path)
        label_by_id.update(lh_label_map)
        label_by_id.update(rh_label_map)
        lh_labels, n_lh_filled = _fill_unassigned_nearest(lh_labels, fsaverage.pial_left)
        rh_labels, n_rh_filled = _fill_unassigned_nearest(rh_labels, fsaverage.pial_right)
        ordered_labels = [
            label_by_id.get(i, f"parcel_{i}")
            for i in range(1, n_rois + 1)
        ]
        source_metadata = {
            "name": "Schaefer2018",
            "parcels": n_rois,
            "networks": yeo_networks,
            "mesh": "fsaverage5",
            "source_space": "fsaverage5_surface_label_gifti",
            "label_source_kind": "surface_label_gifti",
            "projection": "",
            "unassigned_vertex_fill": "nearest_assigned_vertex_within_hemisphere",
            "filled_unassigned_vertices": n_lh_filled + n_rh_filled,
            "labels_gii_lh": str(gifti_lh_path),
            "labels_gii_rh": str(gifti_rh_path),
            "nilearn_fetcher": "",
        }
        vertex_order_proof = {
            "status": "surface_label_gifti_no_projection_check",
            "reorder_applied": False,
            "report_path": str(alignment_report_path),
            "report_sha256": "",
        }
    elif allow_volume_projection_fallback:
        warnings.warn(
            "Using deprecated volumetric Schaefer projection fallback. "
            "Download CBIG .annot files for surface-native labels.",
            stacklevel=2,
        )
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois,
            yeo_networks=yeo_networks,
            resolution_mm=resolution_mm,
            data_dir=str(data_dir) if data_dir is not None else None,
        )
        label_by_id = _atlas_labels(atlas.labels)
        lh_labels = _project_hemi(atlas.maps, fsaverage.pial_left)
        rh_labels = _project_hemi(atlas.maps, fsaverage.pial_right)
        lh_labels, n_lh_filled = _fill_unassigned_nearest(lh_labels, fsaverage.pial_left)
        rh_labels, n_rh_filled = _fill_unassigned_nearest(rh_labels, fsaverage.pial_right)
        ordered_labels = [
            label_by_id.get(i, f"parcel_{i}")
            for i in range(1, n_rois + 1)
        ]
        source_metadata = {
            "name": "Schaefer2018",
            "parcels": n_rois,
            "networks": yeo_networks,
            "mesh": "fsaverage5",
            "source_space": "MNI152_volume_projected_to_fsaverage5",
            "label_source_kind": "volume_projection_fallback",
            "projection": "nilearn.surface.vol_to_surf nearest_most_frequent radius=3.0",
            "unassigned_vertex_fill": "nearest_assigned_vertex_within_hemisphere",
            "filled_unassigned_vertices": n_lh_filled + n_rh_filled,
            "maps": str(atlas.maps),
            "nilearn_fetcher": "nilearn.datasets.fetch_atlas_schaefer_2018",
        }
        vertex_order_proof = {
            "status": "volume_projection_fallback",
            "reorder_applied": False,
            "report_path": str(alignment_report_path),
            "report_sha256": "",
        }
    else:
        raise FileNotFoundError(
            "No surface-native Schaefer labels found. Download the LH/RH .annot pair per "
            "docs/atlas-setup/cbig-schaefer2018-fsaverage5.md or pass --allow-volume-projection-fallback "
            "for diagnostic volumetric projection only."
        )

    if annot_lh_path is not None:
        ordered_labels = [
            label_by_id.get(i, f"parcel_{i}")
            for i in range(1, n_rois + 1)
        ]

    combined = np.concatenate([lh_labels, rh_labels], axis=0)

    if combined.shape[0] != EXPECTED_VERTICES:
        raise ValueError(f"Expected {EXPECTED_VERTICES} combined vertices, got {combined.shape}")
    if np.any(combined <= 0):
        n_zero = int(np.sum(combined <= 0))
        raise ValueError(
            f"Atlas produced {n_zero} unassigned/background vertices after fill. "
            "Check surface label source before training."
        )

    network_ids = _network_id_map(ordered_labels)
    rows: list[dict[str, Any]] = []
    for vertex_index, parcel_id in enumerate(combined.tolist()):
        parcel_label = label_by_id.get(int(parcel_id), f"parcel_{parcel_id}")
        network_name = network_name_from_label(parcel_label)
        rows.append(
            {
                "vertex_index": vertex_index,
                "parcel_id": int(parcel_id),
                "parcel_label": parcel_label,
                "yeo_network_id": network_ids.get(network_name, 0),
                "yeo_network_name": network_name,
                "hemisphere": "lh" if vertex_index < EXPECTED_HEMI_VERTICES else "rh",
            }
        )

    _write_vertex_csv(output_csv, rows)
    report_path = canonical_vertex_equivalence_report_path(vertex_equivalence_report)
    report = load_vertex_equivalence_report(report_path)
    vertex_equivalence = vertex_equivalence_reference(report, report_path=report_path)

    _write_manifest(
        manifest,
        vertex_csv=output_csv,
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        labels=ordered_labels,
        source_metadata=source_metadata,
        labels_gii_lh=gifti_lh_path,
        labels_gii_rh=gifti_rh_path,
        annot_lh=annot_lh_path,
        annot_rh=annot_rh_path,
        vertex_equivalence=vertex_equivalence,
        vertex_equivalence_report_path=report_path,
        vertex_order_proof=vertex_order_proof,
        alignment_report_path=alignment_report_path,
    )

    print(f"Vertex CSV: {output_csv}")
    print(f"Manifest:  {manifest}")
    print(f"Atlas:     Schaefer2018 {n_rois} parcels, {yeo_networks} networks")
    print(f"Vertices:  {len(rows)}")
    print(f"Label src: {source_metadata['label_source_kind']}")
    print(f"Filled:    {source_metadata['filled_unassigned_vertices']} unassigned atlas vertices")
    print(f"Order:     {vertex_order_proof.get('status')}")
    print(f"Proof:     {vertex_equivalence['proof_status']} ({vertex_equivalence['reference_kind']})")
    print(f"CSV sha:   {_sha256(output_csv)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rois", type=int, default=400, choices=(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000))
    parser.add_argument("--yeo-networks", type=int, default=7, choices=(7, 17))
    parser.add_argument("--resolution-mm", type=int, default=1, choices=(1, 2))
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_VERTEX_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ATLAS_DATA_DIR)
    parser.add_argument("--annot-lh", type=Path, default=None, help="CBIG left-hemisphere Schaefer .annot")
    parser.add_argument("--annot-rh", type=Path, default=None, help="CBIG right-hemisphere Schaefer .annot")
    parser.add_argument("--labels-gii-lh", type=Path, default=None, help="Optional surface label.gii (LH)")
    parser.add_argument("--labels-gii-rh", type=Path, default=None, help="Optional surface label.gii (RH)")
    parser.add_argument(
        "--allow-volume-projection-fallback",
        action="store_true",
        help="Diagnostic only: use Nilearn volumetric projection if .annot files are missing.",
    )
    parser.add_argument(
        "--skip-vertex-order-proof",
        action="store_true",
        help="Test-only: skip annot vs projection alignment gate.",
    )
    parser.add_argument(
        "--alignment-report",
        type=Path,
        default=None,
        help="Output path for schaefer annot alignment JSON report.",
    )
    parser.add_argument(
        "--vertex-equivalence-report",
        type=Path,
        default=None,
        help="Pinned vertex equivalence report path. Defaults to scout_data/neuroEmoCode/vertex_equivalence_report.json",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_schaefer_vertex_regions(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=args.resolution_mm,
        output_csv=args.output_csv,
        manifest=args.manifest,
        data_dir=args.data_dir,
        annot_lh=args.annot_lh,
        annot_rh=args.annot_rh,
        labels_gii_lh=args.labels_gii_lh,
        labels_gii_rh=args.labels_gii_rh,
        vertex_equivalence_report=args.vertex_equivalence_report,
        alignment_report=args.alignment_report,
        allow_volume_projection_fallback=args.allow_volume_projection_fallback,
        skip_vertex_order_proof=args.skip_vertex_order_proof,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
