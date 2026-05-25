#!/usr/bin/env python3
"""Build atlas-backed vertex_regions.csv from Schaefer 2018.

The current TribeV2/NeuroEmo ROI training path expects a dense vertex table:

    vertex_index, parcel_id, parcel_label, yeo_network_id, yeo_network_name, hemisphere

This script fetches Schaefer 2018 from nilearn, projects the volumetric labels to
fsaverage5, and writes that table in TribeV2 order: left hemisphere vertices
first, then right hemisphere vertices.

Usage:
    python scripts/build_schaefer_vertex_regions.py --n-rois 400 --yeo-networks 7
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERTEX_CSV = PROJECT_ROOT / "configs" / "vertex_regions.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "parcellation_manifest.yaml"
DEFAULT_ATLAS_DATA_DIR = PROJECT_ROOT / "scout_data" / "atlases"
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


def _network_name(label: str) -> str:
    parts = label.split("_")
    if len(parts) >= 4 and parts[0].endswith("Networks") and parts[1] in {"LH", "RH"}:
        return "_".join(parts[2:-1]).strip()
    return parts[-2].strip() if len(parts) >= 2 else ""


def _atlas_labels(labels: Any) -> dict[int, str]:
    decoded = [_normalise_label(_decode_label(item)) for item in labels]
    return {idx: label for idx, label in enumerate(decoded)}


def _network_id_map(labels: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for label in labels:
        name = _network_name(label)
        if name and name not in mapping:
            mapping[name] = len(mapping) + 1
    return mapping


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
    """Fill background labels from the nearest assigned vertex in the same hemi."""
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


def _resolve_surface_label_paths(
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

    roots = [
        data_dir,
        data_dir / "schaefer_2018",
        data_dir / "schaefer_2018" / "fsaverage5",
    ]
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
        "artifact_sha256": {
            "vertex_regions_csv": _sha256(vertex_csv),
            "labels_gii_lh": _sha256(labels_gii_lh) if labels_gii_lh is not None and labels_gii_lh.is_file() else "",
            "labels_gii_rh": _sha256(labels_gii_rh) if labels_gii_rh is not None and labels_gii_rh.is_file() else "",
        },
        "inputs": {
            "labels_gii_lh": str(labels_gii_lh) if labels_gii_lh is not None else "",
            "labels_gii_rh": str(labels_gii_rh) if labels_gii_rh is not None else "",
            "nilearn_fetcher": source_metadata.get("nilearn_fetcher", ""),
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
    labels_gii_lh: Path | None,
    labels_gii_rh: Path | None,
) -> None:
    from nilearn import datasets

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    surface_lh, surface_rh = _resolve_surface_label_paths(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        data_dir=data_dir,
        labels_gii_lh=labels_gii_lh,
        labels_gii_rh=labels_gii_rh,
    )

    label_by_id: dict[int, str] = {}
    source_metadata: dict[str, Any]
    if surface_lh is not None and surface_rh is not None:
        lh_labels, lh_label_map = _load_label_gifti(surface_lh)
        rh_labels, rh_label_map = _load_label_gifti(surface_rh)
        label_by_id.update(lh_label_map)
        label_by_id.update(rh_label_map)
        lh_labels, n_lh_filled = _fill_unassigned_nearest(lh_labels, fsaverage.pial_left)
        rh_labels, n_rh_filled = _fill_unassigned_nearest(rh_labels, fsaverage.pial_right)
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
            "labels_gii_lh": str(surface_lh),
            "labels_gii_rh": str(surface_rh),
            "nilearn_fetcher": "",
        }
    else:
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

    combined = np.concatenate([lh_labels, rh_labels], axis=0)

    if combined.shape[0] != EXPECTED_VERTICES:
        raise ValueError(f"Expected {EXPECTED_VERTICES} combined vertices, got {combined.shape}")
    if np.any(combined <= 0):
        n_zero = int(np.sum(combined <= 0))
        raise ValueError(
            f"Projected atlas produced {n_zero} unassigned/background vertices. "
            "Use a surface-native label source or adjust projection settings before training."
        )

    ordered_labels = [
        label_by_id.get(i, f"parcel_{i}")
        for i in range(1, n_rois + 1)
    ]
    network_ids = _network_id_map(ordered_labels)
    rows: list[dict[str, Any]] = []
    for vertex_index, parcel_id in enumerate(combined.tolist()):
        parcel_label = label_by_id.get(int(parcel_id), f"parcel_{parcel_id}")
        network_name = _network_name(parcel_label)
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

    _write_manifest(
        manifest,
        vertex_csv=output_csv,
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        labels=ordered_labels,
        source_metadata=source_metadata,
        labels_gii_lh=surface_lh,
        labels_gii_rh=surface_rh,
    )

    print(f"Vertex CSV: {output_csv}")
    print(f"Manifest:  {manifest}")
    print(f"Atlas:     Schaefer2018 {n_rois} parcels, {yeo_networks} networks")
    print(f"Vertices:  {len(rows)}")
    print(f"Label src: {source_metadata['label_source_kind']}")
    print(f"Filled:    {n_lh_filled + n_rh_filled} unassigned atlas vertices")
    print(f"CSV sha:   {_sha256(output_csv)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rois", type=int, default=400, choices=(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000))
    parser.add_argument("--yeo-networks", type=int, default=7, choices=(7, 17))
    parser.add_argument("--resolution-mm", type=int, default=1, choices=(1, 2))
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_VERTEX_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ATLAS_DATA_DIR)
    parser.add_argument("--labels-gii-lh", type=Path, default=None, help="Optional surface-native left hemisphere Schaefer label.gii")
    parser.add_argument("--labels-gii-rh", type=Path, default=None, help="Optional surface-native right hemisphere Schaefer label.gii")
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
        labels_gii_lh=args.labels_gii_lh,
        labels_gii_rh=args.labels_gii_rh,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
