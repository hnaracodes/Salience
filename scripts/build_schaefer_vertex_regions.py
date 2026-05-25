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
EXPECTED_HEMI_VERTICES = 10242
EXPECTED_VERTICES = EXPECTED_HEMI_VERTICES * 2

YEO_NETWORK_ID = {
    "Vis": 1,
    "SomMot": 2,
    "DorsAttn": 3,
    "SalVentAttn": 4,
    "Limbic": 5,
    "Cont": 6,
    "Default": 7,
}


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
    for part in parts:
        if part in YEO_NETWORK_ID:
            return part
    return ""


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
    source_maps: str,
    labels: list[str],
    filled_unassigned_vertices: int,
) -> None:
    atlas_id = f"schaefer2018_{n_rois}_{yeo_networks}networks_fsaverage5"
    manifest = {
        "mesh": "fsaverage5",
        "tribe_vertex_order": "facebook/tribev2_lh_then_rh",
        "atlas_id": atlas_id,
        "yeo7_map_version": f"{yeo_networks}Networks",
        "n_vertices_expected": EXPECTED_VERTICES,
        "n_parcels": n_rois,
        "hemisphere_order": "lh_then_rh",
        "source": {
            "name": "Schaefer2018",
            "parcels": n_rois,
            "networks": yeo_networks,
            "mesh": "fsaverage5",
            "source_space": "MNI152_volume_projected_to_fsaverage5",
            "projection": "nilearn.surface.vol_to_surf nearest_most_frequent radius=3.0",
            "unassigned_vertex_fill": "nearest_assigned_vertex_within_hemisphere",
            "filled_unassigned_vertices": filled_unassigned_vertices,
            "maps": source_maps,
        },
        "artifact_sha256": {
            "vertex_regions_csv": _sha256(vertex_csv),
            "labels_gii_lh": "",
            "labels_gii_rh": "",
        },
        "inputs": {
            "labels_gii_lh": "",
            "labels_gii_rh": "",
            "nilearn_fetcher": "nilearn.datasets.fetch_atlas_schaefer_2018",
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
) -> None:
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm,
        data_dir=str(data_dir) if data_dir is not None else None,
    )
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    label_by_id = _atlas_labels(atlas.labels)

    lh_labels = _project_hemi(atlas.maps, fsaverage.pial_left)
    rh_labels = _project_hemi(atlas.maps, fsaverage.pial_right)
    lh_labels, n_lh_filled = _fill_unassigned_nearest(lh_labels, fsaverage.pial_left)
    rh_labels, n_rh_filled = _fill_unassigned_nearest(rh_labels, fsaverage.pial_right)
    combined = np.concatenate([lh_labels, rh_labels], axis=0)

    if combined.shape[0] != EXPECTED_VERTICES:
        raise ValueError(f"Expected {EXPECTED_VERTICES} combined vertices, got {combined.shape}")
    if np.any(combined <= 0):
        n_zero = int(np.sum(combined <= 0))
        raise ValueError(
            f"Projected atlas produced {n_zero} unassigned/background vertices. "
            "Use a surface-native label source or adjust projection settings before training."
        )

    rows: list[dict[str, Any]] = []
    for vertex_index, parcel_id in enumerate(combined.tolist()):
        parcel_label = label_by_id.get(int(parcel_id), f"parcel_{parcel_id}")
        network_name = _network_name(parcel_label)
        rows.append(
            {
                "vertex_index": vertex_index,
                "parcel_id": int(parcel_id),
                "parcel_label": parcel_label,
                "yeo_network_id": YEO_NETWORK_ID.get(network_name, 0),
                "yeo_network_name": network_name,
                "hemisphere": "lh" if vertex_index < EXPECTED_HEMI_VERTICES else "rh",
            }
        )

    _write_vertex_csv(output_csv, rows)

    ordered_labels = [
        label_by_id.get(i, f"parcel_{i}")
        for i in range(1, n_rois + 1)
    ]
    _write_manifest(
        manifest,
        vertex_csv=output_csv,
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        source_maps=str(atlas.maps),
        labels=ordered_labels,
        filled_unassigned_vertices=n_lh_filled + n_rh_filled,
    )

    print(f"Vertex CSV: {output_csv}")
    print(f"Manifest:  {manifest}")
    print(f"Atlas:     Schaefer2018 {n_rois} parcels, {yeo_networks} networks")
    print(f"Vertices:  {len(rows)}")
    print(f"Filled:    {n_lh_filled + n_rh_filled} unassigned projected vertices")
    print(f"CSV sha:   {_sha256(output_csv)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rois", type=int, default=400, choices=(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000))
    parser.add_argument("--yeo-networks", type=int, default=7, choices=(7, 17))
    parser.add_argument("--resolution-mm", type=int, default=1, choices=(1, 2))
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_VERTEX_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "scout_data" / "atlases")
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
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
