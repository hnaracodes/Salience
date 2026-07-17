"""Subcortical fMRI prediction I/O and atlas helpers."""

from scout_core.subcortical.atlas import (
    N_SUBCORTICAL_VOXELS,
    SUBCORTICAL_REGION_LABELS,
    build_region_index,
    build_region_index_from_tribev2_ho,
    load_subcortical_manifest,
    load_voxel_regions_csv,
    region_voxel_counts,
    roi_means_from_voxels,
    roi_stats_from_voxels,
    save_voxel_regions_csv,
)
from scout_core.subcortical.io import (
    load_subcortical_preds,
    save_subcortical_npz,
    subcortical_npz_path,
)

__all__ = [
    "N_SUBCORTICAL_VOXELS",
    "SUBCORTICAL_REGION_LABELS",
    "build_region_index",
    "build_region_index_from_tribev2_ho",
    "load_subcortical_manifest",
    "load_voxel_regions_csv",
    "region_voxel_counts",
    "load_subcortical_preds",
    "roi_means_from_voxels",
    "roi_stats_from_voxels",
    "save_subcortical_npz",
    "save_voxel_regions_csv",
    "subcortical_npz_path",
]
