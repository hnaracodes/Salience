"""Harvard-Oxford subcortical atlas metadata aligned with TRIBE v2 subcortical head."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "subcortical_manifest.yaml"
DEFAULT_VOXEL_REGIONS_CSV = PROJECT_ROOT / "configs" / "subcortical_voxel_regions.csv"

N_SUBCORTICAL_VOXELS = 8802
SUBCORTICAL_REGION_LABELS = (
    "hippocampus",
    "lateral_ventricles",
    "amygdala",
    "thalamus",
    "caudate",
    "putamen",
    "pallidum",
    "accumbens",
)

# Harvard-Oxford label fragments -> manifest region id (matches tribev2/plotting/subcortical.py).
_HO_LABEL_TO_REGION: tuple[tuple[str, int], ...] = (
    ("hippocampus", 0),
    ("lateral ventricle", 1),
    ("amygdala", 2),
    ("thalamus", 3),
    ("caudate", 4),
    ("putamen", 5),
    ("pallidum", 6),
    ("accumbens", 7),
)


@lru_cache(maxsize=1)
def load_subcortical_manifest(manifest_path: str | None = None) -> dict[str, Any]:
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not path.is_file():
        return {
            "atlas_id": "harvard_oxford_subcortical_tribev2",
            "n_voxels": N_SUBCORTICAL_VOXELS,
            "region_labels": list(SUBCORTICAL_REGION_LABELS),
        }
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _index_from_sizes(sizes: list[int]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for rid, size in enumerate(sizes):
        parts.append(np.full(size, rid, dtype=np.int32))
    out = np.concatenate(parts)
    if out.shape[0] != N_SUBCORTICAL_VOXELS:
        raise ValueError(
            f"region index length {out.shape[0]} != expected {N_SUBCORTICAL_VOXELS}"
        )
    return out


def load_voxel_regions_csv(csv_path: Path | None = None) -> np.ndarray:
    """Load per-voxel region ids from CSV (columns: voxel_index, region_id)."""
    path = csv_path or DEFAULT_VOXEL_REGIONS_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Missing subcortical voxel region map: {path}")
    region_ids = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int32)
    if region_ids.ndim == 2:
        region_ids = region_ids[:, 1]
    if region_ids.shape[0] != N_SUBCORTICAL_VOXELS:
        raise ValueError(
            f"CSV region map length {region_ids.shape[0]} != {N_SUBCORTICAL_VOXELS}"
        )
    return region_ids.astype(np.int32)


def save_voxel_regions_csv(region_index: np.ndarray, csv_path: Path | None = None) -> Path:
    path = csv_path or DEFAULT_VOXEL_REGIONS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["voxel_index,region_id\n"]
    for i, rid in enumerate(region_index.astype(int)):
        lines.append(f"{i},{rid}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def build_region_index(
    manifest: dict[str, Any] | None = None,
    *,
    voxel_regions_csv: Path | None = None,
    use_fake_equal_split: bool = False,
) -> np.ndarray:
    """Return int array (8802,) mapping each voxel to region id 0..7."""
    if use_fake_equal_split:
        return _build_equal_split_index(manifest)

    csv_path = voxel_regions_csv or DEFAULT_VOXEL_REGIONS_CSV
    if csv_path.is_file():
        return load_voxel_regions_csv(csv_path)
    if voxel_regions_csv is not None:
        raise FileNotFoundError(
            f"Missing real subcortical voxel region map: {csv_path}. "
            "Run scripts/horikawaCode/extract_subcortical_voxel_regions.py, "
            "or pass use_fake_equal_split=True for the legacy fake-8 baseline."
        )

    manifest = manifest or load_subcortical_manifest()
    csv_from_manifest = manifest.get("voxel_regions_csv")
    if csv_from_manifest:
        alt = Path(csv_from_manifest)
        if not alt.is_absolute():
            alt = PROJECT_ROOT / alt
        if alt.is_file():
            return load_voxel_regions_csv(alt)

    raise FileNotFoundError(
        f"Missing real subcortical voxel region map: {csv_path}. "
        "Run scripts/horikawaCode/extract_subcortical_voxel_regions.py, "
        "or pass use_fake_equal_split=True for the legacy fake-8 baseline."
    )


def _build_equal_split_index(manifest: dict[str, Any] | None = None) -> np.ndarray:
    """Legacy equal-split fallback (anatomically incorrect; ablation control only)."""
    manifest = manifest or load_subcortical_manifest()
    labels = manifest.get("region_labels") or list(SUBCORTICAL_REGION_LABELS)
    counts = manifest.get("region_voxel_counts")
    if isinstance(counts, dict):
        sizes = [int(counts.get(lbl, 0)) for lbl in labels]
        if sum(sizes) == N_SUBCORTICAL_VOXELS:
            return _index_from_sizes(sizes)

    n_regions = len(labels)
    base, rem = divmod(N_SUBCORTICAL_VOXELS, n_regions)
    sizes = [base + (1 if i < rem else 0) for i in range(n_regions)]
    return _index_from_sizes(sizes)


def region_voxel_counts(region_index: np.ndarray) -> dict[str, int]:
    labels = list(SUBCORTICAL_REGION_LABELS)
    counts: dict[str, int] = {}
    for rid, lbl in enumerate(labels):
        counts[lbl] = int(np.sum(region_index == rid))
    return counts


def _ho_atlas_index_to_region(atlas_label_index: int, ho_labels: list[str]) -> int:
    if atlas_label_index <= 0 or atlas_label_index >= len(ho_labels):
        return -1
    name = ho_labels[atlas_label_index].lower()
    for fragment, rid in _HO_LABEL_TO_REGION:
        if fragment in name:
            return rid
    return -1


def build_region_index_from_tribev2_ho(*, expected_voxels: int = N_SUBCORTICAL_VOXELS) -> np.ndarray:
    """Build voxel->region map using tribev2's Harvard-Oxford subcortical mask (2mm, thr50)."""
    from tribev2.plotting.subcortical import cached_ho_atlas, get_subcortical_mask

    mask_img = get_subcortical_mask()
    mask_data = mask_img.get_fdata()
    ho = cached_ho_atlas(resolution="2mm")
    ho_labels = list(ho.labels)

    flat_indices = np.where(mask_data.ravel() > 0)[0]
    n_mask = flat_indices.shape[0]
    if n_mask < expected_voxels:
        raise ValueError(
            f"tribev2 HO mask has {n_mask} voxels, expected at least {expected_voxels}"
        )
    if n_mask != expected_voxels:
        # tribev2 plotting mask can include a few extra voxels; model head uses 8802 in-order.
        flat_indices = flat_indices[:expected_voxels]

    region_index = np.full(expected_voxels, -1, dtype=np.int32)
    for out_i, flat_i in enumerate(flat_indices):
        atlas_val = int(mask_data.ravel()[flat_i])
        rid = _ho_atlas_index_to_region(atlas_val, ho_labels)
        if rid < 0:
            raise ValueError(f"Unmapped HO atlas index {atlas_val} ({ho_labels[atlas_val]})")
        region_index[out_i] = rid

    unmapped = int(np.sum(region_index < 0))
    if unmapped:
        raise ValueError(f"{unmapped} voxels could not be mapped to a region")
    return region_index


def build_region_index_from_nilearn_ho(*, expected_voxels: int = N_SUBCORTICAL_VOXELS) -> np.ndarray:
    """Nilearn-only fallback mirroring tribev2.plotting.subcortical.get_subcortical_mask."""
    from nilearn import datasets
    import nibabel as nib

    atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm")
    excluded = ["Cortex", "White", "Stem", "Background"]
    selected_indices = [
        i
        for i, label in enumerate(atlas.labels)
        if any(exc.lower() in label.lower() for exc in excluded)
    ]
    mask_data = atlas.maps.get_fdata().copy()
    mask_data[np.isin(mask_data, selected_indices)] = 0
    ho_labels = list(atlas.labels)

    flat_indices = np.where(mask_data.ravel() > 0)[0]
    if flat_indices.shape[0] < expected_voxels:
        raise ValueError(
            f"nilearn HO mask has {flat_indices.shape[0]} voxels, expected at least {expected_voxels}"
        )
    flat_indices = flat_indices[:expected_voxels]

    region_index = np.full(expected_voxels, -1, dtype=np.int32)
    for out_i, flat_i in enumerate(flat_indices):
        atlas_val = int(mask_data.ravel()[flat_i])
        rid = _ho_atlas_index_to_region(atlas_val, ho_labels)
        if rid < 0:
            raise ValueError(f"Unmapped HO atlas index {atlas_val}")
        region_index[out_i] = rid
    return region_index


def roi_means_from_voxels(
    preds: np.ndarray,
    region_index: np.ndarray | None = None,
    *,
    use_fake_equal_split: bool = False,
) -> np.ndarray:
    """Aggregate subcortical voxels to ROI means. preds shape (T, 8802) or (8802,)."""
    preds = np.asarray(preds, dtype=np.float32)
    if region_index is None:
        region_index = build_region_index(use_fake_equal_split=use_fake_equal_split)
    n_regions = int(region_index.max()) + 1

    if preds.ndim == 1:
        if preds.shape[0] != N_SUBCORTICAL_VOXELS:
            raise ValueError(f"expected {N_SUBCORTICAL_VOXELS} voxels, got {preds.shape[0]}")
        out = np.zeros(n_regions, dtype=np.float32)
        for rid in range(n_regions):
            mask = region_index == rid
            if np.any(mask):
                out[rid] = float(np.mean(preds[mask]))
        return out

    if preds.ndim != 2 or preds.shape[1] != N_SUBCORTICAL_VOXELS:
        raise ValueError(f"expected (T, {N_SUBCORTICAL_VOXELS}), got {preds.shape}")

    t_count = preds.shape[0]
    out = np.zeros((t_count, n_regions), dtype=np.float32)
    for rid in range(n_regions):
        mask = region_index == rid
        if np.any(mask):
            out[:, rid] = preds[:, mask].mean(axis=1)
    return out


def roi_stats_from_voxels(
    preds: np.ndarray,
    region_index: np.ndarray | None = None,
    *,
    use_fake_equal_split: bool = False,
) -> np.ndarray:
    """Per-region mean and std (16 features for 8 regions)."""
    preds = np.asarray(preds, dtype=np.float32)
    if region_index is None:
        region_index = build_region_index(use_fake_equal_split=use_fake_equal_split)
    means = roi_means_from_voxels(preds, region_index)
    n_regions = int(region_index.max()) + 1

    if preds.ndim == 1:
        stds = np.zeros(n_regions, dtype=np.float32)
        for rid in range(n_regions):
            mask = region_index == rid
            if np.any(mask):
                stds[rid] = float(np.std(preds[mask]))
        return np.concatenate([means, stds], axis=0)

    t_count = preds.shape[0]
    std_block = np.zeros((t_count, n_regions), dtype=np.float32)
    for rid in range(n_regions):
        mask = region_index == rid
        if np.any(mask):
            std_block[:, rid] = preds[:, mask].std(axis=1)
    return np.concatenate([means, std_block], axis=1).astype(np.float32)
