#!/usr/bin/env python3
"""Build fsLR Schaefer400 -> Yeo7 parcellation artifact for HCP CIFTI M0 extraction.

Usage:
  python scripts/build_hcp_parcellation_artifact.py --fixture
  python scripts/build_hcp_parcellation_artifact.py --dlabel path/to/Schaefer2018_400Parcels_7Networks_order.dlabel.nii

Without --dlabel, writes a compact test fixture (deterministic, for unit tests).
Production M0 on Modal should use a real Schaefer fsLR dlabel when available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.constants import YEO7_NAMES  # noqa: E402
from scout_core.parcellation import coarse_yeo7_from_subnetwork  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "configs/hcp_fslr_schaefer400_yeo7.npz"
HCP_CIFTI_BRAIN_AXIS = 91282


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_fixture_artifact(*, brain_axis_size: int = HCP_CIFTI_BRAIN_AXIS) -> dict[str, np.ndarray | str]:
    """Deterministic small fixture: cortical mask + modular parcel assignment."""
    rng = np.random.default_rng(42)
    cortical_mask = np.zeros(brain_axis_size, dtype=bool)
    # HCP CIFTI: left+right cortex ~first 64984 grayordinates
    n_cortex = min(64984, brain_axis_size)
    cortical_mask[:n_cortex] = True

    parcel_id = np.zeros(brain_axis_size, dtype=np.int16)
    cortex_idx = np.where(cortical_mask)[0]
    parcel_id[cortex_idx] = (cortex_idx % 400) + 1

    parcel_ids = np.arange(1, 401, dtype=np.int16)
    net_ids = np.array([(i % 7) + 1 for i in range(400)], dtype=np.int8)

    provenance = {
        "mode": "fixture",
        "brain_axis_size": brain_axis_size,
        "n_parcels": 400,
        "yeo7_names": list(YEO7_NAMES),
    }
    return {
        "brain_axis_size": np.int32(brain_axis_size),
        "cortical_mask": cortical_mask,
        "parcel_id": parcel_id,
        "parcel_ids": parcel_ids,
        "parcel_to_net_id": net_ids,
        "provenance_json": json.dumps(provenance),
    }


def build_from_dlabel(dlabel_path: Path, *, brain_axis_size: int = HCP_CIFTI_BRAIN_AXIS) -> dict[str, np.ndarray | str]:
    import nibabel as nib

    img = nib.load(str(dlabel_path))
    labels = np.asarray(img.get_fdata()).squeeze()
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    if labels.shape[0] != brain_axis_size:
        raise ValueError(
            f"dlabel length {labels.shape[0]} != expected CIFTI brain axis {brain_axis_size}"
        )

    parcel_id = labels.astype(np.int16)
    cortical_mask = parcel_id > 0
    unique_parcels = np.unique(parcel_id[cortical_mask])
    parcel_ids = np.sort(unique_parcels).astype(np.int16)

    # Schaefer 400 7-network: 400 parcels / 7 networks ≈ 57 parcels per network block
    net_ids = np.zeros(parcel_ids.shape[0], dtype=np.int8)
    for j, pid in enumerate(parcel_ids):
        block = int((int(pid) - 1) * 7 // max(int(parcel_ids.max()), 400))
        block = min(block, 6)
        coarse = YEO7_NAMES[block]
        net_ids[j] = list(YEO7_NAMES).index(coarse) + 1 if coarse in YEO7_NAMES else block + 1

    provenance = {
        "mode": "dlabel",
        "dlabel_path": str(dlabel_path),
        "brain_axis_size": brain_axis_size,
        "n_parcels": int(parcel_ids.size),
        "yeo7_names": list(YEO7_NAMES),
    }
    return {
        "brain_axis_size": np.int32(brain_axis_size),
        "cortical_mask": cortical_mask,
        "parcel_id": parcel_id,
        "parcel_ids": parcel_ids,
        "parcel_to_net_id": net_ids,
        "provenance_json": json.dumps(provenance),
    }


def write_artifact(artifact: dict[str, np.ndarray | str], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in artifact.items() if k != "provenance_json"}
    prov = str(artifact["provenance_json"])
    np.savez_compressed(output_path, provenance_json=np.array(prov), **arrays)
    return _sha256_bytes(output_path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HCP fsLR parcellation artifact")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixture", action="store_true", help="Write deterministic test fixture")
    parser.add_argument("--dlabel", type=Path, help="Schaefer fsLR dlabel.nii path")
    parser.add_argument("--brain-axis-size", type=int, default=HCP_CIFTI_BRAIN_AXIS)
    args = parser.parse_args()

    if args.dlabel:
        artifact = build_from_dlabel(args.dlabel, brain_axis_size=args.brain_axis_size)
    else:
        artifact = build_fixture_artifact(brain_axis_size=args.brain_axis_size)

    sha = write_artifact(artifact, args.output)
    print(f"Wrote {args.output} sha256={sha}")


if __name__ == "__main__":
    main()
