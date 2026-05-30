#!/usr/bin/env python3
"""Verify CBIG Schaefer .annot labels align with Nilearn fsaverage5 vertex order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.neuroEmoCode.schaefer_surface_labels import (
    DEFAULT_ALIGNMENT_REPORT,
    load_freesurfer_annot,
    load_nilearn_fsaverage5_coords,
    remap_to_contiguous_parcel_ids,
    resolve_annot_paths,
    verify_annot_vertex_order,
    write_alignment_report,
)

DEFAULT_DATA_DIR = PROJECT_ROOT / "scout_data" / "atlases" / "cbig_schaefer2018"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rois", type=int, default=400, choices=(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000))
    parser.add_argument("--yeo-networks", type=int, default=7, choices=(7, 17))
    parser.add_argument("--resolution-mm", type=int, default=1, choices=(1, 2))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--annot-lh", type=Path, default=None)
    parser.add_argument("--annot-rh", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_ALIGNMENT_REPORT)
    parser.add_argument(
        "--skip-projection-check",
        action="store_true",
        help="Test-only: skip comparison against volumetric projection labels.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    annot_lh, annot_rh = resolve_annot_paths(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        data_dir=args.data_dir,
        annot_lh=args.annot_lh,
        annot_rh=args.annot_rh,
    )
    lh_raw, lh_map = load_freesurfer_annot(annot_lh)
    rh_raw, rh_map = load_freesurfer_annot(annot_rh)
    lh_labels, rh_labels, parcel_id_to_name, _ordered = remap_to_contiguous_parcel_ids(
        lh_raw,
        rh_raw,
        lh_map,
        rh_map,
        n_rois=args.n_rois,
    )
    name_to_parcel_id = {name: pid for pid, name in parcel_id_to_name.items()}
    coords = load_nilearn_fsaverage5_coords()

    lh_final, rh_final, proof = verify_annot_vertex_order(
        lh_labels,
        rh_labels,
        lh_coords=coords["lh"],
        rh_coords=coords["rh"],
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=args.resolution_mm,
        data_dir=args.data_dir,
        name_to_parcel_id=name_to_parcel_id,
        skip_projection_check=args.skip_projection_check,
    )
    report = write_alignment_report(
        args.output,
        {
            "annot_lh": str(annot_lh),
            "annot_rh": str(annot_rh),
            "n_rois": args.n_rois,
            "yeo_networks": args.yeo_networks,
            "vertex_order_proof": proof,
            "final_label_counts": {
                "lh_assigned": int((lh_final > 0).sum()),
                "rh_assigned": int((rh_final > 0).sum()),
            },
        },
    )
    print(f"Alignment report: {report['report_path']}")
    print(f"Status: {proof['status']}")
    if "lh_projection_agreement_pct" in proof:
        print(f"LH projection overlap (diagnostic): {proof['lh_projection_agreement_pct']:.1%}")
        print(f"RH projection overlap (diagnostic): {proof['rh_projection_agreement_pct']:.1%}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
