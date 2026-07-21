#!/usr/bin/env python3
"""Extract subject-level network features for M0 from aligned npz or preprocessed scans.

After downloading preprocessed fMRI from HCP/Cam-CAN/NNDb and running align_timeseries,
use this script to produce the CSV consumed by run_demographic_viability.py.

Usage:
  python scripts/extract_m0_network_features.py \\
    --aligned-root scout_data/demographic/aligned_subject_ts \\
    --metadata scout_data/demographic/metadata_harmonized.parquet \\
    --output scout_data/demographic/viability/subjects.csv

  python scripts/run_demographic_viability.py --subjects-csv scout_data/demographic/viability/subjects.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.extract_network_features import build_subjects_csv_from_aligned_npz  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract M0 network feature CSV")
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scout_data/demographic/viability/subjects.csv",
    )
    parser.add_argument(
        "--vertex-csv",
        type=Path,
        default=PROJECT_ROOT / "configs/vertex_regions.csv",
    )
    parser.add_argument(
        "--target-key",
        choices=["network_ts", "parcel_ts", "vertex_ts"],
        default="network_ts",
    )
    args = parser.parse_args()

    df = build_subjects_csv_from_aligned_npz(
        args.aligned_root,
        args.metadata,
        args.output,
        vertex_csv=args.vertex_csv,
        target_key=args.target_key,
    )
    print(f"Wrote {len(df)} subjects -> {args.output}")
    print("Next: python scripts/run_demographic_viability.py --subjects-csv", args.output)


if __name__ == "__main__":
    main()
