#!/usr/bin/env python3
"""Orchestrate demographic data prep steps (M2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.metadata_harmonize import harmonize_metadata  # noqa: E402
from data_prep.segment_clusters import segment_clusters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Demographic data prep orchestrator")
    parser.add_argument(
        "--raw-metadata-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data/demographic/raw_metadata",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data/demographic",
    )
    parser.add_argument(
        "--clusters-yaml",
        type=Path,
        default=PROJECT_ROOT / "configs/clusters.yaml",
    )
    parser.add_argument("--step", choices=["all", "harmonize", "segment"], default="all")
    args = parser.parse_args()

    meta_out = args.output_dir / "metadata_harmonized.parquet"
    if args.step in ("all", "harmonize"):
        if not args.raw_metadata_dir.is_dir():
            raise SystemExit(f"raw metadata dir not found: {args.raw_metadata_dir}")
        df = harmonize_metadata(args.raw_metadata_dir, meta_out)
        print(f"Harmonized {len(df)} subjects -> {meta_out}")

    if args.step in ("all", "segment"):
        if not meta_out.is_file():
            raise SystemExit(f"Run harmonize first; missing {meta_out}")
        members = segment_clusters(meta_out, args.clusters_yaml, args.output_dir)
        print(f"Cluster members: {len(members)} rows")


if __name__ == "__main__":
    main()
