#!/usr/bin/env python3
"""M0 gating experiment: demographic variance partition on movie-fMRI network features.

Usage:
  python scripts/run_demographic_viability.py --synthetic
  python scripts/run_demographic_viability.py --subjects-csv path/to/subjects.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.viability_partition import (  # noqa: E402
    generate_synthetic_viability_df,
    load_subjects_csv,
    run_viability_analysis,
    write_m0_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="M0 demographic viability partition")
    parser.add_argument(
        "--subjects-csv",
        type=Path,
        help="CSV with subject_id, site_id, age_band, sex, net_<Yeo7Name> columns",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on synthetic fixture data (pipeline test only; never gates production)",
    )
    parser.add_argument("--n-subjects", type=int, default=120, help="Synthetic N")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scout_data/demographic/viability/m0_report.json",
    )
    args = parser.parse_args()

    if args.synthetic:
        df = generate_synthetic_viability_df(
            n_subjects=args.n_subjects,
            random_state=args.random_state,
        )
        data_source = "synthetic_fixture"
        is_synthetic = True
    elif args.subjects_csv:
        csv_path = args.subjects_csv
        if not csv_path.is_file():
            raise SystemExit(f"CSV not found: {csv_path}")
        df = load_subjects_csv(csv_path)
        data_source = f"csv:{csv_path.name}"
        is_synthetic = False
    else:
        raise SystemExit("Provide --synthetic or --subjects-csv")

    report = run_viability_analysis(
        df,
        data_source=data_source,
        is_synthetic=is_synthetic,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )
    write_m0_report(report, args.output)

    print(f"M0 report: {args.output}")
    print(f"data_source: {report.data_source}")
    print(f"is_synthetic: {report.is_synthetic}")
    print(f"passed: {report.passed}")
    print(f"go_criteria: {report.go_criteria}")
    for net in report.networks:
        if net.network_name in ("Vis", "Default", "SalVentAttn"):
            print(
                f"  {net.network_name}: r2_demo|site={net.r2_demographic_given_site:.4f} "
                f"p_perm={net.permutation_p:.4f}"
            )
    if report.notes:
        for note in report.notes:
            print(f"NOTE: {note}")

    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
