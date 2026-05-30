#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scout_core.vertex_equivalence import (
    ALL_PROOF_STATUSES,
    build_vertex_equivalence_report,
    canonical_vertex_equivalence_report_path,
    load_vertex_equivalence_report,
    validate_vertex_equivalence_summary,
    vertex_equivalence_reference,
    write_vertex_equivalence_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "scout_data" / "neuroEmoCode" / "vertex_equivalence_report.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bold-path", type=Path, default=None, help="Optional 4D BOLD NIfTI for projection comparison.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the JSON report.")
    parser.add_argument("--mode", choices=("modal", "local"), default="modal")
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--interpolation", choices=("linear", "nearest"), default="linear")
    parser.add_argument("--allclose-atol", type=float, default=1e-5)
    parser.add_argument(
        "--require-status",
        choices=ALL_PROOF_STATUSES,
        default=None,
        help="Exit non-zero unless the report reaches at least this proof_status.",
    )
    return parser


def _status_rank(value: str) -> int:
    order = {
        "contract_only": 0,
        "mesh_identity_verified": 1,
        "projection_equivalence_verified": 2,
    }
    return order.get(value, -1)


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = canonical_vertex_equivalence_report_path(args.output)

    if args.mode == "modal":
        raise SystemExit(
            "Modal mode must be launched via Modal's local entrypoint, not plain python:\n"
            "  modal run tribe.py::verify_vertex_equivalence "
            f"--bold-path {args.bold_path or '<path>'}\n"
            "Or run: scripts/run_vertex_equivalence_verification.ps1"
        )
    else:
        report = build_vertex_equivalence_report(
            bold_path=args.bold_path,
            mesh="fsaverage5",
            radius=args.radius,
            interpolation=args.interpolation,
            allclose_atol=args.allclose_atol,
            runtime_provenance={"runtime": "local"},
        )

    write_vertex_equivalence_report(report, output_path)
    print(json.dumps(report, indent=2))
    print(f"Report: {output_path}")
    print(
        json.dumps(
            vertex_equivalence_reference(report, report_path=output_path),
            indent=2,
        )
    )
    if args.require_status is not None and _status_rank(report["proof_status"]) < _status_rank(
        args.require_status
    ):
        raise SystemExit(
            f"vertex_equivalence proof_status={report['proof_status']!r} did not reach "
            f"required {args.require_status!r}"
        )
    # Final schema/policy sanity check for downstream consumers.
    validate_vertex_equivalence_summary(load_vertex_equivalence_report(output_path), allow_contract_only=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
