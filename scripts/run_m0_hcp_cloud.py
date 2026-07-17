#!/usr/bin/env python3
"""Launch cloud M0 viability extraction on Modal (HCP S3 + optional Cam-CAN).

Prerequisites:
  1. BALSA/ConnectomeDB HCP Open Access + AWS S3 credentials
  2. modal secret create hcp-aws-secret AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...

Usage:
  python scripts/run_m0_hcp_cloud.py --preflight
  python scripts/run_m0_hcp_cloud.py --phase hcp --subjects 5
  python scripts/run_m0_hcp_cloud.py --phase hcp_camcan
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _modal_cmd() -> list[str]:
    """Resolve Modal CLI (venv Scripts on Windows)."""
    venv_modal = PROJECT_ROOT / ".venv311" / "Scripts" / "modal.exe"
    if venv_modal.is_file():
        return [str(venv_modal)]
    venv_modal = PROJECT_ROOT / ".venv" / "Scripts" / "modal.exe"
    if venv_modal.is_file():
        return [str(venv_modal)]
    return ["modal"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud M0 via Modal + HCP S3")
    parser.add_argument("--preflight", action="store_true", help="Run S3 preflight only")
    parser.add_argument(
        "--phase",
        choices=["hcp", "hcp_camcan"],
        default="hcp",
        help="hcp=HCP only (not production gate); hcp_camcan=merged multi-site",
    )
    parser.add_argument("--subjects", type=int, default=None, help="Limit subjects (pilot)")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument(
        "--download",
        action="store_true",
        help="After run, modal volume get subjects.csv and m0_report.json",
    )
    args = parser.parse_args()

    modal_py = PROJECT_ROOT / "salience" / "modal_app" / "m0_hcp_cloud.py"

    if args.preflight:
        cmd = [*_modal_cmd(), "run", str(modal_py), "--cmd", "preflight"]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        raise SystemExit(result.returncode)

    cmd = [
        *_modal_cmd(),
        "run",
        str(modal_py),
        "--cmd",
        "pipeline",
        f"--phase={args.phase}",
        f"--n-permutations={args.n_permutations}",
    ]
    if args.subjects:
        cmd.append(f"--max-subjects={args.subjects}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)

    if args.download:
        out_dir = PROJECT_ROOT / "scout_data" / "demographic" / "viability"
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                *_modal_cmd(),
                "volume",
                "get",
                "tribe-demographic-mux-vol",
                "viability/subjects.csv",
                str(out_dir / "subjects.csv"),
                "--force",
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        subprocess.run(
            [
                *_modal_cmd(),
                "volume",
                "get",
                "tribe-demographic-mux-vol",
                "viability/m0_report.json",
                str(out_dir / "m0_report.json"),
                "--force",
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        print(f"Downloaded results to {out_dir}")

    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
