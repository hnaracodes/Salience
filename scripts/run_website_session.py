#!/usr/bin/env python3
"""Orchestrate website session pipeline stages.

Usage:
    python scripts/run_website_session.py --stage capture --script configs/walkthrough_scripts/localhost_demo.yaml
    python scripts/run_website_session.py --session-id <id> --stage tribe
    python scripts/run_website_session.py --session-id <id> --stage dual_track --baseline-session-id <bid>
    python scripts/run_website_session.py --session-id <id> --stage heatmaps --uniform-heatmap
    python scripts/run_website_session.py --session-id <id> --stage analyze --norm-id default_v1 --website
    python scripts/run_website_session.py --session-id <id> --stage narrative
    python scripts/run_website_session.py --session-id <id> --stage all --script ... --norm-id default_v1

Stages: capture | tribe | dual_track | heatmaps | analyze | narrative | all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from activation_store import SESSIONS_DIR
from scout_core.session_align import validate_session_dir

STAGES = ("capture", "tribe", "dual_track", "heatmaps", "analyze", "narrative", "export_viewer", "all")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _stage_capture(args: argparse.Namespace) -> str:
    if not args.script:
        raise SystemExit("--script required for capture stage")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "record_website_session.py"),
        "--script",
        str(args.script),
    ]
    if args.session_id:
        cmd.extend(["--session-id", args.session_id])
    _run(cmd)
    if args.session_id:
        return args.session_id
    sessions = sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in sessions:
        if (d / "session_manifest.json").is_file():
            return d.name
    raise SystemExit("Could not determine session_id after capture")


def _stage_tribe(session_id: str) -> None:
    _run(["modal", "run", "tribe.py::record_session", "--session-id", session_id])


def _stage_dual_track(session_id: str, args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_dual_track.py"), "--session-id", session_id]
    if args.baseline_session_id:
        cmd.extend(["--baseline-session-id", args.baseline_session_id])
    elif args.baseline_preds:
        cmd.extend(["--baseline-preds", str(args.baseline_preds)])
    _run(cmd)


def _stage_heatmaps(session_id: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "extract_section_heatmaps.py"),
        "--session-id",
        session_id,
    ]
    if args.uniform_heatmap:
        cmd.append("--uniform-heatmap")
    if args.no_modal:
        pass
    elif not args.uniform_heatmap:
        cmd.append("--modal")
    if args.refresh_sections:
        cmd.append("--refresh-sections")
    _run(cmd)


def _stage_analyze(session_id: str, args: argparse.Namespace) -> None:
    if not args.norm_id:
        raise SystemExit("--norm-id required for analyze stage")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analyze_session.py"),
        "--session-id",
        session_id,
        "--norm-id",
        args.norm_id,
    ]
    if args.website or args.sections:
        cmd.append("--sections")
    if args.ground:
        cmd.append("--ground")
    if args.website:
        cmd.append("--website")
    if args.with_heatmaps:
        cmd.append("--with-heatmaps")
    _run(cmd)


def _stage_narrative(session_id: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_session_narrative.py"),
        "--session-id",
        session_id,
    ]
    if args.llm_provider:
        cmd.extend(["--provider", args.llm_provider])
    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--script", type=Path, default=None)
    parser.add_argument("--norm-id", default=None)
    parser.add_argument("--baseline-session-id", default=None)
    parser.add_argument("--baseline-preds", type=Path, default=None)
    parser.add_argument("--uniform-heatmap", action="store_true")
    parser.add_argument("--no-modal", action="store_true", help="Heatmaps: frames only, no Modal")
    parser.add_argument("--refresh-sections", action="store_true")
    parser.add_argument("--sections", action="store_true")
    parser.add_argument("--ground", action="store_true")
    parser.add_argument("--website", action="store_true")
    parser.add_argument("--with-heatmaps", action="store_true")
    parser.add_argument("--llm-provider", default=None)
    args = parser.parse_args()

    session_id = args.session_id
    stages = (
        ["capture", "tribe", "dual_track", "heatmaps", "analyze", "narrative", "export_viewer"]
        if args.stage == "all"
        else [args.stage]
    )

    for stage in stages:
        if stage == "capture":
            session_id = _stage_capture(args)
            print(f"capture → session_id={session_id}")
            continue
        if not session_id:
            raise SystemExit("--session-id required for stages after capture")
        if stage == "tribe":
            _stage_tribe(session_id)
        elif stage == "dual_track":
            _stage_dual_track(session_id, args)
        elif stage == "heatmaps":
            _stage_heatmaps(session_id, args)
        elif stage == "analyze":
            _stage_analyze(session_id, args)
        elif stage == "narrative":
            _stage_narrative(session_id, args)
        elif stage == "export_viewer":
            _run([
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "export_ux_viewer.py"),
                "--session-id",
                session_id,
            ])

    if session_id:
        report = validate_session_dir(SESSIONS_DIR / session_id)
        if not report.get("skipped"):
            print(report.get("message", ""))
            manifest_path = SESSIONS_DIR / session_id / "session_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.setdefault("alignment", {})["preds_validation"] = report
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
