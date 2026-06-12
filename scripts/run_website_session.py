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

Stages: capture | tribe | dual_track | heatmaps | analyze | narrative | export_viewer | all

Optional: --create-baseline (Modal gray-video baseline before dual_track)
           --inspect (print preds sample after pipeline)
"""

from __future__ import annotations

import argparse
import json
import shutil
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


def _modal_run(target: str, extra: list[str] | None = None) -> None:
    extra = extra or []
    if shutil.which("modal"):
        _run(["modal", "run", target, *extra])
        return
    _run([sys.executable, "-m", "modal", "run", target, *extra])


def _stage_baseline() -> None:
    _modal_run("tribe.py::record_baseline")


def _stage_inspect(session_id: str) -> None:
    _run([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "inspect_session.py"),
        "--session-id",
        session_id,
    ])


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
    _modal_run("tribe.py::record_session", ["--session-id", session_id])


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
    if args.goal:
        cmd.extend(["--goal", args.goal])
    if args.script:
        cmd.extend(["--script", str(args.script)])
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
    parser.add_argument("--goal", default=None, help="Site goal paragraph for narrative stage")
    parser.add_argument(
        "--create-baseline",
        action="store_true",
        help="Before dual_track: modal run tribe.py::record_baseline (gray video → preds_baseline.npz)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="After pipeline: print sample preds / SQLite row via inspect_session.py",
    )
    parser.add_argument("--no-marketing-scores", action="store_true", dest="no_marketing_scores")
    parser.add_argument("--skip-narrative", action="store_true", help="Omit narrative stage when --stage all")
    args = parser.parse_args()

    session_id = args.session_id
    all_stages = ["capture", "tribe", "dual_track", "heatmaps", "analyze", "narrative", "export_viewer"]
    if args.skip_narrative:
        all_stages = [s for s in all_stages if s != "narrative"]
    stages = all_stages if args.stage == "all" else [args.stage]

    if args.create_baseline and "dual_track" in stages:
        print("create-baseline → tribe.py::record_baseline")
        _stage_baseline()

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
        if args.inspect:
            _stage_inspect(session_id)
        viewer_dir = SESSIONS_DIR / session_id / "ux_viewer"
        if viewer_dir.is_dir():
            print(f"\nUX viewer: http://127.0.0.1:8787/index.html?base=.  (serve: python -m http.server 8787 --directory {viewer_dir})")
            print(f"session_id={session_id}")


if __name__ == "__main__":
    main()
