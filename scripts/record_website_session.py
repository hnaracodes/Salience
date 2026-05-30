#!/usr/bin/env python3
"""Orchestrated website session: walkthrough video + session_manifest.json (schema v2).

Requires: pip install playwright && playwright install chromium

Usage:
    python scripts/record_website_session.py --script configs/walkthrough_scripts/localhost_demo.yaml
    python scripts/record_website_session.py --session-id <id> --script path/to/script.yaml

Writes:
    scout_data/sessions/<id>/walkthrough.mp4|.webm
    scout_data/sessions/<id>/session_manifest.json
    scout_data/sessions/<id>/walkthrough_script.yaml (copy of script)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from activation_store import SESSIONS_DIR
from scout_core.walkthrough import load_walkthrough_script, record_website_session_async

DEFAULT_SCRIPTS_DIR = PROJECT_ROOT / "configs" / "walkthrough_scripts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", type=Path, required=True, help="Walkthrough YAML path")
    parser.add_argument("--session-id", default=None, help="Reuse or pre-assign session id")
    parser.add_argument("--interval-sec", type=float, default=None, help="Override TR interval (default from script or 1.0)")
    args = parser.parse_args()

    script_path = args.script if args.script.is_absolute() else PROJECT_ROOT / args.script
    if not script_path.is_file():
        raise SystemExit(f"Walkthrough script not found: {script_path}")

    script = load_walkthrough_script(script_path)
    interval_sec = args.interval_sec if args.interval_sec is not None else float(
        script.get("interval_sec", 1.0)
    )

    session_id = args.session_id or uuid.uuid4().hex
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    script_copy = session_dir / "walkthrough_script.yaml"
    shutil.copy2(script_path, script_copy)

    try:
        manifest = asyncio.run(
            record_website_session_async(
                session_id=session_id,
                session_dir=session_dir,
                script=script,
                interval_sec=interval_sec,
            ),
        )
    except ImportError as exc:
        raise SystemExit(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    manifest["walkthrough_script"] = "walkthrough_script.yaml"
    manifest["expected_tr_count"] = len(manifest["dom_snapshots"])

    out_path = session_dir / "session_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Session id: {session_id}")
    print(f"DOM snapshots: {len(manifest['dom_snapshots'])} (expected TR count)")
    print(f"Video: {manifest['video']['path']}")
    print(f"Manifest: {out_path}")


if __name__ == "__main__":
    main()
