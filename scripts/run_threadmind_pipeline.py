#!/usr/bin/env python3
"""End-to-end Threadmind pipeline: Playwright capture → TRIBEv2 → triple-track → inspect → analyze → heatmaps → UX viewer.

Prerequisites:
  - .venv activated (or use .venv\\Scripts\\python.exe)
  - Modal auth for tribe + DINOv2 heatmaps
  - Optional: GEMINI_API_KEY for narrative (falls back to template)

Usage:
    # Full run (starts local site server on :8780 if needed):
    python scripts/run_threadmind_pipeline.py

    # Include gray baseline generation before dual-track:
    python scripts/run_threadmind_pipeline.py --create-baseline

    # Autonomous explore capture instead of linear scroll:
    python scripts/run_threadmind_pipeline.py --explore

    # Site already served elsewhere:
    python scripts/run_threadmind_pipeline.py --no-serve

    # Resume after capture:
    python scripts/run_threadmind_pipeline.py --no-serve --session-id <id> --from-stage tribe
"""

from __future__ import annotations

import argparse
import atexit
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = PROJECT_ROOT / "chatbot_product_site"
DEFAULT_SCRIPT = PROJECT_ROOT / "configs" / "walkthrough_scripts" / "threadmind_showcase.yaml"
EXPLORE_SCRIPT = PROJECT_ROOT / "configs" / "walkthrough_scripts" / "explore_threadmind.yaml"
DEFAULT_PORT = 8780
DEFAULT_NORM = "synthetic_bootstrap_v1"

_server_proc: subprocess.Popen | None = None


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def _start_site_server(port: int) -> None:
    global _server_proc
    if _port_open(port):
        print(f"Site already reachable on http://127.0.0.1:{port}/")
        return
    if not SITE_DIR.is_dir():
        raise SystemExit(f"Threadmind site not found: {SITE_DIR}")
    _server_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--directory", str(SITE_DIR)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(_stop_server)
    for _ in range(30):
        if _port_open(port):
            print(f"Serving Threadmind at http://127.0.0.1:{port}/")
            return
        time.sleep(0.2)
    raise SystemExit(f"Timed out waiting for http.server on port {port}")


def _stop_server() -> None:
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    _server_proc = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--explore", action="store_true", help="Use explore_threadmind.yaml instead of linear scroll")
    parser.add_argument("--create-baseline", action="store_true", help="Run modal tribe.py::record_baseline before dual-track")
    parser.add_argument("--no-serve", action="store_true", help="Do not start chatbot_product_site http.server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--norm-id", default=DEFAULT_NORM)
    parser.add_argument(
        "--from-stage",
        default="all",
        choices=("all", "capture", "tribe", "dual_track", "heatmaps", "analyze", "narrative", "export_viewer"),
        help="Pipeline stage (default: all)",
    )
    parser.add_argument("--uniform-heatmap", action="store_true", help="CPU placeholder heatmaps (no Modal)")
    parser.add_argument("--skip-narrative", action="store_true", help="Skip Gemini/template narrative stage")
    parser.add_argument("--provider", default=None, help="Narrative provider: gemini | template")
    args = parser.parse_args()

    if not args.no_serve and args.from_stage in ("all", "capture"):
        _start_site_server(args.port)

    script = EXPLORE_SCRIPT if args.explore else DEFAULT_SCRIPT
    if not script.is_file():
        raise SystemExit(f"Walkthrough script missing: {script}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_website_session.py"),
        "--stage",
        args.from_stage,
        "--script",
        str(script),
        "--norm-id",
        args.norm_id,
        "--website",
        "--ground",
        "--with-heatmaps",
        "--refresh-sections",
        "--inspect",
    ]
    if args.session_id:
        cmd.extend(["--session-id", args.session_id])
    if args.create_baseline:
        cmd.append("--create-baseline")
    if args.uniform_heatmap:
        cmd.append("--uniform-heatmap")
    else:
        pass  # run_website_session uses --modal for heatmaps by default
    if args.provider:
        cmd.extend(["--llm-provider", args.provider])
    if args.skip_narrative:
        cmd.append("--skip-narrative")

    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
