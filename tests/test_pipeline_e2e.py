"""End-to-end backend pipeline smoke test (FAKE_TRIBE, local fixture site).

Runs capture → tribe → dual_track → heatmaps → analyze → copy_signals →
narrative → export_viewer against tests/fixtures/walkthrough_site on a
local HTTP server. Requires Playwright Chromium (worker container or dev env).

Mark: pytest -m integration tests/test_pipeline_e2e.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "walkthrough_site"
LOCALHOST_DEMO = PROJECT_ROOT / "configs" / "walkthrough_scripts" / "localhost_demo.yaml"
DEFAULT_NORM_ID = "synthetic_bootstrap_v1"


@pytest.fixture(scope="module")
def ensure_norm_bundle():
    """Ensure bootstrap norms exist with portable repo-relative SQLite paths."""
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "bootstrap_norms.py"),
            "--norm-id",
            DEFAULT_NORM_ID,
        ],
        check=True,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def fixture_http_server():
    port = _free_port()
    fixture_dir = str(FIXTURE_DIR)

    class FixtureHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=fixture_dir, **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", port), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield port
    server.shutdown()


@pytest.mark.integration
def test_full_pipeline_fake_tribe_localhost_fixture(
    fixture_http_server, ensure_norm_bundle, monkeypatch
):
    """Deployment smoke: all pipeline stages complete and emit viewer artifacts."""
    if shutil_which("playwright") is None and not _playwright_importable():
        pytest.skip("Playwright not available in this environment")

    monkeypatch.setenv("FAKE_TRIBE", "1")
    monkeypatch.setenv("SSRF_ALLOW_LOCALHOST", "1")
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)  # local file:// viewer URL

    session_id = uuid.uuid4().hex
    scan_id = session_id
    url = f"http://127.0.0.1:{fixture_http_server}/index.html"
    config = {"explore_script": str(LOCALHOST_DEMO)}

    stages: list[tuple[str, str]] = []

    def on_stage(stage: str, status: str) -> None:
        stages.append((stage, status))

    from services.pipeline.runner import run_scan

    result = run_scan(
        session_id,
        url,
        site_goal="Drive sign-ups on the landing page.",
        config=config,
        on_stage=on_stage,
        scan_id=scan_id,
    )

    assert result["viewer_url"]
    from activation_store import SESSIONS_DIR

    session_dir = SESSIONS_DIR / session_id
    assert (session_dir / "session_manifest.json").is_file()
    assert (session_dir / "preds.npz").is_file()
    assert (session_dir / "ux_viewer" / "index.html").is_file()

    completed = [s for s, st in stages if st == "done"]
    assert completed == [
        "capture",
        "tribe",
        "dual_track",
        "heatmaps",
        "analyze",
        "copy_signals",
        "narrative",
        "export_viewer",
    ]


def shutil_which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def _playwright_importable() -> bool:
    try:
        import playwright  # noqa: F401

        return True
    except ImportError:
        return False
