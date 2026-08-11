#!/usr/bin/env python3
"""Capture a few public landing-page screenshots for DeepGaze MSDB evaluation.

Uses Playwright only. Does not touch TRIBE, dual-track, or Horikawa code paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TARGETS = [
    {"id": "stripe", "url": "https://stripe.com/"},
    {"id": "mozilla_firefox", "url": "https://www.mozilla.org/en-US/firefox/"},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture eval screenshots for DeepGaze MSDB")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "deepgaze_msdb_eval" / "screenshots",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--wait-ms", type=int, default=2500)
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for capture. Use the website venv "
            "(venv311) which already has it installed."
        ) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=1,
        )
        page = context.new_page()
        for target in DEFAULT_TARGETS:
            out_path = args.out_dir / f"{target['id']}.png"
            try:
                page.goto(target["url"], wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(args.wait_ms)
                page.screenshot(path=str(out_path), type="png")
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - capture continues
                status = f"error: {exc}"
                out_path = None
            rows.append(
                {
                    "id": target["id"],
                    "url": target["url"],
                    "path": str(out_path.resolve()) if out_path else None,
                    "status": status,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"{target['id']}: {status}")
        browser.close()

    manifest = args.out_dir / "screenshots_manifest.json"
    manifest.write_text(json.dumps({"targets": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
