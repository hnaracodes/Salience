#!/usr/bin/env python3
"""Record session_manifest.json with DOM snapshots aligned to TRIBE timesteps.

Requires: pip install playwright && playwright install chromium

Usage:
    python scripts/record_session_manifest.py --session-id <id> --url https://example.com
    python scripts/record_session_manifest.py --session-id <id> --url https://site.com --duration-sec 21

Writes scout_data/sessions/<id>/session_manifest.json with one dom_snapshot per TR.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from activation_store import SESSIONS_DIR

_EXTRACT_JS = """
() => {
  const scrollY = window.scrollY;
  const scrollX = window.scrollX;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const nodes = Array.from(document.querySelectorAll(
    'header, nav, main, footer, section, article, aside, button, a, h1, h2, h3, [role]'
  ));
  const elements = [];
  for (const el of nodes) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const id = el.id ? '#' + el.id : '';
    const cls = (el.className && typeof el.className === 'string')
      ? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.') : '';
    const dom_id = id || (el.tagName.toLowerCase() + cls) || el.tagName.toLowerCase();
    elements.push({
      dom_id: dom_id.slice(0, 120),
      tag: el.tagName,
      role: el.getAttribute('role') || '',
      bbox: [
        Math.round(r.left + scrollX),
        Math.round(r.top + scrollY),
        Math.round(r.width),
        Math.round(r.height),
      ],
      z_index: parseInt(getComputedStyle(el).zIndex, 10) || 0,
      is_intersecting_viewport: r.bottom > 0 && r.right > 0 && r.top < vh && r.left < vw,
    });
  }
  return { scrollY, scrollX, viewport_width: vw, viewport_height: vh, elements };
}
"""


async def _record_async(
    url: str,
    session_dir: Path,
    n_timesteps: int,
    interval_sec: float,
    width: int,
    height: int,
) -> dict:
    from playwright.async_api import async_playwright

    snapshots: list[dict] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(url, wait_until="networkidle", timeout=120_000)

        for t in range(n_timesteps):
            if t > 0:
                await asyncio.sleep(interval_sec)
            payload = await page.evaluate(_EXTRACT_JS)
            snapshots.append({
                "t_idx": t,
                "pts_sec": round(t * interval_sec, 3),
                "url": page.url,
                "scrollY": int(payload["scrollY"]),
                "scrollX": int(payload["scrollX"]),
                "elements": payload["elements"],
            })

        await browser.close()

    return {
        "schema_version": 1,
        "initial_url": url,
        "capture": {"width": width, "height": height, "fps": 1.0 / interval_sec if interval_sec else 1.0},
        "dom_snapshots": snapshots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--duration-sec", type=float, default=None, help="Override TR count from duration / interval")
    parser.add_argument("--interval-sec", type=float, default=1.0, help="Seconds between snapshots (TRIBE TR rate)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    session_dir = SESSIONS_DIR / args.session_id
    if not session_dir.is_dir():
        raise SystemExit(f"Session directory not found: {session_dir}")

    preds_path = session_dir / "preds.npz"
    if args.duration_sec is not None:
        n_t = max(1, int(round(args.duration_sec / args.interval_sec)))
    elif preds_path.is_file():
        n_t = int(np.load(preds_path)["preds"].shape[0])
    else:
        raise SystemExit("Provide --duration-sec or ensure preds.npz exists in session dir.")

    try:
        manifest = asyncio.run(
            _record_async(args.url, session_dir, n_t, args.interval_sec, args.width, args.height),
        )
    except ImportError as exc:
        raise SystemExit(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    out_path = session_dir / "session_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest['dom_snapshots'])} DOM snapshots → {out_path}")


if __name__ == "__main__":
    main()
