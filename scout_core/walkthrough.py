"""Scripted website walkthrough: DOM extraction, step execution, manifest v2.

DOM snapshots are interleaved with scripted walkthrough actions so that
``dom_snapshots[t].pts_sec`` describes the actual page state at that moment
in the recorded video, not the state after all steps completed.
"""

from __future__ import annotations

import asyncio
import math
import shutil
from pathlib import Path
from typing import Any

import yaml

EXTRACT_DOM_JS = """
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


def load_walkthrough_script(path: Path) -> dict[str, Any]:
    """Load YAML walkthrough: initial_url, viewport, steps[], optional duration_sec."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not data.get("initial_url"):
        raise ValueError(f"walkthrough script missing initial_url: {path}")
    return data


def expected_tr_count(duration_sec: float, interval_sec: float) -> int:
    return max(1, int(math.ceil(duration_sec / interval_sec)))


def build_manifest_v2(
    *,
    session_id: str,
    initial_url: str,
    dom_snapshots: list[dict[str, Any]],
    width: int,
    height: int,
    interval_sec: float,
    video_path: str = "walkthrough.mp4",
    duration_sec: float | None = None,
    walkthrough_script: str | None = "walkthrough_script.yaml",
) -> dict[str, Any]:
    """Build session_manifest.json schema v2 payload."""
    tr_duration = float(interval_sec)
    dur = duration_sec if duration_sec is not None else len(dom_snapshots) * tr_duration
    return {
        "schema_version": 2,
        "session_id": session_id,
        "initial_url": initial_url,
        "video": {
            "path": video_path,
            "duration_sec": round(dur, 3),
        },
        "capture": {
            "width": width,
            "height": height,
            "fps": 1.0 / tr_duration if tr_duration else 1.0,
            "tr_duration_sec": tr_duration,
        },
        "tr_mapping": {
            "type": "linear",
            "tr_duration_sec": tr_duration,
        },
        "walkthrough_script": walkthrough_script,
        "dom_snapshots": dom_snapshots,
    }


def build_step_schedule(
    steps: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any]]]:
    """Return [(offset_sec, step)] assigning each step a cumulative time offset.

    Only ``wait_ms`` steps advance the clock; all other actions are
    considered instantaneous for scheduling purposes.
    """
    schedule: list[tuple[float, dict[str, Any]]] = []
    t_cursor = 0.0
    for step in steps:
        schedule.append((t_cursor, step))
        action = step.get("action") or step.get("type")
        if action == "wait_ms":
            t_cursor += step.get("ms", 0) / 1000.0
    return schedule


async def _run_step(page: Any, step: dict[str, Any]) -> None:
    """Execute a single walkthrough step.

    Note: ``wait_ms`` steps are NOT called through this function when using
    the merged timeline loop — their duration is already represented as a time
    gap between scheduled events.
    """
    action = step.get("action") or step.get("type")
    if action == "goto":
        url = step["url"]
        await page.goto(url, wait_until=step.get("wait_until", "networkidle"), timeout=120_000)
    elif action == "wait_ms":
        await asyncio.sleep(step["ms"] / 1000.0)
    elif action == "scroll_to_y":
        y = int(step["y"])
        await page.evaluate(f"window.scrollTo(0, {y})")
    elif action == "click":
        await page.click(step["selector"], timeout=step.get("timeout_ms", 30_000))
    elif action == "wait_for_selector":
        await page.wait_for_selector(
            step["selector"],
            timeout=step.get("timeout_ms", 30_000),
        )
    else:
        raise ValueError(f"Unknown walkthrough step action: {action!r}")


async def record_website_session_async(
    *,
    session_id: str,
    session_dir: Path,
    script: dict[str, Any],
    interval_sec: float = 1.0,
) -> dict[str, Any]:
    """Playwright capture: video + DOM snapshots interleaved with walkthrough steps.

    DOM snapshots are taken during the walkthrough (not after all steps complete)
    so that ``dom_snapshots[t]`` reflects the actual page state at that video
    timestamp. The merged timeline fires sample events (every ``interval_sec``)
    and step events at their scheduled wall-clock offsets, in chronological
    order. At the same timestamp, samples run before steps so each snapshot
    captures the page state immediately *before* the action that fires at that
    moment.

    ``wait_ms`` steps are represented purely as gaps between scheduled events —
    they are NOT re-executed via ``_run_step`` to avoid double-sleeping.
    """
    from playwright.async_api import async_playwright

    viewport = script.get("viewport") or {}
    width = int(viewport.get("width", 1920))
    height = int(viewport.get("height", 1080))
    initial_url = script["initial_url"]
    steps: list[dict[str, Any]] = list(script.get("steps") or [])

    # Compute total capture duration.
    duration_sec = script.get("duration_sec")
    if duration_sec is None:
        step_total_sec = sum(
            s.get("ms", 0) / 1000.0
            for s in steps
            if (s.get("action") or s.get("type")) == "wait_ms"
        )
        duration_sec = step_total_sec + float(script.get("post_steps_duration_sec", 0))
        if duration_sec <= 0:
            duration_sec = 10.0

    n_timesteps = expected_tr_count(duration_sec, interval_sec)

    # Build step schedule: cumulative offset per step (only wait_ms advances clock).
    step_schedule = build_step_schedule(steps)

    # Merged event list: (time_sec, priority, data)
    # priority=0 for samples (comes first at same time), 1 for steps.
    all_events: list[tuple[float, int, Any]] = []
    for t_idx in range(n_timesteps):
        all_events.append((t_idx * interval_sec, 0, t_idx))  # sample event
    for t_offset, step in step_schedule:
        all_events.append((t_offset, 1, step))  # step event
    all_events.sort(key=lambda x: (x[0], x[1]))

    video_dir = session_dir / "_playwright_video"
    video_dir.mkdir(parents=True, exist_ok=True)

    snapshots: list[dict[str, Any]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(video_dir),
            record_video_size={"width": width, "height": height},
        )
        page = await context.new_page()

        # Initial navigation is outside the timed capture loop so the clock
        # starts after the page is fully loaded.
        await page.goto(initial_url, wait_until="networkidle", timeout=120_000)

        current_time = 0.0
        for event_time, kind, data in all_events:
            gap = event_time - current_time
            if gap > 0.001:  # > 1 ms threshold to avoid redundant micro-sleeps
                await asyncio.sleep(gap)
                current_time = event_time

            if kind == 0:
                # DOM sample event
                t_idx = data
                payload = await page.evaluate(EXTRACT_DOM_JS)
                snapshots.append({
                    "t_idx": t_idx,
                    "pts_sec": round(event_time, 3),
                    "url": page.url,
                    "scrollY": int(payload["scrollY"]),
                    "scrollX": int(payload["scrollX"]),
                    "elements": payload["elements"],
                })
            else:
                # Step event
                step = data
                action = step.get("action") or step.get("type")
                if action == "wait_ms":
                    # Duration already encoded as gap to next event; skip execution.
                    pass
                else:
                    await _run_step(page, step)

        await context.close()
        await browser.close()

    video_rel = "walkthrough.mp4"
    candidates = sorted(video_dir.glob("*.webm")) + sorted(video_dir.glob("*.mp4"))
    if candidates:
        src = candidates[0]
        dest = session_dir / ("walkthrough.webm" if src.suffix == ".webm" else "walkthrough.mp4")
        shutil.move(str(src), str(dest))
        video_rel = dest.name
    if video_dir.exists():
        shutil.rmtree(video_dir, ignore_errors=True)

    actual_duration = max((s["pts_sec"] for s in snapshots), default=0.0) + interval_sec
    return build_manifest_v2(
        session_id=session_id,
        initial_url=initial_url,
        dom_snapshots=snapshots,
        width=width,
        height=height,
        interval_sec=interval_sec,
        duration_sec=actual_duration,
        video_path=video_rel,
    )
