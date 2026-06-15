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
    'header, nav, main, footer, section, article, aside, button, a, h1, h2, h3, h4, p, li, img, video, form, input, textarea, select, [role], [data-cta], [data-section]'
  ));
  const elements = [];
  const visibility = window.__tribeVisibility || {};
  for (const el of nodes) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const id = el.id ? '#' + el.id : '';
    const cls = (el.className && typeof el.className === 'string')
      ? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.') : '';
    const dom_id = id || (el.tagName.toLowerCase() + cls) || el.tagName.toLowerCase();
    const rawText = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
    const style = getComputedStyle(el);
    const vis = visibility[dom_id] || {};
    const visibleW = Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0));
    const visibleH = Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
    const viewportArea = Math.max(1, r.width * r.height);
    elements.push({
      dom_id: dom_id.slice(0, 120),
      tag: el.tagName,
      role: el.getAttribute('role') || '',
      aria_label: el.getAttribute('aria-label') || '',
      href: el.getAttribute('href') || '',
      text: rawText.slice(0, 200),
      bbox: [
        Math.round(r.left + scrollX),
        Math.round(r.top + scrollY),
        Math.round(r.width),
        Math.round(r.height),
      ],
      style: {
        color: style.color,
        background_color: style.backgroundColor,
        font_size: style.fontSize,
        font_weight: style.fontWeight,
        opacity: style.opacity,
        cursor: style.cursor,
      },
      is_image: el.tagName === 'IMG' || el.tagName === 'VIDEO' || Boolean(style.backgroundImage && style.backgroundImage !== 'none'),
      visibility_ratio: Math.max(0, Math.min(1, (visibleW * visibleH) / viewportArea)),
      visibility_ms: Math.round(vis.ms || 0),
      z_index: parseInt(style.zIndex, 10) || 0,
      is_intersecting_viewport: r.bottom > 0 && r.right > 0 && r.top < vh && r.left < vw,
    });
  }
  return { scrollY, scrollX, viewport_width: vw, viewport_height: vh, elements };
}
"""


VISIBILITY_TRACKER_JS = """
() => {
  window.__tribeVisibility = window.__tribeVisibility || {};
  const domId = (el) => {
    const id = el.id ? '#' + el.id : '';
    const cls = (el.className && typeof el.className === 'string')
      ? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.') : '';
    return (id || (el.tagName.toLowerCase() + cls) || el.tagName.toLowerCase()).slice(0, 120);
  };
  const tick = () => performance.now();
  if (!window.__tribeVisibilityObserver) {
    window.__tribeVisibilityObserver = new IntersectionObserver((entries) => {
      const now = tick();
      for (const entry of entries) {
        const key = domId(entry.target);
        const row = window.__tribeVisibility[key] || { ms: 0, visible: false, last: now, ratio: 0 };
        if (row.visible) row.ms += Math.max(0, now - row.last) * Math.max(row.ratio, 0);
        row.visible = entry.isIntersecting;
        row.last = now;
        row.ratio = entry.intersectionRatio || 0;
        window.__tribeVisibility[key] = row;
      }
    }, { threshold: [0, 0.1, 0.25, 0.5, 0.75, 1] });
  }
  document.querySelectorAll('header, nav, main, footer, section, article, aside, button, a, h1, h2, h3, h4, p, li, img, video, form, input, textarea, select, [role], [data-cta], [data-section]').forEach(el => {
    if (!el.__tribeObserved) {
      el.__tribeObserved = true;
      window.__tribeVisibilityObserver.observe(el);
    }
  });
  return true;
}
"""


RESOLVE_ELEMENT_JS = """
(selector) => {
  const el = document.querySelector(selector);
  if (!el) return null;
  const id = el.id ? '#' + el.id : '';
  const cls = (el.className && typeof el.className === 'string')
    ? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.') : '';
  const r = el.getBoundingClientRect();
  return {
    dom_id: (id || (el.tagName.toLowerCase() + cls) || el.tagName.toLowerCase()).slice(0, 120),
    tag: el.tagName,
    role: el.getAttribute('role') || '',
    text: (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 160),
    bbox: [Math.round(r.left + window.scrollX), Math.round(r.top + window.scrollY), Math.round(r.width), Math.round(r.height)],
  };
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


def interaction_t_idx(event_time: float, interval_sec: float) -> int:
    """Map wall-clock offset to TR index (matches scripted capture marker logic)."""
    return int(round(float(event_time) / max(float(interval_sec), 1e-6)))


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
    interaction_events: list[dict[str, Any]] | None = None,
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
        "interaction_events": interaction_events or [],
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


def plan_linear_scroll_y_targets(
    scroll_height: int,
    viewport_height: int,
    scroll_px_per_tr: int,
) -> list[int]:
    """Return monotonic scrollY targets: one per TR, from top to page bottom."""
    max_scroll = max(0, int(scroll_height) - int(viewport_height))
    step = scroll_px_per_tr if scroll_px_per_tr > 0 else max(1, int(viewport_height * 0.75))
    targets = [0]
    y = 0
    while y < max_scroll:
        y = min(y + step, max_scroll)
        if targets[-1] != y:
            targets.append(y)
    return targets


PAGE_METRICS_JS = """
() => ({
  scrollHeight: Math.max(
    document.body.scrollHeight,
    document.documentElement.scrollHeight,
    document.body.offsetHeight,
    document.documentElement.offsetHeight
  ),
  clientHeight: window.innerHeight,
})
"""


async def _capture_snapshot(
    page: Any,
    session_dir: Path,
    *,
    t_idx: int,
    event_time: float,
) -> dict[str, Any]:
    """Screenshot + DOM extract for one TR (aligned frame and manifest row)."""
    await page.evaluate(VISIBILITY_TRACKER_JS)
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frames_dir / f"t_{t_idx}.jpg"
    await page.screenshot(path=str(frame_path), type="jpeg", quality=92)
    payload = await page.evaluate(EXTRACT_DOM_JS)
    return {
        "t_idx": t_idx,
        "pts_sec": round(event_time, 3),
        "url": page.url,
        "scrollY": int(payload["scrollY"]),
        "scrollX": int(payload["scrollX"]),
        "elements": payload["elements"],
        "frame_path": f"frames/t_{t_idx}.jpg",
    }


async def _record_linear_scroll_async(
    *,
    session_id: str,
    session_dir: Path,
    script: dict[str, Any],
    interval_sec: float,
    width: int,
    height: int,
    initial_url: str,
    video_dir: Path,
) -> dict[str, Any]:
    """Fixed scroll increment per TR; TR count derived from page height."""
    from playwright.async_api import async_playwright

    scroll_px = int(script.get("scroll_px_per_tr", max(1, int(height * 0.75))))
    scroll_settle_ms = int(script.get("scroll_settle_ms", 500))
    initial_settle_ms = int(script.get("initial_settle_ms", 1500))

    snapshots: list[dict[str, Any]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(video_dir),
            record_video_size={"width": width, "height": height},
        )
        page = await context.new_page()
        await page.goto(initial_url, wait_until="networkidle", timeout=120_000)
        await asyncio.sleep(initial_settle_ms / 1000.0)

        metrics = await page.evaluate(PAGE_METRICS_JS)
        scroll_targets = plan_linear_scroll_y_targets(
            int(metrics["scrollHeight"]),
            int(metrics["clientHeight"]),
            scroll_px,
        )

        for t_idx, target_y in enumerate(scroll_targets):
            await page.evaluate(f"window.scrollTo(0, {int(target_y)})")
            await asyncio.sleep(scroll_settle_ms / 1000.0)
            snapshots.append(await _capture_snapshot(
                page, session_dir, t_idx=t_idx, event_time=t_idx * interval_sec,
            ))
            if t_idx + 1 < len(scroll_targets):
                await asyncio.sleep(interval_sec)

        await asyncio.sleep(interval_sec)

        await context.close()
        await browser.close()

    actual_duration = len(snapshots) * interval_sec
    manifest = build_manifest_v2(
        session_id=session_id,
        initial_url=initial_url,
        dom_snapshots=snapshots,
        width=width,
        height=height,
        interval_sec=interval_sec,
        duration_sec=actual_duration,
    )
    manifest["scroll_plan"] = {
        "mode": "linear",
        "scroll_px_per_tr": scroll_px,
        "scroll_targets": [s["scrollY"] for s in snapshots],
        "page_scroll_height": int(metrics["scrollHeight"]),
    }
    return manifest


async def _resolve_selector(page: Any, selector: str | None) -> dict[str, Any] | None:
    if not selector:
        return None
    return await page.evaluate(RESOLVE_ELEMENT_JS, selector)


async def _run_step(page: Any, step: dict[str, Any]) -> dict[str, Any] | None:
    """Execute a single walkthrough step.

    Note: ``wait_ms`` steps are NOT called through this function when using
    the merged timeline loop — their duration is already represented as a time
    gap between scheduled events.
    """
    action = step.get("action") or step.get("type")
    if action == "goto":
        url = step["url"]
        await page.goto(url, wait_until=step.get("wait_until", "networkidle"), timeout=120_000)
        await page.evaluate(VISIBILITY_TRACKER_JS)
    elif action == "wait_ms":
        await asyncio.sleep(step["ms"] / 1000.0)
    elif action == "scroll_to_y":
        y = int(step["y"])
        await page.evaluate(f"window.scrollTo(0, {y})")
    elif action == "click":
        target = await _resolve_selector(page, step.get("selector"))
        await page.click(step["selector"], timeout=step.get("timeout_ms", 30_000))
        return {"action": "click", "selector": step.get("selector"), "target": target}
    elif action == "hover":
        target = await _resolve_selector(page, step.get("selector"))
        await page.hover(step["selector"], timeout=step.get("timeout_ms", 30_000))
        return {"action": "hover", "selector": step.get("selector"), "target": target}
    elif action == "wait_for_selector":
        await page.wait_for_selector(
            step["selector"],
            timeout=step.get("timeout_ms", 30_000),
        )
    else:
        raise ValueError(f"Unknown walkthrough step action: {action!r}")
    return None


async def _finalize_video(session_dir: Path, video_dir: Path) -> str:
    video_rel = "walkthrough.mp4"
    candidates = sorted(video_dir.glob("*.webm")) + sorted(video_dir.glob("*.mp4"))
    if candidates:
        src = candidates[0]
        dest = session_dir / ("walkthrough.webm" if src.suffix == ".webm" else "walkthrough.mp4")
        shutil.move(str(src), str(dest))
        video_rel = dest.name
    if video_dir.exists():
        shutil.rmtree(video_dir, ignore_errors=True)
    return video_rel


async def _execute_explore_action(page: Any, action: Any) -> dict[str, Any] | None:
    """Run one explore_policy action; return interaction marker or None."""
    from scout_core.explore_policy import ExploreAction

    if not isinstance(action, ExploreAction):
        return None
    if action.kind == "scroll_down":
        scroll_px = int((action.metadata or {}).get("scroll_px", 540))
        await page.evaluate(f"window.scrollBy(0, {scroll_px})")
        return None
    if action.kind in ("click", "hover"):
        if action.locator_role and action.locator_name is not None:
            locator = page.get_by_role(action.locator_role, name=action.locator_name)
            if action.locator_href:
                # Extra uniqueness check: prefer the href-anchored link but fall
                # back to the role/name locator if the href form finds nothing.
                esc = action.locator_href.replace("'", "\\'")
                locator = page.locator(f"a[href='{esc}']").or_(locator)
            await locator.first.click(timeout=5000)
            return {
                "action": action.kind,
                "locator_role": action.locator_role,
                "locator_name": action.locator_name,
                "locator_href": action.locator_href,
            }
        if action.selector:
            step = {"action": action.kind, "selector": action.selector}
            return await _run_step(page, step)
    return None


async def _record_explore_async(
    *,
    session_id: str,
    session_dir: Path,
    script: dict[str, Any],
    interval_sec: float,
    width: int,
    height: int,
    initial_url: str,
    video_dir: Path,
) -> dict[str, Any]:
    """Bounded autonomous exploration: sample-then-act per TR clock."""
    from playwright.async_api import async_playwright

    from scout_core.explore_policy import ExploreBudget, load_explore_config, next_action

    cfg = load_explore_config(script)
    budget = ExploreBudget(
        max_tr=int(cfg.get("max_tr", 24)),
        max_clicks=int(cfg.get("max_clicks", 12)),
        max_pages=int(cfg.get("max_pages", 6)),
    )
    scroll_settle_ms = int(cfg.get("scroll_settle_ms", 500))
    initial_settle_ms = int(cfg.get("initial_settle_ms", 1500))

    snapshots: list[dict[str, Any]] = []
    interaction_events: list[dict[str, Any]] = []
    exploration_log: list[dict[str, Any]] = []
    pages_visited: list[str] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(video_dir),
            record_video_size={"width": width, "height": height},
        )
        page = await context.new_page()
        await page.goto(initial_url, wait_until="networkidle", timeout=120_000)
        await page.evaluate(VISIBILITY_TRACKER_JS)
        await asyncio.sleep(initial_settle_ms / 1000.0)
        budget.record_page(page.url)

        while not budget.at_limit():
            t_idx = budget.t_idx
            event_time = t_idx * interval_sec
            snapshots.append(await _capture_snapshot(
                page, session_dir, t_idx=t_idx, event_time=event_time,
            ))

            metrics = await page.evaluate(PAGE_METRICS_JS)
            action = next_action(
                snapshot=snapshots[-1],
                page_url=page.url,
                initial_url=initial_url,
                budget=budget,
                cfg=cfg,
                viewport_h=height,
                scroll_height=int(metrics.get("scrollHeight", 0)),
                client_height=int(metrics.get("clientHeight", height)),
            )

            exploration_log.append({
                "t_idx": t_idx,
                "action": action.kind,
                "selector": action.selector,
                "locator_role": action.locator_role,
                "locator_name": action.locator_name,
                "url": page.url,
                "reason": action.reason,
            })

            if action.kind == "stop":
                budget.t_idx += 1
                break

            _has_target = bool(action.selector or action.locator_role)
            if action.kind in ("click", "hover") and _has_target:
                marker = await _execute_explore_action(page, action)
                # Build a stable action key for deduplication regardless of locator style.
                action_key = (
                    f"{page.url}|role={action.locator_role}:{action.locator_name}"
                    if action.locator_role
                    else f"{page.url}|{action.selector}"
                )
                budget.record_click(action_key)
                if marker is not None:
                    marker.update({
                        "pts_sec": round(float(event_time), 3),
                        "t_idx": t_idx,
                    })
                    interaction_events.append(marker)
                await asyncio.sleep(scroll_settle_ms / 1000.0)
                budget.record_page(page.url)
                from urllib.parse import urlparse as _urlparse
                _path = _urlparse(page.url).path or "/"
                if _path not in pages_visited:
                    pages_visited.append(_path)
            elif action.kind == "scroll_down":
                await _execute_explore_action(page, action)
                await asyncio.sleep(scroll_settle_ms / 1000.0)

            budget.t_idx += 1
            if budget.t_idx < budget.max_tr:
                await asyncio.sleep(interval_sec)

        await context.close()
        await browser.close()

    actual_duration = len(snapshots) * interval_sec
    manifest = build_manifest_v2(
        session_id=session_id,
        initial_url=initial_url,
        dom_snapshots=snapshots,
        width=width,
        height=height,
        interval_sec=interval_sec,
        duration_sec=actual_duration,
        interaction_events=interaction_events,
    )
    manifest["capture_mode"] = "explore"
    manifest["pages_visited"] = pages_visited or budget.pages_visited
    manifest["exploration_log"] = exploration_log
    return manifest


async def record_website_session_async(
    *,
    session_id: str,
    session_dir: Path,
    script: dict[str, Any],
    interval_sec: float = 1.0,
) -> dict[str, Any]:
    """Playwright capture: video + aligned DOM snapshots (+ JPEG frames per TR).

    ``scroll_mode: linear`` — fixed scroll increment per TR; TR count is derived
    from measured page height (recommended for heatmap / viewer alignment).

    ``scroll_mode: explore`` — bounded autonomous nav/CTA discovery per TR clock.

    ``scroll_mode: scripted`` (default) — YAML ``steps`` interleaved with TR
    samples on a fixed ``duration_sec`` timeline.
    """
    viewport = script.get("viewport") or {}
    width = int(viewport.get("width", 1920))
    height = int(viewport.get("height", 1080))
    initial_url = script["initial_url"]
    scroll_mode = (script.get("scroll_mode") or "scripted").lower()

    video_dir = session_dir / "_playwright_video"
    video_dir.mkdir(parents=True, exist_ok=True)

    if scroll_mode == "explore":
        manifest = await _record_explore_async(
            session_id=session_id,
            session_dir=session_dir,
            script=script,
            interval_sec=interval_sec,
            width=width,
            height=height,
            initial_url=initial_url,
            video_dir=video_dir,
        )
        manifest["video"]["path"] = await _finalize_video(session_dir, video_dir)
        from scout_core.session_align import pad_manifest_snapshots_to_video_duration

        preds_path = session_dir / "preds.npz"
        target_tr = None
        if preds_path.is_file():
            try:
                import numpy as _np

                target_tr = int(_np.load(preds_path)["preds"].shape[0])
            except Exception:
                target_tr = None
        manifest = pad_manifest_snapshots_to_video_duration(
            manifest, session_dir, target_tr_count=target_tr,
        )
        return manifest

    if scroll_mode == "linear":
        manifest = await _record_linear_scroll_async(
            session_id=session_id,
            session_dir=session_dir,
            script=script,
            interval_sec=interval_sec,
            width=width,
            height=height,
            initial_url=initial_url,
            video_dir=video_dir,
        )
        manifest["video"]["path"] = await _finalize_video(session_dir, video_dir)
        from scout_core.session_align import pad_manifest_snapshots_to_video_duration

        preds_path = session_dir / "preds.npz"
        target_tr = None
        if preds_path.is_file():
            try:
                import numpy as _np

                target_tr = int(_np.load(preds_path)["preds"].shape[0])
            except Exception:
                target_tr = None
        manifest = pad_manifest_snapshots_to_video_duration(
            manifest, session_dir, target_tr_count=target_tr,
        )
        return manifest

    from playwright.async_api import async_playwright

    steps: list[dict[str, Any]] = list(script.get("steps") or [])
    initial_settle_ms = int(script.get("initial_settle_ms", 800))

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

    n_timesteps = expected_tr_count(float(duration_sec), interval_sec)
    step_schedule = build_step_schedule(steps)

    all_events: list[tuple[float, int, Any]] = []
    for t_idx in range(n_timesteps):
        all_events.append((t_idx * interval_sec, 0, t_idx))
    for t_offset, step in step_schedule:
        all_events.append((t_offset, 1, step))
    all_events.sort(key=lambda x: (x[0], x[1]))

    snapshots: list[dict[str, Any]] = []
    interaction_events: list[dict[str, Any]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(video_dir),
            record_video_size={"width": width, "height": height},
        )
        page = await context.new_page()
        await page.goto(initial_url, wait_until="networkidle", timeout=120_000)
        await asyncio.sleep(initial_settle_ms / 1000.0)

        current_time = 0.0
        for event_time, kind, data in all_events:
            gap = event_time - current_time
            if gap > 0.001:
                await asyncio.sleep(gap)
                current_time = event_time

            if kind == 0:
                t_idx = data
                snapshots.append(await _capture_snapshot(
                    page, session_dir, t_idx=t_idx, event_time=event_time,
                ))
            else:
                step = data
                action = step.get("action") or step.get("type")
                if action != "wait_ms":
                    marker = await _run_step(page, step)
                    if marker is not None:
                        marker.update({
                            "pts_sec": round(float(event_time), 3),
                            "t_idx": interaction_t_idx(event_time, interval_sec),
                        })
                        interaction_events.append(marker)

        await context.close()
        await browser.close()

    video_rel = await _finalize_video(session_dir, video_dir)
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
        interaction_events=interaction_events,
    )
