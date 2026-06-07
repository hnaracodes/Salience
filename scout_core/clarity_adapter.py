"""Parse Microsoft Clarity heatmap CSV exports into element click signals.

Clarity's public API exposes dashboard aggregates, not raw heatmap coordinates.
For owned sites, a user can manually export a click/area heatmap CSV from the
Clarity UI and feed it through this adapter.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _pick(row: dict[str, str], *names: str) -> str:
    lowered = {k.strip().lower(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return ""


def parse_clarity_click_csv(path: Path) -> list[dict[str, Any]]:
    """Return normalized click rows from a Clarity CSV export.

    The CSV schema has varied across Clarity heatmap types. This parser accepts
    common column names such as "Element", "Selector", "Clicks", "Click count",
    "URL", and "Area".
    """
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            selector = _pick(raw, "Selector", "Element", "Element selector", "CSS selector", "Area")
            text = _pick(raw, "Text", "Element text", "Label")
            url = _pick(raw, "URL", "Page URL", "Visited URL")
            clicks_raw = _pick(raw, "Clicks", "Click count", "Total clicks", "Count")
            sessions_raw = _pick(raw, "Sessions", "Session count", "Views", "Page views")
            try:
                clicks = int(float(clicks_raw.replace(",", ""))) if clicks_raw else 0
            except ValueError:
                clicks = 0
            try:
                sessions = int(float(sessions_raw.replace(",", ""))) if sessions_raw else None
            except ValueError:
                sessions = None
            click_rate = (clicks / sessions) if sessions and sessions > 0 else None
            if not selector and not text and clicks <= 0:
                continue
            rows.append({
                "dom_id": selector,
                "selector": selector,
                "text": text,
                "url": url,
                "clicks": clicks,
                "sessions": sessions,
                "click_rate": round(click_rate, 6) if click_rate is not None else None,
                "source": "microsoft_clarity_csv",
            })
    rows.sort(key=lambda r: (r.get("clicks") or 0, r.get("click_rate") or 0), reverse=True)
    return rows


def click_rows_by_selector(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index normalized Clarity rows by selector/dom_id."""
    return {
        str(row.get("selector") or row.get("dom_id")): row
        for row in rows
        if row.get("selector") or row.get("dom_id")
    }
