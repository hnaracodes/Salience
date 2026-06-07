#!/usr/bin/env python3
"""Regenerate or check sessions/INDEX.md from Ledger Protocol manifests."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = PROJECT_ROOT / "sessions"
INDEX_PATH = SESSIONS_DIR / "INDEX.md"


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("'\"") for part in inner.split(",")]
    return value.strip("'\"")


def read_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Missing frontmatter: {path}")
    data: dict[str, Any] = {}
    key_for_list: str | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("  - ") and key_for_list:
            data.setdefault(key_for_list, []).append(line[4:].strip())
            continue
        key_for_list = None
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip()
        if raw == "":
            data[key] = []
            key_for_list = key
        else:
            data[key] = _parse_scalar(raw)
    return data


def load_sessions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_md in sorted(SESSIONS_DIR.glob("CS-*/session.md")):
        fm = read_frontmatter(session_md)
        fm["folder"] = session_md.parent.name
        fm["session_path"] = session_md
        rows.append(fm)
    rows.sort(key=lambda row: (str(row.get("date", "")), str(row.get("session_id", ""))))
    return rows


def _topic_for(row: dict[str, Any]) -> str:
    path = row.get("session_path")
    if isinstance(path, Path) and path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            if line.strip() == "## Topics":
                for candidate in lines[idx + 1:]:
                    text = candidate.strip()
                    if text and not text.startswith("#"):
                        return text.replace("|", "\\|")
                    if text.startswith("## "):
                        break
    title = str(row.get("title") or row.get("session_id") or "")
    return title.replace("|", "\\|")


def _row(row: dict[str, Any]) -> str:
    sid = str(row.get("session_id") or row["folder"])
    date = str(row.get("date") or "")
    status = str(row.get("status") or "")
    track = str(row.get("track") or "")
    signature = str(row.get("signature") or "")
    folder = str(row["folder"])
    title = _topic_for(row)
    return (
        f"| `{sid}` | {date} | {status} | {track} | `{signature}` | "
        f"[sessions/{folder}](./{folder}/session.md) | {title} |"
    )


def render_index(rows: list[dict[str, Any]]) -> str:
    main = [r for r in rows if r.get("track") != "neuroemo" or r.get("status") != "archived"]
    archived = [r for r in rows if r.get("track") == "neuroemo" and r.get("status") == "archived"]
    header = """# Ledger Session Index

This is the unified registry for Ledger Protocol sessions. Future agents should start here, then read the relevant `session.md`, `ledger.md`, category files, and any migrated `report.md`.

## Main Sessions

| Session ID | Date | Status | Track | Signature | Folder | Topics |
|------------|------|--------|-------|-----------|--------|--------|
"""
    archive_header = """

## Archived NeuroEmo Sessions

| Session ID | Date | Status | Track | Signature | Folder | Topics |
|------------|------|--------|-------|-----------|--------|--------|
"""
    footer = """

## Query Hints

- Find open work: search `sessions/**/issues/*.md` and `sessions/**/bugs/*.md` for `status: open` or `status: in_progress`.
- Continue a feature: read `sessions/<SESSION_ID>/session.md`, then `ledger.md`, then category files.
- Recover old long-form context: read `report.md` inside migrated session folders.
"""
    return (
        header
        + "\n".join(_row(r) for r in main)
        + archive_header
        + "\n".join(_row(r) for r in archived)
        + footer
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit non-zero if INDEX.md is stale")
    args = parser.parse_args()
    rendered = render_index(load_sessions())
    if args.check:
        current = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.is_file() else ""
        if current != rendered:
            diff = difflib.unified_diff(
                current.splitlines(),
                rendered.splitlines(),
                fromfile=str(INDEX_PATH),
                tofile="generated",
                lineterm="",
            )
            print("\n".join(diff), file=sys.stderr)
            return 1
        print("sessions/INDEX.md is up to date")
        return 0
    INDEX_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
