#!/usr/bin/env python3
"""Run the preliminary public-site capture/TRIBE corpus.

Public URLs in this corpus are engineering and norm-corpus inputs only. They do
not provide behavioral ground truth for attention T2.

Examples:
    python scripts/run_validation_site_corpus.py --stage capture --repeats 1
    python scripts/run_validation_site_corpus.py --stage all --site stripe
    python scripts/run_validation_site_corpus.py --stage all --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "configs" / "validation_site_corpus.yaml"
RUN_LOG = ROOT / "scout_data" / "validation" / "preliminary_web_corpus" / "runs.jsonl"


def load_corpus(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        corpus = yaml.safe_load(handle) or {}
    sites = list(corpus.get("sites") or [])
    local_sites = list(corpus.get("local_sites") or [])
    if not sites and not local_sites:
        raise ValueError(f"No sites configured in {path}")
    seen: set[str] = set()
    for site in sites + local_sites:
        site_id = str(site.get("id") or "")
        url = str(site.get("url") or "")
        if not site_id or not url:
            raise ValueError("Every corpus site requires id and url")
        if site_id in seen:
            raise ValueError(f"Duplicate site id: {site_id}")
        seen.add(site_id)
    return corpus


def build_command(
    *,
    site: dict[str, Any],
    corpus: dict[str, Any],
    session_id: str,
    stage: str,
) -> list[str]:
    # Local fixtures must bake initial_url into the script: --url is SSRF-blocked
    # for loopback addresses by design.
    script_rel = site.get("script") or corpus["base_profile"]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_website_session.py"),
        "--stage",
        stage,
        "--script",
        str(ROOT / str(script_rel)),
        "--session-id",
        session_id,
    ]
    if not site.get("script"):
        cmd.extend(["--url", str(site["url"])])
    if stage == "all":
        cmd.extend(
            [
                "--norm-id",
                str(corpus.get("norm_id") or "synthetic_bootstrap_v1"),
                "--website",
                "--with-heatmaps",
                "--skip-narrative",
            ]
        )
    return cmd


def inspect_capture(session_id: str, gates: dict[str, Any]) -> dict[str, Any]:
    manifest_path = ROOT / "scout_data" / "sessions" / session_id / "session_manifest.json"
    if not manifest_path.is_file():
        return {"promoted": False, "reason": "missing_manifest"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_count = len(manifest.get("dom_snapshots") or [])
    capture_mode = manifest.get("capture_mode")
    min_snapshots = int(gates.get("min_dom_snapshots") or 1)
    required_mode = gates.get("require_capture_mode")
    promoted = snapshot_count >= min_snapshots and (
        not required_mode or capture_mode == required_mode
    )
    return {
        "promoted": promoted,
        "snapshot_count": snapshot_count,
        "capture_mode": capture_mode,
        "reason": None if promoted else "capture_gate_failed",
    }


def append_run_log(row: dict[str, Any]) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--stage", choices=("capture", "all"), default="capture")
    parser.add_argument("--site", action="append", default=[], help="Site id; repeat to select several")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run only local_sites fixtures (requires http.server on configured ports)",
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    corpus_path = args.corpus if args.corpus.is_absolute() else ROOT / args.corpus
    corpus = load_corpus(corpus_path)
    selected = set(args.site)
    pool = list(corpus.get("local_sites") or []) if args.local_only else (
        list(corpus.get("local_sites") or []) + list(corpus.get("sites") or [])
    )
    if selected:
        sites = [site for site in pool if site["id"] in selected]
    elif args.local_only:
        sites = list(corpus.get("local_sites") or [])
    else:
        sites = list(corpus.get("sites") or [])
    missing = selected - {str(site["id"]) for site in pool}
    if missing:
        raise SystemExit(f"Unknown site ids: {', '.join(sorted(missing))}")
    if not sites:
        raise SystemExit("No sites selected")
    repeats = args.repeats if args.repeats is not None else int(corpus.get("repeats") or 1)
    if repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    failures = 0
    for site in sites:
        for repeat in range(1, repeats + 1):
            session_id = f"validation_{site['id']}_{timestamp}_r{repeat}"
            cmd = build_command(
                site=site,
                corpus=corpus,
                session_id=session_id,
                stage=args.stage,
            )
            print("+", subprocess.list2cmdline(cmd))
            if args.dry_run:
                continue
            started_at = datetime.now(UTC).isoformat()
            result = subprocess.run(cmd, cwd=ROOT, check=False)
            capture = inspect_capture(session_id, corpus.get("promotion_gates") or {})
            row = {
                "session_id": session_id,
                "site_id": site["id"],
                "url": site["url"],
                "archetype": site.get("archetype"),
                "stage": args.stage,
                "started_at": started_at,
                "exit_code": result.returncode,
                **capture,
            }
            append_run_log(row)
            if result.returncode or not capture.get("promoted"):
                failures += 1
                print(
                    f"FAILED {site['id']} repeat {repeat}: "
                    f"exit={result.returncode} reason={capture.get('reason')}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"OK {site['id']} repeat {repeat}: "
                    f"session={session_id} snapshots={capture.get('snapshot_count')}"
                )
    if failures and not args.dry_run:
        raise SystemExit(f"{failures} capture(s) failed or failed promotion gates")


if __name__ == "__main__":
    main()
