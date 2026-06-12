#!/usr/bin/env python3
"""Compare multiple sessions on norm-referenced marketing scores.

Usage:
    python scripts/compare_sessions.py --session-ids ID1,ID2,ID3 --norm-id naturalistic_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activation_store import SESSIONS_DIR
from scout_core.marketing_scores import build_marketing_scores, load_marketing_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-ids", required=True, help="Comma-separated session IDs")
    parser.add_argument("--norm-id", default="naturalistic_v1")
    args = parser.parse_args()

    cfg = load_marketing_config()
    cross = dict(cfg.get("cross_session") or {})
    cross["enabled"] = True
    cross["norm_id"] = args.norm_id
    cross["comparison_mode"] = "norm_referenced"
    cfg["cross_session"] = cross

    rows = []
    for sid in args.session_ids.split(","):
        sid = sid.strip()
        session_dir = SESSIONS_DIR / sid
        bundle_path = session_dir / "analysis_bundle.json"
        if not bundle_path.is_file():
            print(f"Skip {sid}: no analysis_bundle.json", file=sys.stderr)
            continue
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        ms = build_marketing_scores(session_dir, bundle, config=cfg)
        rows.append({
            "session_id": sid,
            "comparison_score": ms.get("comparison_score"),
            "overall_score": ms.get("overall_score"),
            "comparison_mode": (ms.get("provenance") or {}).get("comparison_mode"),
            "norm_id": (ms.get("provenance") or {}).get("norm_id"),
        })

    rows.sort(key=lambda r: r.get("comparison_score") or 0, reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    out = {"norm_id": args.norm_id, "sessions": rows}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
