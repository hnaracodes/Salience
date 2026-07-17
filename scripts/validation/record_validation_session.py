"""Record validation session with optional post-task SAM ratings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--url", default="https://example.com")
    parser.add_argument("--ratings", type=Path, default=None, help="Merge SAM/UX ratings JSON")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "validation_study.yaml",
    )
    args = parser.parse_args()

    cfg_path = args.config
    cfg = {}
    if cfg_path.is_file():
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    session_dir = PROJECT_ROOT / "scout_data" / "validation" / "sessions" / args.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    if args.ratings:
        if not args.ratings.is_file():
            raise SystemExit(f"Ratings file not found: {args.ratings}")
        ratings = json.loads(args.ratings.read_text(encoding="utf-8"))
        out = session_dir / "ratings.json"
        out.write_text(json.dumps(ratings, indent=2), encoding="utf-8")
        print(f"Saved ratings -> {out}")
        return

    explore_script = cfg.get("explore_script", "configs/explore_production.yaml")
    py = sys.executable
    subprocess.run(
        [
            py,
            str(PROJECT_ROOT / "scripts" / "run_website_session.py"),
            "--session-id",
            args.session_id,
            "--url",
            args.url,
            "--script",
            str(PROJECT_ROOT / explore_script),
            "--stage",
            "all",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    meta = {
        "session_id": args.session_id,
        "url": args.url,
        "protocol": cfg.get("protocol_id", "ux_affect_v1"),
        "source_session_dir": str(
            (PROJECT_ROOT / "scout_data" / "sessions" / args.session_id).resolve()
        ),
    }
    (session_dir / "validation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Validation session recorded -> {session_dir}")


if __name__ == "__main__":
    main()
