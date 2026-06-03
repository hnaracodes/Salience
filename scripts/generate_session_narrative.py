#!/usr/bin/env python3
"""Generate marketing_narrative from structured analysis_bundle (no raw HTML).

Usage:
    python scripts/generate_session_narrative.py --session-id <id>
    python scripts/generate_session_narrative.py --session-id <id> --provider template
    python scripts/generate_session_narrative.py --session-id <id> --goal "..." --script configs/walkthrough_scripts/aurora_showcase.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from activation_store import SESSIONS_DIR
from scout_core.element_goals import resolve_site_goal
from scout_core.llm_narrative import generate_marketing_narrative
from scout_core.schemas import analysis_bundle_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--provider", default=None, help="Override configs/llm_narrative.yaml provider")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--goal", default=None, help="Single paragraph site goal (overrides script)")
    parser.add_argument("--script", type=Path, default=None, help="Walkthrough YAML with optional site_goal:")
    args = parser.parse_args()

    session_dir = SESSIONS_DIR / args.session_id
    bundle_path = analysis_bundle_path(session_dir)
    if not bundle_path.is_file():
        raise SystemExit(f"Missing {bundle_path} — run analyze_session.py first.")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not bundle.get("section_report"):
        raise SystemExit("section_report[] empty — run analyze_session.py --website/--sections first.")

    script_path = args.script
    if script_path is None:
        candidate = session_dir / "walkthrough_script.yaml"
        if candidate.is_file():
            script_path = candidate

    site_goal = resolve_site_goal(override=args.goal, script_path=script_path)
    narrative = generate_marketing_narrative(
        bundle,
        site_goal=site_goal,
        config_path=args.config,
        provider=args.provider,
    )
    bundle["marketing_narrative"] = narrative.model_dump()
    if bundle.get("schema_version", 1) < 3:
        bundle["schema_version"] = 3
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(
        f"Wrote marketing_narrative ({narrative.provider}, "
        f"{len(narrative.element_insights)} element insights) → {bundle_path}"
    )


if __name__ == "__main__":
    main()
