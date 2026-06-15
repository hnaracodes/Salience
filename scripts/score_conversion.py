#!/usr/bin/env python3
"""Score conversion intent for a session bundle.

Usage:
    python scripts/score_conversion.py --session-id <id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activation_store import SESSIONS_DIR
from scout_core.conversion_model import LogisticConversionModel, score_bundle

DEFAULT_MODEL = ROOT / "scout_data" / "models" / "conversion_v1.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--write-bundle", action="store_true")
    args = parser.parse_args()

    session_dir = SESSIONS_DIR / args.session_id
    bundle_path = session_dir / "analysis_bundle.json"
    if not bundle_path.is_file():
        raise SystemExit(f"Missing {bundle_path}")
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}. Run train_conversion_model.py first.")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    model = LogisticConversionModel.load(args.model)
    prediction = score_bundle(bundle, model)
    print(json.dumps(prediction, indent=2))

    if args.write_bundle:
        bundle["conversion_prediction"] = prediction
        bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(f"Updated {bundle_path}")


if __name__ == "__main__":
    main()
