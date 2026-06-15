#!/usr/bin/env python3
"""Validate attention proxy against Clarity/behavioral ground truth.

Usage:
    python scripts/validate_attention_proxy.py
    python scripts/validate_attention_proxy.py --validation-dir scout_data/validation/attention_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scout_core.attention_calibration import validate_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=ROOT / "scout_data" / "validation" / "attention_v1",
    )
    parser.add_argument("--write-memo", type=Path, default=None)
    args = parser.parse_args()

    report = validate_corpus(args.validation_dir)
    print(json.dumps(report, indent=2))

    if args.write_memo:
        lines = [
            "# Attention Proxy Validation Memo",
            "",
            f"- Calibration ID: `{report.get('calibration_id')}`",
            f"- Sessions evaluated: {report.get('n_sessions')}",
            f"- Sessions passed: {report.get('n_passed')}",
            "",
        ]
        for row in report.get("sessions") or []:
            lines.append(
                f"- `{row.get('session_id')}`: spearman={row.get('spearman_mean')} "
                f"top1_hit={row.get('top1_hit_rate')} passed={row.get('passed')}",
            )
        args.write_memo.parent.mkdir(parents=True, exist_ok=True)
        args.write_memo.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Memo: {args.write_memo}")


if __name__ == "__main__":
    main()
