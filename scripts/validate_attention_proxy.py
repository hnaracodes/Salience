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
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scout_core.attention_calibration import (
    fit_attention_weight,
    load_calibration_config,
    validate_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=ROOT / "scout_data" / "validation" / "attention_v1",
    )
    parser.add_argument("--write-memo", type=Path, default=None)
    parser.add_argument("--fit-weights", action="store_true")
    parser.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="Evaluate only untouched holdout properties after fitting",
    )
    parser.add_argument("--write-config", type=Path, default=None)
    parser.add_argument("--calibration-id", default=None)
    args = parser.parse_args()

    fit_report = None
    attention_weight = None
    if args.fit_weights:
        fit_report = fit_attention_weight(args.validation_dir)
        attention_weight = float(fit_report["best"]["attention_weight"])

    roles = {"holdout"} if args.evaluate_holdout else None
    report = validate_corpus(
        args.validation_dir,
        roles=roles,
        attention_weight=attention_weight,
    )
    if fit_report is not None:
        report["weight_fit"] = fit_report
    print(json.dumps(report, indent=2))

    summary = report.get("summary") or {}
    if args.write_config:
        if not args.fit_weights:
            raise SystemExit("--write-config requires --fit-weights")
        if not args.evaluate_holdout:
            raise SystemExit("--write-config requires --evaluate-holdout")
        if not summary.get("passed"):
            raise SystemExit("Holdout gates failed; refusing to publish calibration config")
        config = load_calibration_config()
        config["calibration_id"] = args.calibration_id or (
            "clarity_" + datetime.now(UTC).strftime("%Y%m%d")
        )
        config["attention_weight"] = round(float(attention_weight), 4)
        config["click_weight"] = round(1.0 - float(attention_weight), 4)
        config["validation"] = {
            "evaluated_at": datetime.now(UTC).isoformat(),
            "n_holdout_pages": summary.get("n_qualifying"),
            "spearman_macro_mean": summary.get("spearman_macro_mean"),
            "spearman_macro_ci95": summary.get("spearman_macro_ci95"),
            "top1_macro_mean": summary.get("top1_macro_mean"),
            "top1_macro_ci95": summary.get("top1_macro_ci95"),
        }
        config_path = args.write_config
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"Config: {config_path}")

    if args.write_memo:
        memo_path = args.write_memo
        if not memo_path.is_absolute():
            memo_path = ROOT / memo_path
        lines = [
            "# Attention Proxy Validation Memo",
            "",
            f"- Calibration ID: `{report.get('calibration_id')}`",
            f"- Sessions evaluated: {report.get('n_sessions')}",
            f"- Sessions passed: {report.get('n_passed')}",
            f"- Roles evaluated: `{report.get('roles')}`",
            f"- Attention weight: `{report.get('attention_weight')}`",
            f"- Spearman macro mean: `{summary.get('spearman_macro_mean')}`",
            f"- Spearman 95% CI: `{summary.get('spearman_macro_ci95')}`",
            f"- Top-1 macro mean: `{summary.get('top1_macro_mean')}`",
            f"- Top-1 95% CI: `{summary.get('top1_macro_ci95')}`",
            f"- Claim gate passed: `{summary.get('passed')}`",
            "",
        ]
        for row in report.get("sessions") or []:
            lines.append(
                f"- `{row.get('session_id')}`: spearman={row.get('spearman_mean')} "
                f"top1_hit={row.get('top1_hit_rate')} passed={row.get('passed')}",
            )
        memo_path.parent.mkdir(parents=True, exist_ok=True)
        memo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Memo: {memo_path}")


if __name__ == "__main__":
    main()
