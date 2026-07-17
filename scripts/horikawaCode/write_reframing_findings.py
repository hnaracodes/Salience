"""Generate reframing_findings.md from ablation and alignment reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

GATE_MIN_R = 0.15
GATE_MAX_P = 0.05


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "reports",
    )
    args = parser.parse_args()

    phase0 = _load_json(args.reports_dir / "phase0_consistency.json")
    alignment = _load_json(args.reports_dir / "alignment_report.json")
    ablation = _load_json(args.reports_dir / "ablation_matrix.json")

    lines = [
        "# Horikawa reframing findings",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Phase 0 — metric consistency",
        "",
    ]
    if phase0:
        lines += [
            f"- Train `cv_mean_r` == eval `fused_lovo_mean_r`: **{phase0.get('consistent')}** (delta={phase0.get('abs_delta')})",
            f"- Corpus: `{phase0.get('corpus_npz')}` | N={phase0.get('n_samples_npz')}",
            f"- Synthetic fallback: {phase0.get('synthetic_fallback_used')}",
            "",
        ]
    else:
        lines += ["- phase0_consistency.json not found", ""]

    lines += ["## Phase 1 — label alignment", ""]
    if alignment:
        lines += [
            f"- Verdict: **{alignment.get('verdict')}**",
            f"- Aligned LOVO r: {alignment.get('aligned_lovo_mean_r', 0):.4f}",
            f"- Shuffle control r: {alignment.get('shuffle_negative_control_r', 0):.4f}",
            f"- In-sample mean R²: {alignment.get('in_sample_mean_r2', 0):.4f}",
            f"- Off-by-one red flag: {alignment.get('misalignment_red_flag')}",
            f"- Spot-check: `{alignment.get('spotcheck_csv')}`",
            "",
        ]
    else:
        lines += ["- alignment_report.json not found", ""]

    lines += ["## Phase 4 — ablation matrix", ""]
    proceed = False
    if ablation:
        best = ablation.get("best_cell") or {}
        lines += [
            f"- Clips: {ablation.get('n_clips')}",
            f"- Best cell: `{best.get('feature_spec')}` × `{best.get('label_framing')}`",
            f"  - LOVO mean r: **{best.get('lovo_mean_r', 0):.4f}**",
            f"  - Permutation p: {best.get('permutation_p_value', 1):.4f}",
            f"- Significant cells (p<{GATE_MAX_P}): {ablation.get('n_significant_p005', 0)}",
            "",
            "### Top cells by LOVO r",
            "",
            "| feature_spec | label_framing | LOVO r | p-value |",
            "|---|---:|---:|---:|",
        ]
        cells = sorted(ablation.get("cells") or [], key=lambda c: c.get("lovo_mean_r", -9), reverse=True)
        for cell in cells[:8]:
            lines.append(
                f"| {cell.get('feature_spec')} | {cell.get('label_framing')} | "
                f"{cell.get('lovo_mean_r', 0):.4f} | {cell.get('permutation_p_value', 1):.4f} |"
            )
        lines.append("")
        proceed = (
            float(best.get("lovo_mean_r", 0)) >= GATE_MIN_R
            and float(best.get("permutation_p_value", 1)) < GATE_MAX_P
        )
    else:
        lines += ["- ablation_matrix.json not found", ""]

    lines += [
        "## Decision gate — resume Modal full corpus?",
        "",
        f"Criteria: LOVO mean r ≥ {GATE_MIN_R} AND permutation p < {GATE_MAX_P} on some framing after fixes.",
        "",
        f"**Recommendation: {'PROCEED' if proceed else 'DO NOT PROCEED'}** with full Modal TRIBE extraction.",
        "",
        "## Subcortical fix",
        "",
        "- Real Harvard-Oxford voxel map extracted via Modal (`configs/subcortical_voxel_regions.csv`).",
        "- Compare `subcortical_fake8_v1` vs `subcortical_real16_v1` / `fused_v2_real_subcortical` in ablation table above.",
        "",
    ]

    out = args.reports_dir / "reframing_findings.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Modal resume gate: {'PASS' if proceed else 'FAIL'}")


if __name__ == "__main__":
    main()
