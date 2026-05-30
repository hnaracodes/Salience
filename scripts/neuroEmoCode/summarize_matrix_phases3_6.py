#!/usr/bin/env python3
"""Summarize phases 3-6 NeuroEmo matrix results for the session report."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "scout_data/neuroEmoCode/models/neuroemo_remaining_high_leverage_matrix_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "scout_data/neuroEmoCode/models/2026-05-27_surface_annot_matrix"
BASELINE_FLAT = {"acc": 0.445, "macro_f1": 0.433494}
BASELINE_SURFACE = {"acc": 0.42, "macro_f1": 0.412}
HARD_CLASSES = ("afraid", "calm", "delighted")


def _load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        alt = path.with_name(path.stem.replace(".metrics", "_metrics") + ".json")
        path = alt if alt.is_file() else path
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cv_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    cv = metrics.get("cross_validation") or {}
    return {
        "mean_accuracy": cv.get("mean_accuracy"),
        "mean_balanced_accuracy": cv.get("mean_balanced_accuracy"),
        "mean_macro_f1": cv.get("mean_macro_f1"),
        "pooled_log_loss": cv.get("pooled_log_loss"),
    }


def _per_class_f1(metrics: dict[str, Any]) -> dict[str, float | None]:
    cv = metrics.get("cross_validation") or {}
    pooled = cv.get("pooled_per_class") or cv.get("per_class") or {}
    out: dict[str, float | None] = {}
    labels = metrics.get("labels") or []
    for label in labels:
        entry = pooled.get(label) if isinstance(pooled, dict) else None
        if isinstance(entry, dict):
            out[str(label)] = entry.get("f1")
    return out


def _top_by_macro(runs: list[dict[str, Any]], n: int = 3) -> list[dict[str, Any]]:
    scored = []
    for run in runs:
        summary = run.get("metrics_summary") or {}
        f1 = summary.get("mean_macro_f1")
        if f1 is None:
            metrics = _load_metrics(PROJECT_ROOT / run["metrics_path"])
            if metrics:
                summary = _cv_summary(metrics)
                f1 = summary.get("mean_macro_f1")
        if f1 is None:
            continue
        scored.append({**run, "mean_macro_f1": f1, "metrics_summary": summary})
    scored.sort(key=lambda row: row["mean_macro_f1"], reverse=True)
    return scored[:n]


def _parse_timing_name(name: str) -> dict[str, str]:
    m = re.match(r"timing_lag(\d+(?:\.\d+)?)s_drop(\d+)tr_stride(\d+)", name)
    if not m:
        return {}
    return {"lag_s": m.group(1), "drop_trs": m.group(2), "stride": m.group(3)}


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing manifest: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runs = manifest.get("runs") or []

    by_phase: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_phase.setdefault(run.get("phase", "unknown"), []).append(run)

    flat_ref = _load_metrics(OUTPUT_DIR / "logistic_saga_5class_10tr_metrics.json")
    if flat_ref is None:
        flat_ref = _load_metrics(OUTPUT_DIR / "logistic_saga_5class_10tr.metrics.json")

    static_baseline = None
    for run in by_phase.get("phase4_dynamic_window_features", []):
        if run["name"].endswith("_mean") or "10tr_mean" in run["name"]:
            static_baseline = run
            break
    if static_baseline is None:
        for run in by_phase.get("phase4_dynamic_window_features", []):
            if "static" not in run["name"] and "dynamic" not in run["name"]:
                continue
        static_name = "logistic_saga_5class_10tr_mean"
        static_baseline = next((r for r in by_phase.get("phase4_dynamic_window_features", []) if r["name"] == static_name), None)

    report = {
        "manifest_path": str(MANIFEST.relative_to(PROJECT_ROOT)),
        "n_runs_total": len(runs),
        "baselines": {"projected_postfix": BASELINE_FLAT, "surface_logistic": BASELINE_SURFACE},
        "phase3_timing_grid": {
            "n_specs": len(by_phase.get("phase3_timing_grid", [])),
            "n_completed": sum(
                1
                for r in by_phase.get("phase3_timing_grid", [])
                if (PROJECT_ROOT / r["metrics_path"]).is_file()
            ),
            "top3_macro_f1": _top_by_macro(by_phase.get("phase3_timing_grid", [])),
        },
        "phase4_dynamic_features": {
            "n_specs": len(by_phase.get("phase4_dynamic_window_features", [])),
            "n_completed": sum(
                1
                for r in by_phase.get("phase4_dynamic_window_features", [])
                if (PROJECT_ROOT / r["metrics_path"]).is_file()
            ),
            "top3_macro_f1": _top_by_macro(by_phase.get("phase4_dynamic_window_features", [])),
            "static_reducer_note": "Static baseline is temporal_reducer=mean (default contiguous window).",
        },
        "phase5_preprocessing_roi": {
            "n_specs": len(by_phase.get("phase5_preprocessing_roi_ablation", [])),
            "n_completed": sum(
                1
                for r in by_phase.get("phase5_preprocessing_roi_ablation", [])
                if (PROJECT_ROOT / r["metrics_path"]).is_file()
            ),
            "top3_macro_f1": _top_by_macro(by_phase.get("phase5_preprocessing_roi_ablation", [])),
        },
        "phase6_structured_decoding": {
            "n_specs": len(by_phase.get("phase6_structured_decoding", [])),
            "n_completed": sum(
                1
                for r in by_phase.get("phase6_structured_decoding", [])
                if (PROJECT_ROOT / r["metrics_path"]).is_file()
            ),
            "top3_macro_f1": _top_by_macro(by_phase.get("phase6_structured_decoding", [])),
            "flat_reference": _cv_summary(flat_ref) if flat_ref else None,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
