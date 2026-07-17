"""Correlate decoder/template emotion traces with SAM ratings (transfer report)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or y.size < 3:
        return None
    if x.std() < 1e-8 or y.std() < 1e-8:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "validation_study.yaml",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "validation" / "reports" / "transfer_report.json",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    threshold = float(cfg.get("product_gate", {}).get("min_transfer_r_valence", 0.25))
    sessions_root = PROJECT_ROOT / "scout_data" / "validation" / "sessions"

    rows: list[dict] = []
    for meta_path in sessions_root.glob("*/validation_meta.json"):
        session_id = meta_path.parent.name
        ratings_path = meta_path.parent / "ratings.json"
        bundle_path = PROJECT_ROOT / "scout_data" / "sessions" / session_id / "analysis_bundle.json"
        if not ratings_path.is_file() or not bundle_path.is_file():
            continue
        ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        emotion = bundle.get("emotion_decoder_track") or bundle.get("emotion_track") or {}
        valence_rating = float(ratings.get("sam", {}).get("valence", ratings.get("valence", 0.5)))
        mode = emotion.get("mode", bundle.get("emotion_mode", "template"))
        if bundle.get("emotion_decoder_track"):
            mode = "decoder"
        elif mode != "decoder":
            mode = "template"
        if mode == "decoder":
            probs = np.array(emotion.get("probabilities") or [], dtype=np.float32)
            names = emotion.get("class_names") or []
            valence_idx = names.index("contentment") if "contentment" in names else 0
            trace = probs[:, valence_idx] if probs.size else np.array([])
        else:
            z = np.array(emotion.get("z_scores") or [], dtype=np.float32)
            names = emotion.get("template_names") or []
            idx = names.index("contentment") if "contentment" in names else 0
            trace = z[:, idx] if z.size else np.array([])
        mean_trace = float(np.mean(trace)) if trace.size else 0.0
        rows.append({
            "session_id": session_id,
            "mode": mode,
            "valence_rating": valence_rating,
            "mean_trace": mean_trace,
        })

    x = np.array([r["valence_rating"] for r in rows], dtype=np.float32)
    y = np.array([r["mean_trace"] for r in rows], dtype=np.float32)
    transfer_r = _pearson(x, y)

    report = {
        "n_sessions": len(rows),
        "sessions": rows,
        "transfer_r_valence": transfer_r,
        "threshold": threshold,
        "decoder_default_recommended": transfer_r is not None and transfer_r >= threshold,
        "status": "pilot" if len(rows) < int(cfg.get("pilot", {}).get("min_participants", 10)) else "full",
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
