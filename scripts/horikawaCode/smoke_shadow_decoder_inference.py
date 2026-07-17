"""Smoke-test shadow decoder inference parity on one Horikawa intermediate NPZ.

Creates a temporary session-like dir, runs predict_emotion_track, and checks
feature width / output shape against the exported bundle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.affect_features import build_affect_features, resolve_feature_spec
from scout_core.horikawaCode.prepare import load_both_npz
from scout_core.mvpa_engine import load_decoder_bundle, predict_emotion_track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to *_both.npz intermediate (default: first ready_all clip)",
    )
    parser.add_argument("--model-id", default="horikawa_ridge_v1")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "scout_data/horikawaCode/reports/shadow_inference_smoke.json",
    )
    args = parser.parse_args()

    npz_path = args.npz
    if npz_path is None:
        manifest = PROJECT_ROOT / "scout_data/horikawaCode/manifests/ready_all.json"
        clips = json.loads(manifest.read_text(encoding="utf-8")).get("clips") or []
        if not clips:
            raise SystemExit("No clips in ready_all manifest")
        sid = str(clips[0]["stimulus_id"])
        cand = PROJECT_ROOT / "scout_data/horikawaCode/intermediates_modal" / f"{sid}_both.npz"
        if not cand.is_file() and sid.isdigit():
            cand = PROJECT_ROOT / "scout_data/horikawaCode/intermediates_modal" / f"{int(sid):04d}_both.npz"
        npz_path = cand
    if not npz_path.is_file():
        raise SystemExit(f"Missing NPZ: {npz_path}")

    cortical, sub = load_both_npz(npz_path)
    pipeline, label_map, meta = load_decoder_bundle(args.model_id)
    feature_spec = str(meta.get("feature_spec", "fused_schaefer400_subcortical_v1"))
    spec = resolve_feature_spec(feature_spec)
    X = build_affect_features(cortical, sub, spec=spec)
    result = predict_emotion_track(
        cortical,
        sub,
        model_id=args.model_id,
        feature_spec_name=feature_spec,
    )
    scores = np.asarray(result["scores"], dtype=np.float32)
    probs = np.asarray(result["probabilities"], dtype=np.float32)

    report = {
        "npz_path": str(npz_path),
        "model_id": args.model_id,
        "feature_spec": feature_spec,
        "cortical_shape": list(cortical.shape),
        "subcortical_shape": list(sub.shape),
        "feature_matrix_shape": list(X.shape),
        "expected_n_features": int(spec.n_features),
        "feature_width_ok": int(X.shape[1]) == int(spec.n_features),
        "n_classes_meta": int(meta.get("n_classes", 0)),
        "class_names": result.get("class_names"),
        "scores_shape": list(scores.shape),
        "probabilities_shape": list(probs.shape),
        "scores_finite": bool(np.all(np.isfinite(scores))),
        "prob_range": [float(probs.min()), float(probs.max())] if probs.size else None,
        "cv_mean_r_meta": meta.get("cv_mean_r"),
        "corpus_meta": meta.get("corpus"),
        "n_samples_meta": meta.get("n_samples"),
        "inference_ok": bool(
            X.shape[1] == spec.n_features
            and scores.shape[0] == cortical.shape[0]
            and scores.shape[1] == len(result.get("class_names") or [])
            and np.all(np.isfinite(scores))
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.report}")
    if not report["inference_ok"]:
        raise SystemExit("Shadow inference smoke FAILED")


if __name__ == "__main__":
    main()
