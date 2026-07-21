"""One-clip Modal smoke test: prove subcortical preds are from tribev2-subcortical, not proxy."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.labels import resolve_video_path
from scout_core.subcortical.atlas import N_SUBCORTICAL_VOXELS, build_region_index
from scout_core.subcortical.tribev2_adapter import (
    DEFAULT_SUBCORTICAL_CHECKPOINT,
    _cortical_to_subcortical_proxy,
)


def _within_roi_voxel_std(sub: np.ndarray) -> dict[str, float]:
    """Proxy fallback repeats one value per ROI — real model has within-ROI variance."""
    ri = build_region_index()
    per_roi: dict[str, float] = {}
    for rid in range(int(ri.max()) + 1):
        mask = ri == rid
        per_roi[f"roi_{rid}"] = float(sub[:, mask].std())
    return {
        "mean_within_roi_std": float(np.mean(list(per_roi.values()))),
        "min_within_roi_std": float(np.min(list(per_roi.values()))),
        "max_within_roi_std": float(np.max(list(per_roi.values()))),
        "per_roi": per_roi,
    }


def analyze_pair(cortical: np.ndarray, subcortical: np.ndarray, *, prediction_source: str | None = None) -> dict:
    proxy = _cortical_to_subcortical_proxy(cortical)
    diff = np.abs(subcortical - proxy)
    corr = float(np.corrcoef(subcortical.ravel(), proxy.ravel())[0, 1])
    roi_stats = _within_roi_voxel_std(subcortical)
    is_real_checkpoint = prediction_source in {
        "tribev2_subcortical_checkpoint",
        "tribev2_subcortical_checkpoint_regions",
    }
    is_likely_real = is_real_checkpoint or (
        float(np.max(diff)) > 0.01
        and roi_stats["mean_within_roi_std"] > 1e-4
        and corr < 0.999
    )
    return {
        "cortical_shape": list(cortical.shape),
        "subcortical_shape": list(subcortical.shape),
        "expected_subcortical_voxels": N_SUBCORTICAL_VOXELS,
        "checkpoint": DEFAULT_SUBCORTICAL_CHECKPOINT,
        "prediction_source": prediction_source,
        "max_abs_sub_minus_proxy": float(np.max(diff)),
        "mean_abs_sub_minus_proxy": float(np.mean(diff)),
        "corr_sub_vs_proxy": corr,
        "sub_mean": float(subcortical.mean()),
        "sub_std": float(subcortical.std()),
        **roi_stats,
        "verdict": (
            "real_subcortical_model"
            if is_likely_real
            else "likely_cortical_proxy_fallback"
        ),
        "pass": is_likely_real,
    }


def run_modal_clip(video_path: Path) -> tuple[np.ndarray, np.ndarray, dict, float]:
    from scripts.horikawaCode.generate_tribev2_features import TribeModalBothBatch

    t0 = time.perf_counter()
    with TribeModalBothBatch() as batch:
        cortical, sub, meta = batch.predict_both(video_path)
    elapsed = time.perf_counter() - t0
    return cortical, sub, meta, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimulus-id", default="1")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "reports" / "subcortical_verify.json",
    )
    parser.add_argument(
        "--analyze-only",
        type=Path,
        default=None,
        help="Skip Modal; analyze existing *_both.npz",
    )
    args = parser.parse_args()

    if args.analyze_only:
        data = np.load(args.analyze_only)
        cortical = data["preds"]
        sub = data["preds_subcortical"]
        src = str(data["prediction_source"]) if "prediction_source" in data else None
        report = {
            "mode": "analyze_only",
            "source": str(args.analyze_only),
            "analysis": analyze_pair(cortical, sub, prediction_source=src),
        }
    else:
        vp = resolve_video_path(args.videos_dir, args.stimulus_id)
        if vp is None or not vp.is_file():
            raise SystemExit(f"No video for stimulus_id={args.stimulus_id!r} under {args.videos_dir}")
        print(f"[verify] Modal clip stimulus_id={args.stimulus_id} video={vp}", flush=True)
        cortical, sub, sub_meta, elapsed = run_modal_clip(vp)
        if cortical.shape[1] != 20484 or sub.shape[1] != N_SUBCORTICAL_VOXELS:
            raise SystemExit(f"Bad shapes: cortical={cortical.shape} sub={sub.shape}")
        analysis = analyze_pair(
            cortical,
            sub,
            prediction_source=sub_meta.get("prediction_source"),
        )
        out_npz = (
            PROJECT_ROOT
            / "scout_data"
            / "horikawaCode"
            / "intermediates_modal"
            / f"verify_{args.stimulus_id}_both.npz"
        )
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            preds=cortical.astype(np.float32),
            preds_subcortical=sub.astype(np.float32),
            stimulus_id=np.asarray(args.stimulus_id),
            tr_hz=np.float32(2.0),
            prediction_source=np.asarray(sub_meta.get("prediction_source", "unknown")),
            tribe_checkpoint=np.asarray(sub_meta.get("tribe_checkpoint", DEFAULT_SUBCORTICAL_CHECKPOINT)),
        )
        report = {
            "mode": "modal_live",
            "stimulus_id": args.stimulus_id,
            "video_path": str(vp),
            "elapsed_s": round(elapsed, 2),
            "artifact": str(out_npz),
            "sub_meta": sub_meta,
            "analysis": analysis,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    if not report["analysis"]["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
