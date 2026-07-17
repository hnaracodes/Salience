"""Verify Horikawa label<->video alignment (stats + manual spot-check)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.constants import HORIKAWA_14_DIMENSIONS, HORIKAWA_34_CATEGORIES
from scout_core.horikawaCode.cv import in_sample_mean_r2, lovo_mean_r
from scout_core.horikawaCode.labels import DEFAULT_LABEL_CACHE, load_horikawa_ratings, resolve_video_path
from scout_core.horikawaCode.prepare import build_train_npz_from_manifest, load_both_npz


def _load_manifest(path: Path) -> list[dict]:
    return list(json.loads(path.read_text(encoding="utf-8")).get("clips") or [])


def _duration_stats(clips: list[dict], intermediates_dir: Path) -> dict:
    durations: list[float] = []
    for clip in clips:
        sid = str(clip["stimulus_id"])
        npz_path = intermediates_dir / f"{sid}_both.npz"
        if not npz_path.is_file() and sid.isdigit():
            npz_path = intermediates_dir / f"{int(sid):04d}_both.npz"
        if not npz_path.is_file():
            continue
        cortical, _ = load_both_npz(npz_path)
        durations.append(float(cortical.shape[0]))
    if not durations:
        return {"n_clips": 0}
    arr = np.asarray(durations, dtype=np.float32)
    return {
        "n_clips": len(durations),
        "tr_mean": float(arr.mean()),
        "tr_std": float(arr.std()),
        "tr_min": float(arr.min()),
        "tr_max": float(arr.max()),
    }


def _shifted_y(data: dict, shift: int) -> np.ndarray:
    y = np.asarray(data["y"], dtype=np.float32).copy()
    sids = [str(s) for s in data["stimulus_id"].tolist()]
    ratings = load_horikawa_ratings()
    ordered = sorted(sids, key=lambda s: int(s) if s.isdigit() else s)
    id_to_row = {sid: i for i, sid in enumerate(ordered)}
    for i, sid in enumerate(sids):
        if not sid.isdigit():
            continue
        shifted_id = str(int(sid) + shift)
        if shifted_id in ratings and shifted_id in id_to_row:
            from scout_core.horikawaCode.labels import build_y_from_label_dict

            y[i] = build_y_from_label_dict(ratings[shifted_id], str(data.get("target_type", "product_8")))
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "ready_all.json",
    )
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal",
    )
    parser.add_argument("--labels-cache", type=Path, default=DEFAULT_LABEL_CACHE)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--n-spotcheck", type=int, default=10)
    args = parser.parse_args()

    clips = _load_manifest(args.manifest)
    corpus = json.loads(args.manifest.read_text(encoding="utf-8")).get("corpus", args.manifest.stem)
    ratings = load_horikawa_ratings(args.labels_cache)
    all_label_ids = set(ratings.keys())
    manifest_ids = {str(c["stimulus_id"]) for c in clips}

    data = build_train_npz_from_manifest(
        clips,
        intermediates_dir=args.intermediates_dir,
        labels_cache=args.labels_cache,
        corpus=corpus,
        target="dimensions_14",
    )
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    groups = np.asarray(data["groups"], dtype=np.int32)

    aligned_r = lovo_mean_r(X, y, groups, args.alphas)
    shuffle_r = lovo_mean_r(X, y[np.random.default_rng(0).permutation(len(y))], groups, args.alphas)
    in_sample_r2 = in_sample_mean_r2(X, y, args.alphas)
    print(f"aligned LOVO r={aligned_r:.4f} (N={len(X)})", flush=True)
    off_by_one_plus = lovo_mean_r(X, _shifted_y(data, +1), groups, args.alphas)
    print(f"off-by-one +1 r={off_by_one_plus:.4f}", flush=True)
    off_by_one_minus = lovo_mean_r(X, _shifted_y(data, -1), groups, args.alphas)
    print(f"off-by-one -1 r={off_by_one_minus:.4f}", flush=True)

    misalignment_red_flag = (
        off_by_one_plus >= aligned_r - 0.02 or off_by_one_minus >= aligned_r - 0.02
    )
    verdict = "aligned"
    if misalignment_red_flag:
        verdict = "misaligned_suspect"
    elif aligned_r < 0.05 and in_sample_r2 < 0.05:
        verdict = "no_signal_or_misaligned"
    elif aligned_r < 0.05:
        verdict = "weak_signal"

    reports_dir = PROJECT_ROOT / "scout_data" / "horikawaCode" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    spot_path = reports_dir / "alignment_spotcheck.csv"
    videos_dir = PROJECT_ROOT / "scout_data" / "horikawaCode" / "videos"
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(clips), size=min(args.n_spotcheck, len(clips)), replace=False)
    with spot_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stimulus_id",
                "video_filename",
                "n_trs",
                "argmax_category",
                "top3_categories",
                "valence_norm",
                "arousal_norm",
            ],
        )
        writer.writeheader()
        for idx in sorted(sample_idx.tolist()):
            clip = clips[idx]
            sid = str(clip["stimulus_id"])
            vp = resolve_video_path(videos_dir, sid)
            label = ratings[sid]
            cats = label["categories_34"]
            dims = label["dimensions_14"]
            top3 = sorted(
                zip(HORIKAWA_34_CATEGORIES, cats.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            npz_path = args.intermediates_dir / f"{sid}_both.npz"
            if not npz_path.is_file() and sid.isdigit():
                npz_path = args.intermediates_dir / f"{int(sid):04d}_both.npz"
            n_trs = int(load_both_npz(npz_path)[0].shape[0]) if npz_path.is_file() else -1
            writer.writerow({
                "stimulus_id": sid,
                "video_filename": vp.name if vp else "",
                "n_trs": n_trs,
                "argmax_category": HORIKAWA_34_CATEGORIES[int(np.argmax(cats))],
                "top3_categories": ";".join(f"{n}:{v:.3f}" for n, v in top3),
                "valence_norm": f"{(dims[HORIKAWA_14_DIMENSIONS.index('valence')] - 1) / 8:.3f}",
                "arousal_norm": f"{(dims[HORIKAWA_14_DIMENSIONS.index('arousal')] - 1) / 8:.3f}",
            })

    report = {
        "verdict": verdict,
        "n_manifest_clips": len(clips),
        "n_ratings_cache": len(all_label_ids),
        "id_overlap": len(manifest_ids & all_label_ids),
        "duration_stats": _duration_stats(clips, args.intermediates_dir),
        "aligned_lovo_mean_r": aligned_r,
        "shuffle_negative_control_r": shuffle_r,
        "in_sample_mean_r2": in_sample_r2,
        "off_by_one_plus_r": off_by_one_plus,
        "off_by_one_minus_r": off_by_one_minus,
        "misalignment_red_flag": misalignment_red_flag,
        "spotcheck_csv": str(spot_path),
    }
    out = reports_dir / "alignment_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out} and {spot_path}")


if __name__ == "__main__":
    main()
