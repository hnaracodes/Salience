"""Build pilot/full clip manifests for Horikawa TRIBE feature extraction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.constants import HORIKAWA_34_CATEGORIES
from scout_core.horikawaCode.labels import (
    DEFAULT_LABEL_CACHE,
    load_horikawa_ratings,
    resolve_video_path,
)


def _argmax_category(categories_34: np.ndarray) -> int:
    return int(np.argmax(categories_34))


def _load_manifest_clips(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("clips") or payload.get("stimulus_ids") or [])


def build_manifest(
    *,
    labels_cache: Path,
    videos_dir: Path | None,
    pilot_n: int | None,
    corpus: str,
    neutral_top_n: int = 15,
) -> tuple[list[dict], dict]:
    ratings = load_horikawa_ratings(labels_cache)
    by_class: dict[int, list[str]] = defaultdict(list)
    neutral: list[str] = []

    for sid, labels in ratings.items():
        cat = np.asarray(labels["categories_34"], dtype=np.float32)
        argmax = _argmax_category(cat)
        by_class[argmax].append(sid)
        if float(cat.max()) < 0.15:
            neutral.append(sid)

    selected: list[str] = []
    if pilot_n is not None:
        per_class = max(1, pilot_n // len(HORIKAWA_34_CATEGORIES))
        for class_idx in range(len(HORIKAWA_34_CATEGORIES)):
            pool = sorted(by_class[class_idx])
            selected.extend(pool[:per_class])
        for sid in sorted(neutral)[:neutral_top_n]:
            if sid not in selected:
                selected.append(sid)
        selected = sorted(set(selected))[:pilot_n]
    else:
        selected = sorted(ratings.keys(), key=lambda s: int(s) if s.isdigit() else s)

    clips: list[dict] = []
    missing_video = 0
    for sid in selected:
        video_path = None
        if videos_dir is not None:
            resolved = resolve_video_path(videos_dir, sid)
            if resolved is None:
                missing_video += 1
            else:
                video_path = str(resolved)
        cat = np.asarray(ratings[sid]["categories_34"], dtype=np.float32)
        clips.append({
            "stimulus_id": sid,
            "video_path": video_path,
            "argmax_category": HORIKAWA_34_CATEGORIES[_argmax_category(cat)],
            "max_category_score": float(cat.max()),
        })

    class_counts = Counter(c["argmax_category"] for c in clips)
    audit = {
        "corpus": corpus,
        "n_clips": len(clips),
        "n_with_video": sum(1 for c in clips if c.get("video_path")),
        "n_missing_video": missing_video,
        "class_counts": dict(sorted(class_counts.items())),
        "neutral_included": sum(1 for c in clips if c["max_category_score"] < 0.15),
    }
    return clips, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-cache", type=Path, default=DEFAULT_LABEL_CACHE)
    parser.add_argument("--videos-dir", type=Path, default=None)
    parser.add_argument("--pilot-n", type=int, default=None, help="Build stratified pilot manifest (~150)")
    parser.add_argument("--corpus", default="pilot_150")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "pilot_150.json",
    )
    args = parser.parse_args()

    if args.pilot_n is None and args.corpus == "pilot_150":
        args.pilot_n = 150

    corpus = args.corpus
    if args.pilot_n is None:
        corpus = "full_2181"

    clips, audit = build_manifest(
        labels_cache=args.labels_cache,
        videos_dir=args.videos_dir,
        pilot_n=args.pilot_n,
        corpus=corpus,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"corpus": corpus, "clips": clips}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    audit_path = args.output.with_name(args.output.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stimulus_id", "argmax_category", "max_category_score", "video_path"])
        writer.writeheader()
        writer.writerows(clips)

    print(f"Wrote {args.output}  n={len(clips)}")
    print(f"Audit -> {audit_path}")


if __name__ == "__main__":
    main()
