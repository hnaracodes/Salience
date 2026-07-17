"""Build a frozen manifest of clips with valid real-subcortical TRIBE intermediates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.artifacts import classify_npz, npz_path
from scout_core.horikawaCode.labels import resolve_video_path


def _sort_key(sid: str) -> tuple:
    return (0, int(sid)) if str(sid).isdigit() else (1, str(sid))


def _load_clips(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("clips") or [])


def build_ready_manifest(
    *,
    source_clips: list[dict],
    intermediates_dir: Path,
    videos_dir: Path | None,
    max_stimulus_id: int | None,
    corpus: str,
) -> tuple[list[dict], dict]:
    clips = sorted(source_clips, key=lambda c: _sort_key(str(c["stimulus_id"])))
    if max_stimulus_id is not None:
        clips = [
            c for c in clips
            if str(c["stimulus_id"]).isdigit() and int(c["stimulus_id"]) <= max_stimulus_id
        ]

    ready: list[dict] = []
    missing: list[str] = []
    stale: list[str] = []
    invalid: list[str] = []

    for clip in clips:
        sid = str(clip["stimulus_id"])
        path = npz_path(intermediates_dir, sid)
        status = classify_npz(path)
        if status == "ok":
            entry = dict(clip)
            entry["intermediate_path"] = str(path)
            if videos_dir is not None and not entry.get("video_path"):
                vp = resolve_video_path(videos_dir, sid)
                if vp is not None:
                    entry["video_path"] = str(vp)
            ready.append(entry)
        elif status == "missing":
            missing.append(sid)
        elif status == "stale":
            stale.append(sid)
        else:
            invalid.append(sid)

    audit = {
        "corpus": corpus,
        "n_requested": len(clips),
        "n_ready": len(ready),
        "n_missing": len(missing),
        "n_stale": len(stale),
        "n_invalid": len(invalid),
        "max_stimulus_id": max_stimulus_id,
        "missing_stimulus_ids": missing,
        "stale_stimulus_ids": stale,
        "invalid_stimulus_ids": invalid,
    }
    return ready, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "full_2181.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "pilot_275.json",
    )
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "videos",
    )
    parser.add_argument("--max-stimulus-id", type=int, default=None)
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args()

    corpus = args.corpus or args.output.stem
    clips = _load_clips(args.source_manifest)
    ready, audit = build_ready_manifest(
        source_clips=clips,
        intermediates_dir=args.intermediates_dir,
        videos_dir=args.videos_dir,
        max_stimulus_id=args.max_stimulus_id,
        corpus=corpus,
    )

    payload = {"corpus": corpus, "clips": ready}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    audit_path = args.output.with_name(args.output.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    print(f"Wrote {args.output} ({len(ready)} clips)")
    print(f"Audit -> {audit_path}")


if __name__ == "__main__":
    main()
