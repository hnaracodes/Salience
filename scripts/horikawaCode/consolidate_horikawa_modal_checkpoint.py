"""Keep valid Modal intermediates, discard stale/incomplete, write resume checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.labels import resolve_video_path

MANIFEST = PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "full_2181.json"
INTER_DIR = PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal"
VIDEOS_DIR = PROJECT_ROOT / "scout_data" / "horikawaCode" / "videos"
CHECKPOINT = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run" / "full_training_checkpoint.json"
LOCK = PROJECT_ROOT / "scout_data" / "horikawaCode" / "generate_tribev2_features.lock"
VALID_SOURCES = {
    "tribev2_subcortical_checkpoint",
    "tribev2_subcortical_checkpoint_regions",
}


def _npz_path(inter_dir: Path, sid: str) -> Path:
    p = inter_dir / f"{sid}_both.npz"
    if p.is_file():
        return p
    if sid.isdigit():
        alt = inter_dir / f"{int(sid):04d}_both.npz"
        if alt.is_file():
            return alt
    return p


def _classify_npz(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        with np.load(path, allow_pickle=True) as d:
            if "preds" not in d or "preds_subcortical" not in d:
                return "invalid"
            cort = d["preds"]
            sub = d["preds_subcortical"]
            if cort.ndim != 2 or sub.ndim != 2:
                return "invalid"
            if cort.shape[1] != 20484 or sub.shape[1] != 8802:
                return "invalid"
            if not np.all(np.isfinite(cort)) or not np.all(np.isfinite(sub)):
                return "invalid"
            src = str(np.asarray(d["prediction_source"])) if "prediction_source" in d else ""
            if src in VALID_SOURCES:
                return "ok"
            return "stale"
    except OSError:
        return "invalid"


def _load_manifest() -> list[dict]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return list(payload.get("clips") or [])


def _sort_key(sid: str) -> tuple:
    return (0, int(sid)) if str(sid).isdigit() else (1, str(sid))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    clips = _load_manifest()
    clips = [c for c in clips if c.get("video_path") or resolve_video_path(VIDEOS_DIR, str(c["stimulus_id"]))]
    clips_sorted = sorted(clips, key=lambda c: _sort_key(str(c["stimulus_id"])))

    kept: list[str] = []
    removed: list[dict] = []
    missing: list[str] = []

    # Remove lock if present (paused run)
    if LOCK.is_file() and not args.dry_run:
        try:
            LOCK.unlink()
        except OSError:
            pass

    for clip in clips_sorted:
        sid = str(clip["stimulus_id"])
        path = _npz_path(INTER_DIR, sid)
        status = _classify_npz(path)
        if status == "ok":
            kept.append(sid)
        elif status == "missing":
            missing.append(sid)
        else:
            if path.is_file() and not args.dry_run:
                path.unlink(missing_ok=True)
            removed.append({"stimulus_id": sid, "reason": status, "path": str(path)})

    # Drop orphan NPZs not in manifest (verify_*, etc. keep verify if valid?)
    manifest_ids = {str(c["stimulus_id"]) for c in clips_sorted}
    orphans_removed = 0
    for path in INTER_DIR.glob("*_both.npz"):
        sid = path.stem.replace("_both", "").lstrip("0") or "0"
        # normalize: also check raw stem
        raw_sid = path.stem.replace("_both", "")
        if raw_sid not in manifest_ids and sid not in manifest_ids:
            status = _classify_npz(path)
            if status != "ok":
                if not args.dry_run:
                    path.unlink(missing_ok=True)
                orphans_removed += 1

    next_sid = missing[0] if missing else None
    resume_index = len(kept)

    # Rebuild batch_report from kept only
    results = []
    for sid in kept:
        p = _npz_path(INTER_DIR, sid)
        results.append({"stimulus_id": sid, "status": "skipped", "path": str(p), "note": "checkpoint_ok"})
    for row in removed:
        results.append({**row, "status": "discarded"})
    for sid in missing:
        results.append({"stimulus_id": sid, "status": "pending"})

    report = {
        "manifest": str(MANIFEST),
        "output_dir": str(INTER_DIR),
        "checkpoint_at": datetime.now(timezone.utc).isoformat(),
        "ok": len(kept),
        "discarded": len(removed),
        "pending": len(missing),
        "results": results[:50],  # head only; full pending in checkpoint
        "errors": [],
    }
    if not args.dry_run:
        (INTER_DIR / "batch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    checkpoint = {
        "status": "paused",
        "phase": "modal_features",
        "corpus": "full_2181",
        "manifest": str(MANIFEST),
        "intermediates_dir": str(INTER_DIR),
        "expected_clips": len(clips_sorted),
        "ok_real": len(kept),
        "discarded": len(removed),
        "orphans_removed": orphans_removed,
        "remaining": len(missing),
        "pct_complete": round(100.0 * len(kept) / max(len(clips_sorted), 1), 2),
        "last_completed_stimulus_id": kept[-1] if kept else None,
        "next_stimulus_id": next_sid,
        "resume_index": resume_index,
        "paused_at": datetime.now(timezone.utc).isoformat(),
        "resume_command": (
            ".venv311\\Scripts\\python.exe scripts\\horikawaCode\\run_full_horikawa_training.py "
            "--skip-labels --resume"
        ),
    }
    if not args.dry_run:
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    print(json.dumps(checkpoint, indent=2))
    if args.dry_run:
        print(f"DRY RUN: would remove {len(removed)} stale/invalid, keep {len(kept)} ok")
    else:
        print(f"Wrote {CHECKPOINT}")


if __name__ == "__main__":
    main()
