"""Write Horikawa full-corpus checkpoint for pause/resume."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run" / "full_training_checkpoint.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "full_2181.json"
DEFAULT_INTER = PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal"


def count_progress(manifest_path: Path, inter_dir: Path) -> dict:
    import numpy as np

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = [c for c in (payload.get("clips") or []) if c.get("video_path")]
    ok_real = 0
    stale_or_proxy = 0
    missing = 0
    ok_ids: list[str] = []
    pending_ids: list[str] = []

    for clip in clips:
        sid = str(clip["stimulus_id"])
        p = inter_dir / f"{sid}_both.npz"
        if not p.is_file():
            missing += 1
            pending_ids.append(sid)
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                src = str(np.asarray(d["prediction_source"])) if "prediction_source" in d else ""
            if src in ("tribev2_subcortical_checkpoint", "tribev2_subcortical_checkpoint_regions"):
                ok_real += 1
                ok_ids.append(sid)
            else:
                stale_or_proxy += 1
                pending_ids.append(sid)
        except OSError:
            stale_or_proxy += 1
            pending_ids.append(sid)

    expected = len(clips)
    return {
        "corpus": payload.get("corpus", "full_2181"),
        "manifest": str(manifest_path),
        "intermediates_dir": str(inter_dir),
        "expected_clips": expected,
        "ok_real": ok_real,
        "stale_or_proxy": stale_or_proxy,
        "missing": missing,
        "remaining": stale_or_proxy + missing,
        "pct_complete": round(100.0 * ok_real / max(expected, 1), 2),
        "phase": "modal_features" if ok_real < expected else "modal_complete",
        "resume_command": (
            f".venv311\\Scripts\\python.exe scripts/horikawaCode/run_full_horikawa_training.py "
            f"--skip-labels --resume"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--intermediates-dir", type=Path, default=DEFAULT_INTER)
    parser.add_argument("--status", choices=("running", "paused"), default="paused")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    state = count_progress(args.manifest, args.intermediates_dir)
    state.update({
        "status": args.status,
        "paused_at": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
    })
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2))
    print(f"Wrote {CHECKPOINT}")


if __name__ == "__main__":
    main()
