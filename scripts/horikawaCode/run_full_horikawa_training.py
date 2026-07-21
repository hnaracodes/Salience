"""Full Horikawa pipeline: Modal features for entire labeled corpus, then ridge train/eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run"
DEFAULT_MANIFEST = PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "full_2181.json"
DEFAULT_INTER = PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal"
DEFAULT_VIDEOS = PROJECT_ROOT / "scout_data" / "horikawaCode" / "videos"


def _log(msg: str, log_path: Path) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run_py(script: str, args: list[str], log_path: Path) -> None:
    py = sys.executable
    cmd = [py, str(PROJECT_ROOT / script), *args]
    _log(f"RUN {' '.join(cmd)}", log_path)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _manifest_clips_with_video(manifest_path: Path) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = payload.get("clips") or []
    return [c for c in clips if c.get("video_path")]


def _count_intermediates(inter_dir: Path, clips: list[dict]) -> dict[str, int]:
    import numpy as np

    ok = 0
    proxy_or_stale = 0
    missing = 0
    for clip in clips:
        sid = str(clip["stimulus_id"])
        p = inter_dir / f"{sid}_both.npz"
        if not p.is_file():
            missing += 1
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                src = str(np.asarray(d["prediction_source"])) if "prediction_source" in d else ""
            if src in ("tribev2_subcortical_checkpoint", "tribev2_subcortical_checkpoint_regions"):
                ok += 1
            else:
                proxy_or_stale += 1
        except OSError:
            proxy_or_stale += 1
    return {"ok_real": ok, "stale_or_proxy": proxy_or_stale, "missing": missing, "expected": len(clips)}


def _write_ready_manifest(manifest_path: Path, inter_dir: Path, out_path: Path) -> int:
    clips = _manifest_clips_with_video(manifest_path)
    ready: list[dict] = []
    for clip in clips:
        sid = str(clip["stimulus_id"])
        p = inter_dir / f"{sid}_both.npz"
        if p.is_file():
            ready.append(clip)
    payload = {
        "corpus": json.loads(manifest_path.read_text(encoding="utf-8")).get("corpus", "full_2181"),
        "clips": ready,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--intermediates-dir", type=Path, default=DEFAULT_INTER)
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-modal", action="store_true", help="Prepare/train only")
    parser.add_argument("--skip-train", action="store_true", help="Modal features only")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip manifest rebuild; continue Modal with --skip-existing",
    )
    args = parser.parse_args()

    log_path = LOG_DIR / "full_training.log"
    state_path = LOG_DIR / "full_training_state.json"
    checkpoint_path = LOG_DIR / "full_training_checkpoint.json"

    if args.resume:
        _log("=== Horikawa FULL corpus training RESUME ===", log_path)
    else:
        _log("=== Horikawa FULL corpus training start ===", log_path)

    labels_cache = PROJECT_ROOT / "scout_data" / "horikawaCode" / "labels"
    if not args.skip_labels and not (labels_cache / "ratings_cache.npz").is_file():
        _log("Phase 0: download labels", log_path)
        _run_py("scripts/horikawaCode/download_horikawa_labels.py", [], log_path)

    if not args.resume:
        _log("Phase 1: build full manifest", log_path)
        _run_py(
            "scripts/horikawaCode/build_clip_manifest.py",
            [
                "--corpus", "full_2181",
                "--videos-dir", str(args.videos_dir),
                "--output", str(args.manifest),
            ],
            log_path,
        )
    elif checkpoint_path.is_file():
        ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        _log(
            f"Resume checkpoint: {ck.get('ok_real')}/{ck.get('expected_clips')} ok, "
            f"next={ck.get('next_stimulus_id')}",
            log_path,
        )

    audit_path = args.manifest.with_name(args.manifest.stem + "_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    clips_with_video = _manifest_clips_with_video(args.manifest)
    _log(
        f"Manifest: {audit.get('n_clips')} labeled, {audit.get('n_with_video')} with MP4",
        log_path,
    )

    if not args.skip_modal:
        counts_before = _count_intermediates(args.intermediates_dir, clips_with_video)
        _log(f"Intermediates before Modal: {counts_before}", log_path)

        _log("Phase 2: Modal deploy", log_path)
        subprocess.run(
            [str(PROJECT_ROOT / ".venv311" / "Scripts" / "modal.exe"), "deploy", "tribe.py"],
            cwd=PROJECT_ROOT,
            check=True,
        )

        _log(f"Phase 2b: Modal features for ALL {len(clips_with_video)} clips (sequential, single worker)", log_path)
        _run_py(
            "scripts/horikawaCode/generate_tribev2_features.py",
            [
                "--manifest", str(args.manifest),
                "--videos-dir", str(args.videos_dir),
                "--output-dir", str(args.intermediates_dir),
                "--execute-tribe",
                "--skip-existing",
            ],
            log_path,
        )

        counts_after = _count_intermediates(args.intermediates_dir, clips_with_video)
        _log(f"Intermediates after Modal: {counts_after}", log_path)

        if counts_after["missing"] > 0 or counts_after["stale_or_proxy"] > 0:
            msg = (
                f"Modal phase incomplete: missing={counts_after['missing']} "
                f"stale={counts_after['stale_or_proxy']} "
                f"ok={counts_after['ok_real']}/{counts_after['expected']}"
            )
            _log(f"PAUSE: {msg} — re-run with --resume after consolidate", log_path)
            _run_py(
                "scripts/horikawaCode/checkpoint_horikawa_progress.py",
                ["--status", "paused", "--note", msg],
                log_path,
            )
            state_path.write_text(
                json.dumps({"phase": "modal_paused", **counts_after}, indent=2),
                encoding="utf-8",
            )
            sys.exit(2)

    if args.skip_train:
        _log("--skip-train: Modal complete, exiting before ridge training", log_path)
        return

    ready_manifest = args.manifest.with_name("full_ready.json")
    n_ready = _write_ready_manifest(args.manifest, args.intermediates_dir, ready_manifest)
    _log(f"Phase 3: prepare from {n_ready} ready clips -> {ready_manifest}", log_path)

    if n_ready < len(clips_with_video):
        _log(
            f"STOP: only {n_ready}/{len(clips_with_video)} clips have intermediates; cannot train",
            log_path,
        )
        sys.exit(1)

    _run_py(
        "scripts/horikawaCode/prepare_horikawa_tribev2.py",
        ["--manifest", str(ready_manifest)],
        log_path,
    )
    _run_py(
        "scripts/horikawaCode/prepare_horikawa_tribev2.py",
        ["--config", "configs/horikawa_decoding_dims.yaml", "--manifest", str(ready_manifest)],
        log_path,
    )

    _log("Phase 4: train ridge decoders", log_path)
    _run_py("scripts/horikawaCode/train_horikawa_ridge_decoder.py", [], log_path)
    _run_py(
        "scripts/horikawaCode/train_horikawa_ridge_decoder.py",
        ["--config", "configs/horikawa_decoding_dims.yaml"],
        log_path,
    )

    _log("Phase 5: evaluate", log_path)
    _run_py("scripts/horikawaCode/evaluate_horikawa_decoder.py", [], log_path)
    _run_py(
        "scripts/horikawaCode/evaluate_horikawa_decoder.py",
        ["--config", "configs/horikawa_decoding_dims.yaml", "--model-id", "horikawa_ridge_dims_v1"],
        log_path,
    )

    state_path.write_text(
        json.dumps(
            {
                "phase": "complete",
                "n_clips_trained": n_ready,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log("=== Horikawa FULL corpus training COMPLETE ===", log_path)


if __name__ == "__main__":
    main()
