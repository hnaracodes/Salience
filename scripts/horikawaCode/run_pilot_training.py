"""Orchestrate Horikawa pilot training with Modal budget guard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BUDGET_USD = 20.0
DEFAULT_A100_USD_PER_HR = float(os.environ.get("MODAL_A100_USD_PER_HR", "2.50"))
LOG_DIR = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run"


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


def _load_batch_report(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_seconds_from_report(report: dict) -> float:
    total = 0.0
    for row in report.get("results") or []:
        if row.get("status") == "ok" and row.get("elapsed_s"):
            total += float(row["elapsed_s"])
    return total


def _estimate_cost(gpu_seconds: float, rate_usd_per_hr: float) -> float:
    return gpu_seconds / 3600.0 * rate_usd_per_hr


def _max_clips_for_budget(
    probe_gpu_seconds: float,
    probe_n: int,
    budget_usd: float,
    rate_usd_per_hr: float,
    *,
    already_done: int = 0,
) -> int:
    if probe_n <= 0 or probe_gpu_seconds <= 0:
        return 0
    per_clip_s = probe_gpu_seconds / probe_n
    per_clip_usd = _estimate_cost(per_clip_s, rate_usd_per_hr)
    if per_clip_usd <= 0:
        return 0
    remaining_usd = max(0.0, budget_usd - _estimate_cost(probe_gpu_seconds, rate_usd_per_hr))
    extra_clips = int(remaining_usd // per_clip_usd)
    return already_done + extra_clips


def _count_videos(videos_dir: Path) -> int:
    if not videos_dir.is_dir():
        return 0
    return sum(1 for _ in videos_dir.glob("*.mp4"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--probe-n", type=int, default=3, help="Modal probe clips before scaling")
    parser.add_argument("--pilot-n", type=int, default=150)
    parser.add_argument("--a100-usd-per-hr", type=float, default=DEFAULT_A100_USD_PER_HR)
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "videos",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "pilot_150.json",
    )
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal",
    )
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-modal", action="store_true", help="Prepare/train only (no GPU spend)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "pilot_training.log"
    cost_path = LOG_DIR / "cost_estimate.json"

    _log("=== Horikawa pilot training start ===", log_path)
    _log(f"Budget cap: ${args.budget_usd:.2f}  A100 rate: ${args.a100_usd_per_hr:.2f}/hr", log_path)

    labels_cache = PROJECT_ROOT / "scout_data" / "horikawaCode" / "labels"
    ratings_npz = labels_cache / "ratings_cache.npz"

    if not args.skip_labels and not ratings_npz.is_file():
        _log("Phase 0: downloading figshare labels (no Modal cost)", log_path)
        _run_py("scripts/horikawaCode/download_horikawa_labels.py", [], log_path)
    else:
        _log(f"Phase 0: labels cache present ({ratings_npz})", log_path)

    _log("Phase 1: building pilot manifest", log_path)
    _run_py(
        "scripts/horikawaCode/build_clip_manifest.py",
        [
            "--pilot-n", str(args.pilot_n),
            "--corpus", "pilot_150",
            "--videos-dir", str(args.videos_dir),
            "--output", str(args.manifest),
        ],
        log_path,
    )

    audit_path = args.manifest.with_name(args.manifest.stem + "_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    n_with_video = int(audit.get("n_with_video") or 0)
    n_clips = int(audit.get("n_clips") or args.pilot_n)
    mp4_count = _count_videos(args.videos_dir)

    _log(f"Manifest: {n_clips} clips, {n_with_video} with resolved paths, {mp4_count} mp4 in videos dir", log_path)

    cost_state: dict = {
        "budget_usd": args.budget_usd,
        "a100_usd_per_hr": args.a100_usd_per_hr,
        "probe_n": args.probe_n,
        "pilot_n": args.pilot_n,
        "n_clips_manifest": n_clips,
        "n_videos_on_disk": mp4_count,
        "modal_gpu_seconds": 0.0,
        "estimated_modal_usd": 0.0,
        "phases_completed": ["labels", "manifest"],
    }

    if args.skip_modal or n_with_video == 0 or mp4_count == 0:
        reason = "skip_modal flag" if args.skip_modal else "no MP4s in videos dir"
        _log(f"STOP before Modal Phase 2 ({reason}). Zero GPU spend.", log_path)
        cost_state["stopped_reason"] = reason
        cost_state["next_step"] = (
            f"Place Horikawa MP4s in {args.videos_dir} then re-run this script."
        )
        cost_path.write_text(json.dumps(cost_state, indent=2), encoding="utf-8")
        _log(f"Cost tracker -> {cost_path}", log_path)
        return

    batch_report = args.intermediates_dir / "batch_report.json"

    _log(f"Phase 2a: Modal probe ({args.probe_n} clips)", log_path)
    _run_py(
        "scripts/horikawaCode/generate_tribev2_features.py",
        [
            "--manifest", str(args.manifest),
            "--videos-dir", str(args.videos_dir),
            "--output-dir", str(args.intermediates_dir),
            "--execute-tribe",
            "--skip-existing",
            "--limit", str(args.probe_n),
        ],
        log_path,
    )

    probe_report = _load_batch_report(batch_report)
    probe_gpu_s = _gpu_seconds_from_report(probe_report)
    probe_ok = sum(1 for r in probe_report.get("results", []) if r.get("status") == "ok")
    probe_cost = _estimate_cost(probe_gpu_s, args.a100_usd_per_hr)
    per_clip_s = probe_gpu_s / max(probe_ok, 1)
    projected_full = _estimate_cost(per_clip_s * n_clips, args.a100_usd_per_hr)

    cost_state.update({
        "probe_ok": probe_ok,
        "probe_gpu_seconds": round(probe_gpu_s, 2),
        "probe_estimated_usd": round(probe_cost, 4),
        "per_clip_gpu_seconds": round(per_clip_s, 2),
        "per_clip_estimated_usd": round(_estimate_cost(per_clip_s, args.a100_usd_per_hr), 4),
        "projected_full_pilot_usd": round(projected_full, 2),
    })

    _log(
        f"Probe: {probe_ok}/{args.probe_n} ok, {probe_gpu_s:.1f}s GPU, "
        f"~${probe_cost:.2f} spent, projected full pilot ~${projected_full:.2f}",
        log_path,
    )

    if projected_full > args.budget_usd:
        cap = _max_clips_for_budget(probe_gpu_s, max(probe_ok, 1), args.budget_usd, args.a100_usd_per_hr)
        _log(
            f"BUDGET GUARD: full {n_clips} clips projects ${projected_full:.2f} > ${args.budget_usd:.2f}. "
            f"Capping Modal run at {cap} clips total.",
            log_path,
        )
        clip_limit = cap
    else:
        clip_limit = n_clips
        _log(f"Within budget: proceeding with up to {clip_limit} clips", log_path)

    cost_state["clip_limit"] = clip_limit
    cost_path.write_text(json.dumps(cost_state, indent=2), encoding="utf-8")

    if clip_limit <= probe_ok:
        _log("Probe already hit clip limit; skipping Phase 2b scale-up", log_path)
    else:
        _log(f"Phase 2b: Modal scale-up (limit {clip_limit}, skip-existing)", log_path)
        _run_py(
            "scripts/horikawaCode/generate_tribev2_features.py",
            [
                "--manifest", str(args.manifest),
                "--videos-dir", str(args.videos_dir),
                "--output-dir", str(args.intermediates_dir),
                "--execute-tribe",
                "--skip-existing",
                "--limit", str(clip_limit),
            ],
            log_path,
        )

    final_report = _load_batch_report(batch_report)
    total_gpu_s = _gpu_seconds_from_report(final_report)
    total_modal_usd = _estimate_cost(total_gpu_s, args.a100_usd_per_hr)
    cost_state["modal_gpu_seconds"] = round(total_gpu_s, 2)
    cost_state["estimated_modal_usd"] = round(total_modal_usd, 4)
    cost_state["phases_completed"].append("modal_features")
    cost_path.write_text(json.dumps(cost_state, indent=2), encoding="utf-8")
    _log(f"Modal total: {total_gpu_s:.1f}s GPU ~${total_modal_usd:.2f}", log_path)

    if total_modal_usd > args.budget_usd * 1.05:
        _log(f"WARN: estimated Modal spend ${total_modal_usd:.2f} exceeded budget ${args.budget_usd:.2f}", log_path)

    _log("Phase 3: prepare train NPZs (local, no Modal)", log_path)
    _run_py(
        "scripts/horikawaCode/prepare_horikawa_tribev2.py",
        ["--manifest", str(args.manifest)],
        log_path,
    )
    _run_py(
        "scripts/horikawaCode/prepare_horikawa_tribev2.py",
        [
            "--config", "configs/horikawa_decoding_dims.yaml",
            "--manifest", str(args.manifest),
        ],
        log_path,
    )

    _log("Phase 4: train ridge decoders (local)", log_path)
    _run_py("scripts/horikawaCode/train_horikawa_ridge_decoder.py", [], log_path)
    _run_py(
        "scripts/horikawaCode/train_horikawa_ridge_decoder.py",
        ["--config", "configs/horikawa_decoding_dims.yaml"],
        log_path,
    )

    _log("Phase 5: evaluate (local)", log_path)
    _run_py("scripts/horikawaCode/evaluate_horikawa_decoder.py", [], log_path)
    _run_py(
        "scripts/horikawaCode/evaluate_horikawa_decoder.py",
        ["--config", "configs/horikawa_decoding_dims.yaml", "--model-id", "horikawa_ridge_dims_v1"],
        log_path,
    )

    cost_state["phases_completed"].extend(["prepare", "train", "eval"])
    cost_state["finished_at"] = datetime.now(timezone.utc).isoformat()
    cost_path.write_text(json.dumps(cost_state, indent=2), encoding="utf-8")
    _log(f"=== Pilot complete. Cost tracker -> {cost_path} ===", log_path)


if __name__ == "__main__":
    main()
