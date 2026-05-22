#!/usr/bin/env python3
"""Extract sparse ViT attention heatmaps for section-sampled timesteps.

Decodes video frames at sampled ``t_idx`` values and writes
``scout_data/sessions/<id>/heatmaps/t_<N>.npy``.

Usage:
    python scripts/extract_section_heatmaps.py --session-id <id>
    python scripts/extract_section_heatmaps.py --session-id <id> --uniform-heatmap
    python scripts/extract_section_heatmaps.py --session-id <id> --modal
    python scripts/extract_section_heatmaps.py --session-id <id> --dry-run
    python scripts/extract_section_heatmaps.py --session-id <id> --refresh-sections
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from activation_store import SESSIONS_DIR
from scout_core.heatmap_extract import (
    capture_size_from_manifest,
    extract_heatmap_modal,
    file_sha256,
    fps_from_manifest,
    write_heatmaps_manifest,
)
from scout_core.section_pipeline import load_section_config, run_section_analytics
from scout_core.section_sampling import attach_sample_timesteps
from scout_core.session_align import find_walkthrough_video

_SECTION_CFG = PROJECT_ROOT / "configs" / "section_analytics.yaml"


def _extract_frame_ffmpeg(video: Path, t_idx: int, fps: float, out_jpg: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    sec = t_idx / max(fps, 1e-6)
    cmd = [
        "ffmpeg", "-y", "-ss", str(sec), "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(out_jpg),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return out_jpg.is_file()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _write_uniform_heatmap(path: Path, h: int, w: int) -> None:
    grid = np.linspace(0.1, 1.0, w * h, dtype=np.float32).reshape(h, w)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, grid)


def _sample_timesteps_from_bundle(
    session_dir: Path,
    explicit: list[int] | None,
) -> list[int]:
    if explicit:
        return sorted(set(explicit))
    bundle_path = session_dir / "analysis_bundle.json"
    if not bundle_path.is_file():
        raise SystemExit("analysis_bundle.json missing — run run_dual_track.py first.")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("section_report"):
        seen: set[int] = set()
        for sec in bundle["section_report"]:
            for t in sec.get("sample_t_indices") or []:
                seen.add(int(t))
        return sorted(seen)
    cfg = load_section_config(_SECTION_CFG)
    report = run_section_analytics(
        session_dir, bundle, attach_heatmaps=False,
    )
    return attach_sample_timesteps(
        report,
        bundle,
        max_per_section=int(cfg.get("max_samples_per_section", 3)),
        max_total=int(cfg.get("max_heatmaps_per_session", 20)),
    )


def _refresh_section_report(session_dir: Path) -> None:
    bundle_path = session_dir / "analysis_bundle.json"
    if not bundle_path.is_file():
        raise SystemExit("analysis_bundle.json missing — cannot refresh sections.")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = run_section_analytics(session_dir, bundle, attach_heatmaps=True)
    bundle["section_report"] = report
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Refreshed section_report ({len(report)} sections) in {bundle_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--timesteps", default=None, help="Comma-separated t_idx list (override sampling)")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (default from manifest tr_mapping)")
    parser.add_argument("--uniform-heatmap", action="store_true", help="Write placeholder heatmaps (no GPU)")
    parser.add_argument("--modal", action="store_true", help="Run Modal extract_frame_attention (default if not uniform)")
    parser.add_argument("--dry-run", action="store_true", help="Extract frames only; do not write heatmaps")
    parser.add_argument("--refresh-sections", action="store_true", help="Re-merge top_elements into analysis_bundle")
    parser.add_argument("--capture-height", type=int, default=None)
    parser.add_argument("--capture-width", type=int, default=None)
    args = parser.parse_args()

    session_dir = SESSIONS_DIR / args.session_id
    if not session_dir.is_dir():
        raise SystemExit(f"Session not found: {session_dir}")

    if args.refresh_sections:
        _refresh_section_report(session_dir)
        if args.dry_run and not args.modal and not args.uniform_heatmap:
            return

    explicit = None
    if args.timesteps:
        explicit = [int(x.strip()) for x in args.timesteps.split(",") if x.strip()]

    t_list = _sample_timesteps_from_bundle(session_dir, explicit)
    if not t_list:
        print("No timesteps to sample.")
        return

    h, w = capture_size_from_manifest(session_dir)
    if args.capture_height is not None:
        h = args.capture_height
    if args.capture_width is not None:
        w = args.capture_width
    fps = args.fps if args.fps is not None else fps_from_manifest(session_dir)

    use_modal = args.modal and not args.uniform_heatmap and not args.dry_run

    heatmaps_dir = session_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    video = find_walkthrough_video(session_dir)
    hm_entries: list[dict] = []

    for t in t_list:
        out_npy = heatmaps_dir / f"t_{t}.npy"
        jpg = frames_dir / f"t_{t}.jpg"

        if video is not None and not jpg.is_file():
            ok = _extract_frame_ffmpeg(video, t, fps, jpg)
            if ok:
                print(f"  t={t}: frame → {jpg.name}")
            else:
                print(f"  t={t}: ffmpeg frame extract failed")

        if args.dry_run:
            continue

        if out_npy.is_file():
            print(f"  t={t}: heatmap exists, skip")
            hm_entries.append({
                "t_idx": t,
                "frame_path": str(jpg.relative_to(session_dir)) if jpg.is_file() else None,
                "heatmap_path": out_npy.name,
            })
            continue

        if args.uniform_heatmap:
            _write_uniform_heatmap(out_npy, h, w)
            print(f"  t={t}: uniform placeholder → {out_npy.name}")
            hm_entries.append({"t_idx": t, "heatmap_path": out_npy.name, "placeholder": True})
            continue

        if not jpg.is_file():
            print(f"  t={t}: no frame — add video or use --uniform-heatmap")
            continue

        if use_modal:
            try:
                heatmap = extract_heatmap_modal(jpg, h, w)
                np.save(out_npy, heatmap)
                print(f"  t={t}: Modal heatmap → {out_npy.name}")
                hm_entries.append({
                    "t_idx": t,
                    "frame_path": str(jpg.relative_to(session_dir)),
                    "heatmap_path": out_npy.name,
                    "sha256": file_sha256(out_npy),
                })
            except Exception as exc:
                print(f"  t={t}: Modal failed ({exc}) — use --uniform-heatmap for offline tests")
        else:
            print(f"  t={t}: frame ready; pass --modal to run extract_frame_attention")

    if hm_entries and not args.dry_run:
        write_heatmaps_manifest(heatmaps_dir, hm_entries)

    if args.refresh_sections or (use_modal and hm_entries):
        _refresh_section_report(session_dir)

    print(f"\nSampled {len(t_list)} timestep(s). Heatmaps dir: {heatmaps_dir}")


if __name__ == "__main__":
    main()
