#!/usr/bin/env python3
"""Export UX session viewer bundle under scout_data/sessions/<id>/ux_viewer/.

Usage:
    python scripts/export_ux_viewer.py --session-id <id>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from activation_store import SESSIONS_DIR
from scout_core.session_align import find_walkthrough_video, load_manifest

VIEWER_TEMPLATE = PROJECT_ROOT / "viewer" / "ux_session_viewer.html"


def _heatmap_to_png(npy_path: Path, png_path: Path) -> None:
    heatmap = np.load(npy_path).astype(np.float32)
    h, w = heatmap.shape
    flat = heatmap.ravel()
    lo, hi = float(flat.min()), float(flat.max())
    if hi <= lo:
        hi = lo + 1e-6
    norm = (heatmap - lo) / (hi - lo)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (norm * 255).astype(np.uint8)
    rgba[..., 3] = (norm * 200 + 20).clip(0, 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(rgba, mode="RGBA").save(png_path)
    except ImportError:
        rgb = rgba[..., :3]
        # Raw PPM fallback
        ppm = png_path.with_suffix(".ppm")
        with ppm.open("wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(rgb.tobytes())
        png_path = ppm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()

    session_dir = SESSIONS_DIR / args.session_id
    if not session_dir.is_dir():
        raise SystemExit(f"Session not found: {session_dir}")

    out_dir = session_dir / "ux_viewer"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = session_dir / "analysis_bundle.json"
    bundle: dict = {}
    if bundle_path.is_file():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    manifest = load_manifest(session_dir) or {}
    video = find_walkthrough_video(session_dir)
    video_name = None
    if video is not None:
        dest = out_dir / video.name
        if not dest.is_file():
            shutil.copy2(video, dest)
        video_name = video.name

    heatmaps_src = session_dir / "heatmaps"
    heatmap_index: dict[str, str] = {}
    if heatmaps_src.is_dir():
        for npy in sorted(heatmaps_src.glob("t_*.npy")):
            t = npy.stem.replace("t_", "")
            png = out_dir / "heatmaps" / f"t_{t}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            _heatmap_to_png(npy, png)
            heatmap_index[t] = f"heatmaps/t_{t}.png"

    sample_times: list[int] = []
    sections = []
    for sec in bundle.get("section_report") or []:
        sections.append({
            "section_id": sec.get("section_id"),
            "dwell_sec": sec.get("dwell_sec"),
            "flags": sec.get("flags"),
            "sample_t_indices": sec.get("sample_t_indices"),
            "top_elements": sec.get("top_elements"),
            "recommendations": sec.get("recommendations"),
        })
        for t in sec.get("sample_t_indices") or []:
            sample_times.append(int(t))

    events = []
    for ev in bundle.get("events") or []:
        g = ev.get("grounding")
        events.append({
            "t_spike": ev.get("t_spike"),
            "triggers": ev.get("triggers"),
            "grounding": g,
        })
        if ev.get("t_spike") is not None:
            sample_times.append(int(ev["t_spike"]))

    capture = manifest.get("capture") or {}
    viewer_bundle = {
        "session_id": args.session_id,
        "video": video_name,
        "capture": {
            "width": capture.get("width", 1920),
            "height": capture.get("height", 1080),
        },
        "n_timesteps": len(manifest.get("dom_snapshots") or []),
        "sample_t_indices": sorted(set(sample_times)),
        "sections": sections,
        "events": events,
        "heatmap_index": heatmap_index,
        "manifest_path": "../session_manifest.json",
        "analysis_bundle_path": "../analysis_bundle.json",
    }
    (out_dir / "viewer_bundle.json").write_text(
        json.dumps(viewer_bundle, indent=2), encoding="utf-8",
    )

    if VIEWER_TEMPLATE.is_file():
        shutil.copy2(VIEWER_TEMPLATE, out_dir / "index.html")

    print(f"UX viewer exported: {out_dir / 'index.html'}")
    print(f"Open: file://{out_dir.resolve() / 'index.html'}?base=.")


if __name__ == "__main__":
    main()
