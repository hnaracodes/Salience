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
from scout_core.heatmap_extract import read_heatmaps_manifest
from scout_core.session_align import find_walkthrough_video, load_manifest, validate_preds_manifest_alignment

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
    heatmap_provenance: dict[str, dict] = {}
    hm_manifest_entries = read_heatmaps_manifest(heatmaps_src) if heatmaps_src.is_dir() else {}
    if heatmaps_src.is_dir():
        for npy in sorted(heatmaps_src.glob("t_*.npy")):
            t_str = npy.stem.replace("t_", "")
            png = out_dir / "heatmaps" / f"t_{t_str}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            _heatmap_to_png(npy, png)
            heatmap_index[t_str] = f"heatmaps/t_{t_str}.png"
            hm_entry = hm_manifest_entries.get(int(t_str), {})
            heatmap_provenance[t_str] = {
                "source": hm_entry.get("source") or (
                    "uniform_placeholder" if hm_entry.get("placeholder") else "unknown"
                ),
                "placeholder": bool(hm_entry.get("placeholder", False)),
                "sha256": hm_entry.get("sha256"),
            }

    manifest_n_timesteps = len(manifest.get("dom_snapshots") or [])

    # Compute analysis timestep count from bundle if available (preds-derived).
    analysis_n_timesteps: int | None = None
    preds_path = session_dir / "preds.npz"
    if preds_path.is_file():
        try:
            import numpy as _np
            analysis_n_timesteps = int(_np.load(preds_path)["preds"].shape[0])
        except Exception:
            pass

    # Build alignment report for the viewer.
    alignment: dict = {}
    if analysis_n_timesteps is not None and manifest_n_timesteps > 0:
        alignment = validate_preds_manifest_alignment(analysis_n_timesteps, manifest)
    elif bundle.get("session_capture"):
        sc = bundle["session_capture"]
        alignment = {
            "ok": sc.get("alignment_ok"),
            "message": sc.get("alignment_message"),
        }

    sample_times: list[int] = []
    ms = bundle.get("marketing_scores") or {}
    ms_by_section = {s.get("section_id"): s for s in (ms.get("sections") or [])}
    act_ms_by_section = {
        s.get("section_id"): s for s in ((ms.get("activation_analysis") or {}).get("sections") or [])
    }
    sections = []
    for sec in bundle.get("section_report") or []:
        sid = sec.get("section_id")
        ms_row = ms_by_section.get(sid) or {}
        act_row = act_ms_by_section.get(sid) or {}
        act_sec = sec.get("activation") or {}
        sections.append({
            "section_id": sid,
            "dwell_sec": sec.get("dwell_sec"),
            "flags": sec.get("flags"),
            "sample_t_indices": sec.get("sample_t_indices"),
            "top_elements": sec.get("top_elements"),
            "recommendations": sec.get("recommendations"),
            "heatmap_stats": sec.get("heatmap_stats"),
            "marketing_score": ms_row.get("score"),
            "marketing_rank": ms_row.get("rank"),
            "marketing_label": ms_row.get("label"),
            "activation_mean_raw": act_sec.get("mean_raw"),
            "activation_mean_z": act_sec.get("mean_z"),
            "activation_score": act_row.get("score"),
            "activation_rank": act_row.get("rank"),
            "activation_label": act_row.get("label"),
        })
        for t in sec.get("sample_t_indices") or []:
            sample_times.append(int(t))

    # Only include sample_t_indices that are within manifest bounds (+1 tolerance)
    # so the viewer does not attempt to render DOM overlays for invalid timesteps.
    manifest_max_t = manifest_n_timesteps - 1
    sample_times_filtered = [t for t in sample_times if t <= manifest_max_t + 1]

    events = []
    for ev in bundle.get("events") or []:
        g = ev.get("grounding")
        t_spike = ev.get("t_spike")
        events.append({
            "t_spike": t_spike,
            "triggers": ev.get("triggers"),
            "grounding": g,
            "heatmap_provenance": ev.get("heatmap_provenance"),
            "in_manifest_bounds": t_spike is not None and int(t_spike) <= manifest_max_t + 1,
        })
        if t_spike is not None:
            sample_times.append(int(t_spike))

    capture = manifest.get("capture") or {}
    viewer_bundle = {
        "session_id": args.session_id,
        "video": video_name,
        "capture": {
            "width": capture.get("width", 1920),
            "height": capture.get("height", 1080),
        },
        "manifest_n_timesteps": manifest_n_timesteps,
        "analysis_n_timesteps": analysis_n_timesteps,
        # Legacy alias kept for backward compatibility with existing viewer HTML.
        "n_timesteps": manifest_n_timesteps,
        "alignment": alignment,
        "sample_t_indices": sorted(set(sample_times_filtered)),
        "sections": sections,
        "events": events,
        "heatmap_index": heatmap_index,
        "heatmap_provenance": heatmap_provenance,
        "manifest_path": "../session_manifest.json",
        "analysis_bundle_path": "../analysis_bundle.json",
        "marketing_scores": {
            "overall_score": ms.get("overall_score"),
            "display_curve": ms.get("display_curve"),
            "drop_moments": ms.get("drop_moments"),
            "focus_windows": ms.get("focus_windows"),
            "session_metrics": ms.get("session_metrics"),
            "disclaimer": ms.get("disclaimer"),
            "activation_analysis": ms.get("activation_analysis"),
        } if ms else None,
        "activation_track": bundle.get("activation_track"),
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
