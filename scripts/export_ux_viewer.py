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
from scout_core.dom_intersect import (
    filter_elements_to_section,
    find_nearest_snapshot,
    score_all_elements,
)
from scout_core.session_align import (
    find_walkthrough_video,
    load_manifest,
    pad_manifest_snapshots_to_video_duration,
    validate_preds_manifest_alignment,
)

VIEWER_TEMPLATE = PROJECT_ROOT / "viewer" / "ux_session_viewer.html"
BRAIN_SURFACE_JS = PROJECT_ROOT / "viewer" / "brain_surface.js"


def _build_element_tr_index(
    sections: list[dict],
    events: list[dict],
) -> dict[str, dict]:
    """Fast lookup: dom_id -> salient TRs and grounded spike TRs."""
    index: dict[str, dict] = {}

    def _ensure(dom_id: str) -> dict:
        if dom_id not in index:
            index[dom_id] = {"salient_trs": [], "grounded_spikes": []}
        return index[dom_id]

    for sec in sections:
        by_t = sec.get("element_scores_by_t") or {}
        for t_str, elements in by_t.items():
            try:
                t_idx = int(t_str)
            except (TypeError, ValueError):
                continue
            for el in elements or []:
                dom_id = el.get("dom_id")
                if not dom_id:
                    continue
                entry = _ensure(str(dom_id))
                density = el.get("attention_density")
                if density is None:
                    density = el.get("mean_attention_density")
                entry["salient_trs"].append({
                    "t": t_idx,
                    "density": density,
                    "section_id": sec.get("section_id"),
                })

    for ev in events:
        g = ev.get("grounding") or {}
        dom_id = g.get("dom_id")
        t_spike = ev.get("t_spike")
        if dom_id and t_spike is not None:
            entry = _ensure(str(dom_id))
            t_int = int(t_spike)
            if t_int not in entry["grounded_spikes"]:
                entry["grounded_spikes"].append(t_int)

    for dom_id, entry in index.items():
        entry["salient_trs"].sort(
            key=lambda row: (row.get("density") is None, -(float(row.get("density") or 0))),
        )
        entry["grounded_spikes"] = sorted(entry["grounded_spikes"])
    return index


def _build_element_scores_index(sections: list[dict]) -> dict[str, dict]:
    """Per dom_id rollup of attribution scores for click-to-inspect in the viewer."""
    index: dict[str, dict] = {}

    def _merge(dom_id: str, el: dict, *, section_id: str | None, t_idx: int | None) -> None:
        row = index.setdefault(
            dom_id,
            {
                "dom_id": dom_id,
                "tag": el.get("tag"),
                "section_ids": [],
                "by_t": {},
                "rollup": {},
            },
        )
        if el.get("tag") and not row.get("tag"):
            row["tag"] = el.get("tag")
        if section_id and section_id not in row["section_ids"]:
            row["section_ids"].append(section_id)

        score_row = {
            "combined_score": el.get("combined_score"),
            "attention_score": el.get("attention_score"),
            "clickability": el.get("clickability"),
            "attention_density": el.get("attention_density") or el.get("mean_attention_density"),
            "attribution_flags": el.get("attribution_flags") or [],
            "clarity_matched": el.get("clarity_matched"),
            "n_samples": el.get("n_samples"),
            "section_id": section_id,
        }
        if t_idx is not None:
            row["by_t"][str(t_idx)] = score_row

        combined = float(el.get("combined_score") or 0.0)
        best = float(row["rollup"].get("combined_score") or 0.0)
        if combined >= best:
            row["rollup"] = {**score_row, "source": f"TR {t_idx}" if t_idx is not None else "section rollup"}

    for sec in sections:
        sid = sec.get("section_id")
        for el in sec.get("top_elements") or []:
            dom_id = el.get("dom_id")
            if dom_id:
                _merge(str(dom_id), el, section_id=sid, t_idx=None)
        for t_str, elements in (sec.get("element_scores_by_t") or {}).items():
            try:
                t_idx = int(t_str)
            except (TypeError, ValueError):
                continue
            for el in elements or []:
                dom_id = el.get("dom_id")
                if dom_id:
                    _merge(str(dom_id), el, section_id=sid, t_idx=t_idx)

    return index


def _export_brain_assets(session_id: str, session_dir: Path, out_dir: Path, manifest: dict) -> dict | None:
    """Copy brain viewer binaries and optional PlotBrain MP4 into ux_viewer/."""
    if not (session_dir / "preds.npz").is_file():
        return None
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from export_brain_viewer import export_session_viewer

    brain_src = export_session_viewer(session_id)
    brain_dest = out_dir / "brain"
    brain_dest.mkdir(parents=True, exist_ok=True)
    for name in ("coords.bin", "faces.bin", "preds.bin", "manifest.json"):
        src = brain_src / name
        if src.is_file():
            shutil.copy2(src, brain_dest / name)
    manifest_path = brain_dest / "manifest.json"
    if not manifest_path.is_file():
        return None
    brain_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mp4_rel: str | None = None
    brain_mp4_src = session_dir / "brain_results.mp4"
    if brain_mp4_src.is_file():
        mp4_dest = out_dir / "brain_sim.mp4"
        shutil.copy2(brain_mp4_src, mp4_dest)
        mp4_rel = "brain_sim.mp4"

    tr_map = manifest.get("tr_mapping") or {}
    fps = float(tr_map.get("tr_duration_sec") or manifest.get("interval_sec") or 1.0)

    out: dict = {
        "manifest": "brain/manifest.json",
        "preds_layout": brain_manifest.get("preds_layout", "row_major_timestep_vertex"),
        "n_timesteps": brain_manifest.get("n_timesteps"),
        "n_vertices": brain_manifest.get("n_vertices"),
        "fps": fps,
    }
    if mp4_rel:
        out["mp4"] = mp4_rel
    return out


def _warm_heatmap_rgba(norm: np.ndarray) -> np.ndarray:
    """Map normalized attention [0,1] to yellow -> orange -> red RGBA."""
    t = np.clip(norm.astype(np.float32), 0.0, 1.0)
    h, w = t.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # yellow (220,200,80) -> orange -> red (255,40,30)
    rgba[..., 0] = (220 + 35 * t).astype(np.uint8)
    rgba[..., 1] = (40 + 160 * (1 - t)).astype(np.uint8)
    rgba[..., 2] = (30 + 50 * (1 - t)).astype(np.uint8)
    rgba[..., 3] = (t * 200 + 25).clip(0, 255).astype(np.uint8)
    return rgba


def _heatmap_to_png(npy_path: Path, png_path: Path) -> None:
    heatmap = np.load(npy_path).astype(np.float32)
    h, w = heatmap.shape
    flat = heatmap.ravel()
    lo, hi = float(flat.min()), float(flat.max())
    if hi <= lo:
        hi = lo + 1e-6
    norm = (heatmap - lo) / (hi - lo)
    rgba = _warm_heatmap_rgba(norm)
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


def _element_scores_by_t(
    session_dir: Path,
    manifest: dict,
    sec: dict,
    heatmaps_dir: Path,
    *,
    top_k: int = 10,
) -> dict[str, list[dict]]:
    """Per-sample TR element scores for timestep-aware sidebar."""
    by_t: dict[str, list[dict]] = {}
    root_bbox = sec.get("section_root_bbox")
    for t in sec.get("sample_t_indices") or []:
        npy = heatmaps_dir / f"t_{t}.npy"
        if not npy.is_file():
            continue
        heatmap = np.load(npy).astype(np.float32)
        snap = find_nearest_snapshot(manifest, int(t))
        if snap is None:
            continue
        elements = filter_elements_to_section(snap.get("elements") or [], root_bbox)
        scored = score_all_elements(
            heatmap,
            elements,
            int(snap.get("scrollY", 0)),
            int(snap.get("scrollX", 0)),
        )
        rollup_by_id = {
            el.get("dom_id"): el
            for el in (sec.get("top_elements") or [])
            if el.get("dom_id")
        }
        for row in scored:
            rollup = rollup_by_id.get(row.get("dom_id"))
            if not rollup:
                continue
            for key in (
                "attention_score",
                "clickability",
                "combined_score",
                "engagement_attributed",
                "activation_attributed",
                "attribution_flags",
            ):
                if key in rollup:
                    row[key] = rollup[key]
        by_t[str(t)] = scored[:top_k]
    return by_t


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
    preds_path = session_dir / "preds.npz"
    target_tr_count: int | None = None
    if preds_path.is_file():
        try:
            target_tr_count = int(np.load(preds_path)["preds"].shape[0])
        except Exception:
            target_tr_count = None
    if manifest.get("dom_snapshots"):
        manifest = pad_manifest_snapshots_to_video_duration(
            manifest,
            session_dir,
            target_tr_count=target_tr_count,
        )
    video = find_walkthrough_video(session_dir)
    video_name = None
    if video is not None:
        dest = out_dir / video.name
        if not dest.is_file():
            shutil.copy2(video, dest)
        video_name = video.name

    manifest_dest_name = "session_manifest.json"
    if manifest.get("dom_snapshots"):
        (out_dir / manifest_dest_name).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
    else:
        manifest_src = session_dir / "session_manifest.json"
        if manifest_src.is_file():
            shutil.copy2(manifest_src, out_dir / manifest_dest_name)

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

    frames_src = session_dir / "frames"
    frame_index: dict[str, str] = {}
    if frames_src.is_dir():
        frames_dest = out_dir / "frames"
        frames_dest.mkdir(parents=True, exist_ok=True)
        for jpg in sorted(frames_src.glob("t_*.jpg")):
            t_str = jpg.stem.replace("t_", "")
            dest = frames_dest / jpg.name
            if not dest.is_file() or jpg.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(jpg, dest)
            frame_index[t_str] = f"frames/{jpg.name}"

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
            "element_scores_by_t": _element_scores_by_t(
                session_dir, manifest, sec, heatmaps_src,
            ) if heatmaps_src.is_dir() else {},
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
            "element_attribution": sec.get("element_attribution"),
        })
        for t in sec.get("sample_t_indices") or []:
            sample_times.append(int(t))

    # Only include timesteps within manifest bounds (+1 tolerance).
    manifest_max_t = manifest_n_timesteps - 1

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

    sample_times_filtered = [t for t in sample_times if t <= manifest_max_t + 1]

    snapshot_scroll_by_t: dict[str, int] = {}
    for snap in manifest.get("dom_snapshots") or []:
        if "t_idx" in snap:
            snapshot_scroll_by_t[str(int(snap["t_idx"]))] = int(snap.get("scrollY", 0))

    capture = manifest.get("capture") or {}
    brain_viewer = _export_brain_assets(args.session_id, session_dir, out_dir, manifest)
    element_tr_index = _build_element_tr_index(sections, events)
    element_scores_index = _build_element_scores_index(sections)

    marketing_narrative = bundle.get("marketing_narrative")
    element_insight_index: dict[str, dict] = {}
    if marketing_narrative:
        for ins in marketing_narrative.get("element_insights") or []:
            dom_id = ins.get("dom_id")
            if dom_id:
                element_insight_index[str(dom_id)] = ins

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
        "element_tr_index": element_tr_index,
        "element_scores_index": element_scores_index,
        "engagement_track": bundle.get("engagement_track"),
        "emotion_track": bundle.get("emotion_track"),
        "brain_viewer": brain_viewer,
        "frame_index": frame_index,
        "snapshot_scroll_by_t": snapshot_scroll_by_t,
        "scroll_plan": manifest.get("scroll_plan"),
        "heatmap_index": heatmap_index,
        "heatmap_provenance": heatmap_provenance,
        "manifest_path": manifest_dest_name if manifest.get("dom_snapshots") else "../session_manifest.json",
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
        "marketing_narrative": marketing_narrative,
        "element_insight_index": element_insight_index,
        "interaction_events": manifest.get("interaction_events") or [],
    }
    copy_signals_path = session_dir / "copy_signals.json"
    if copy_signals_path.is_file():
        viewer_bundle["copy_signals"] = json.loads(copy_signals_path.read_text(encoding="utf-8"))

    (out_dir / "viewer_bundle.json").write_text(
        json.dumps(viewer_bundle, indent=2), encoding="utf-8",
    )

    if VIEWER_TEMPLATE.is_file():
        shutil.copy2(VIEWER_TEMPLATE, out_dir / "index.html")
    if BRAIN_SURFACE_JS.is_file():
        shutil.copy2(BRAIN_SURFACE_JS, out_dir / "brain_surface.js")

    print(f"UX viewer exported: {out_dir / 'index.html'}")
    print(f"Open: file://{out_dir.resolve() / 'index.html'}?base=.")


if __name__ == "__main__":
    main()
