"""Validate alignment between preds.npz and session_manifest.json."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def validate_preds_manifest_alignment(
    n_preds: int,
    manifest: dict[str, Any],
    *,
    tolerance: int = 1,
) -> dict[str, Any]:
    """Return alignment report: ok, n_preds, n_snapshots, delta, message.

    Extended fields:
    - ``overlap_start``: first valid analysis timestep index (always 0).
    - ``overlap_end``: last timestep index covered by both preds and snapshots.
    - ``out_of_range_count``: analysis timesteps beyond manifest coverage.
    - ``mapping``: "exact" | "tolerated" | "clamped" | "invalid".
    """
    snapshots = manifest.get("dom_snapshots") or []
    n_snap = len(snapshots)
    delta = abs(n_preds - n_snap)
    ok = delta <= tolerance

    overlap_end = min(n_preds, n_snap) - 1
    out_of_range = max(0, n_preds - n_snap)

    if delta == 0:
        mapping = "exact"
    elif ok:
        mapping = "tolerated"
    elif n_snap > 0:
        mapping = "clamped"
    else:
        mapping = "invalid"

    msg = (
        f"preds T={n_preds} vs manifest snapshots={n_snap} "
        f"(delta={delta}, tolerance={tolerance}, mapping={mapping})"
    )
    if not ok:
        msg += " — WARNING: misaligned; check tr_duration_sec and walkthrough duration."
    return {
        "ok": ok,
        "n_preds": n_preds,
        "n_snapshots": n_snap,
        "delta": delta,
        "tolerance": tolerance,
        "overlap_start": 0,
        "overlap_end": max(overlap_end, -1),
        "out_of_range_count": out_of_range,
        "mapping": mapping,
        "message": msg,
    }


def is_timestep_in_manifest(t: int, manifest: dict[str, Any], *, tolerance: int = 0) -> bool:
    """Return True if ``t`` maps to a valid snapshot index in the manifest."""
    n_snap = len(manifest.get("dom_snapshots") or [])
    return t <= (n_snap - 1 + tolerance)


def find_walkthrough_video(session_dir: Path) -> Path | None:
    """Return walkthrough video path from manifest or session dir glob."""
    manifest_path = session_dir / "session_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rel = (manifest.get("video") or {}).get("path")
            if rel:
                candidate = session_dir / rel
                if candidate.is_file():
                    return candidate
        except json.JSONDecodeError:
            pass
    for pattern in ("walkthrough.mp4", "walkthrough.webm", "*.mp4", "*.webm"):
        hits = sorted(session_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def load_manifest(session_dir: Path) -> dict[str, Any] | None:
    path = session_dir / "session_manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def validate_session_dir(session_dir: Path, *, tolerance: int = 1) -> dict[str, Any]:
    """Validate preds.npz vs session_manifest when both exist."""
    preds_path = session_dir / "preds.npz"
    manifest = load_manifest(session_dir)
    if not preds_path.is_file() or manifest is None:
        return {"ok": True, "skipped": True, "message": "preds or manifest missing"}
    n_preds = int(np.load(preds_path)["preds"].shape[0])
    report = validate_preds_manifest_alignment(n_preds, manifest, tolerance=tolerance)
    report["skipped"] = False

    sub_path = session_dir / "preds_subcortical.npz"
    if sub_path.is_file():
        from scout_core.subcortical.io import validate_subcortical_alignment

        sub_preds = np.load(sub_path)["preds"]
        sub_report = validate_subcortical_alignment(n_preds, sub_preds)
        report["subcortical"] = sub_report
        if not sub_report["ok"]:
            report["ok"] = False
            report["message"] += f"; {sub_report['message']}"

    return report


def tr_duration_from_manifest(manifest: dict[str, Any]) -> float:
    """TR duration in seconds from manifest v1/v2 capture block."""
    capture = manifest.get("capture") or {}
    if capture.get("tr_duration_sec") is not None:
        return float(capture["tr_duration_sec"])
    fps = float(capture.get("fps", 1.0))
    return 1.0 / fps if fps > 0 else 1.0


def probe_video_duration_sec(video_path: Path) -> float | None:
    """Best-effort walkthrough video duration in seconds."""
    if not video_path.is_file():
        return None
    try:
        import imageio.v3 as iio

        with iio.imopen(video_path, "r", plugin="ffmpeg") as reader:
            meta = reader.metadata()
            if meta and meta.get("duration") is not None:
                return float(meta["duration"])
    except Exception:
        pass
    try:
        import subprocess

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except Exception:
        pass
    return None


def expected_tr_count_from_duration(duration_sec: float, interval_sec: float) -> int:
    return max(1, int(math.ceil(float(duration_sec) / float(interval_sec))))


def pad_manifest_snapshots_to_video_duration(
    manifest: dict[str, Any],
    session_dir: Path,
    *,
    tolerance: int = 1,
    target_tr_count: int | None = None,
) -> dict[str, Any]:
    """Forward-fill DOM snapshots so manifest TR count matches walkthrough video length."""
    snapshots = list(manifest.get("dom_snapshots") or [])
    if not snapshots:
        return manifest

    interval_sec = tr_duration_from_manifest(manifest)
    video_rel = (manifest.get("video") or {}).get("path")
    video_path = session_dir / video_rel if video_rel else find_walkthrough_video(session_dir)
    duration_sec = probe_video_duration_sec(video_path) if video_path else None
    if duration_sec is None:
        duration_sec = float((manifest.get("video") or {}).get("duration_sec") or 0)

    n_expected = target_tr_count
    if n_expected is None and duration_sec > 0:
        n_expected = expected_tr_count_from_duration(duration_sec, interval_sec)
    if n_expected is None:
        return manifest
    n_original = len(snapshots)
    if n_original >= n_expected - tolerance:
        manifest.setdefault("video", {})["duration_sec"] = round(duration_sec, 3)
        return manifest

    last = dict(snapshots[-1])
    for t_idx in range(n_original, n_expected):
        padded = dict(last)
        padded["t_idx"] = t_idx
        padded["pts_sec"] = round(t_idx * interval_sec, 3)
        padded["synthetic_pad"] = True
        snapshots.append(padded)

    manifest["dom_snapshots"] = snapshots
    manifest.setdefault("video", {})["duration_sec"] = round(duration_sec, 3)
    manifest.setdefault("alignment", {})["snapshot_padding"] = {
        "n_original": n_original,
        "n_padded": len(snapshots),
        "video_duration_sec": round(duration_sec, 3),
    }
    return manifest
