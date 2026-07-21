"""Validate engagement scores against behavioral proxies (dwell, scroll-stop, Clarity)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = PROJECT_ROOT / "configs" / "engagement_calibrated.yaml"


def load_calibration_config(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_CALIBRATION
    if not p.is_file():
        return {
            "calibration_id": "engagement_default_v0",
            "thresholds": {"spearman_min": 0.35, "legacy_vs_norms_delta_max": 2.0},
        }
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _engagement_trace(bundle: dict[str, Any]) -> list[float | None]:
    track = bundle.get("engagement_track") or {}
    return list(track.get("scores") or [])


def _dwell_proxy_from_manifest(manifest: dict[str, Any], n_tr: int) -> np.ndarray:
    """Proxy: seconds visible in viewport per TR from DOM snapshots."""
    dwell = np.zeros(max(n_tr, 1), dtype=np.float64)
    snaps = manifest.get("dom_snapshots") or []
    tr_sec = float((manifest.get("tr_mapping") or {}).get("tr_duration_sec") or 1.0)
    for snap in snaps:
        t = int(snap.get("t_idx", 0))
        if t >= n_tr:
            continue
        vis_sum = 0.0
        for el in snap.get("elements") or []:
            try:
                vis_sum += float(el.get("visibility_ratio") or 0.0)
            except (TypeError, ValueError):
                pass
        dwell[t] = vis_sum * tr_sec
    return dwell


def _scroll_stop_proxy(manifest: dict[str, Any], n_tr: int) -> np.ndarray:
    """Proxy: 1 when scroll position stable between consecutive snapshots."""
    snaps = sorted(manifest.get("dom_snapshots") or [], key=lambda s: int(s.get("t_idx", 0)))
    out = np.zeros(max(n_tr, 1), dtype=np.float64)
    prev_y: int | None = None
    for snap in snaps:
        t = int(snap.get("t_idx", 0))
        if t >= n_tr:
            continue
        y = int(snap.get("scrollY", 0))
        if prev_y is not None and y == prev_y:
            out[t] = 1.0
        prev_y = y
    return out


def spearman_rho(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return float("nan")
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def validate_session(
    session_dir: Path,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    bundle_path = session_dir / "analysis_bundle.json"
    manifest_path = session_dir / "session_manifest.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    eng = _engagement_trace(bundle)
    numeric_eng = [float(x) for x in eng if x is not None]
    if len(numeric_eng) < 3:
        return {
            "session_id": session_dir.name,
            "status": "skipped",
            "reason": "insufficient_engagement_scores",
        }

    n_tr = len(eng)
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dwell = _dwell_proxy_from_manifest(manifest, n_tr)
    scroll_stop = _scroll_stop_proxy(manifest, n_tr)

    eng_arr = np.array([float(x) if x is not None else np.nan for x in eng], dtype=np.float64)
    valid = np.isfinite(eng_arr)
    if valid.sum() < 3:
        return {"session_id": session_dir.name, "status": "skipped", "reason": "non_finite_engagement"}

    rho_dwell = spearman_rho(
        eng_arr[valid].tolist(),
        dwell[valid].tolist(),
    )
    rho_scroll = spearman_rho(
        eng_arr[valid].tolist(),
        scroll_stop[valid].tolist(),
    )

    legacy = bundle.get("engagement_track_legacy") or {}
    legacy_scores = legacy.get("scores") or []
    legacy_delta: float | None = None
    if legacy_scores and len(legacy_scores) == len(eng):
        la = np.array([float(x) if x is not None else np.nan for x in legacy_scores])
        both = valid & np.isfinite(la)
        if both.sum() >= 3:
            legacy_delta = float(np.mean(np.abs(eng_arr[both] - la[both])))

    thr = thresholds or {}
    passed = True
    if np.isfinite(rho_dwell) and rho_dwell < float(thr.get("spearman_min", 0.35)):
        passed = False
    max_delta = float(thr.get("legacy_vs_norms_delta_max", 2.0))
    if legacy_delta is not None and legacy_delta > max_delta:
        passed = False

    return {
        "session_id": session_dir.name,
        "status": "ok",
        "engagement_source": (bundle.get("engagement_track") or {}).get("source"),
        "spearman_dwell": round(rho_dwell, 4) if np.isfinite(rho_dwell) else None,
        "spearman_scroll_stop": round(rho_scroll, 4) if np.isfinite(rho_scroll) else None,
        "legacy_vs_norms_mean_abs_delta": round(legacy_delta, 4) if legacy_delta is not None else None,
        "passed": passed,
    }


def validate_corpus(validation_dir: Path) -> dict[str, Any]:
    manifest_path = validation_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "empty", "sessions": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = load_calibration_config()
    thr = cfg.get("thresholds") or {}
    results = []
    for entry in manifest.get("sessions") or []:
        sid = entry.get("session_id")
        if not sid:
            continue
        session_dir = PROJECT_ROOT / "scout_data" / "sessions" / sid
        results.append(validate_session(session_dir, thresholds=thr))
    passed = sum(1 for r in results if r.get("passed"))
    return {
        "calibration_id": cfg.get("calibration_id"),
        "validation_dir": str(validation_dir),
        "n_sessions": len(results),
        "n_passed": passed,
        "sessions": results,
    }
