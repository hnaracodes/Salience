"""Marketing-friendly 0–100 scores on top of dual-track engagement and preds novelty.

Read-only transform: does not mutate engagement_track or emotion_track.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scout_core.dual_track import compute_mean_activation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "marketing_scores.yaml"

METRIC_KEYS = ("opening", "sustain", "transition", "stability", "density")


def load_marketing_config(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_CONFIG
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_preds_trace(session_dir: Path) -> np.ndarray | None:
    preds_path = session_dir / "preds.npz"
    if not preds_path.is_file():
        return None
    data = np.load(preds_path)
    preds = np.asarray(data["preds"], dtype=np.float32)
    if preds.ndim != 2 or preds.shape[0] < 1:
        return None
    return preds


def compute_novelty(preds: np.ndarray) -> np.ndarray:
    """L2 norm of consecutive pred vectors; novelty[0] = 0."""
    preds = np.asarray(preds, dtype=np.float32)
    if preds.shape[0] < 2:
        return np.zeros(preds.shape[0], dtype=np.float32)
    diff = np.diff(preds, axis=0)
    out = np.zeros(preds.shape[0], dtype=np.float32)
    out[1:] = np.linalg.norm(diff, axis=1).astype(np.float32)
    return out


def normalize_minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high - low < 1e-6:
        return np.full_like(arr, 50.0, dtype=np.float64)
    return 100.0 * (arr - low) / (high - low)


def _clip_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def score_from_ratio(raw: float, center: float, spread: float) -> int:
    return _clip_score(50.0 + 28.0 * ((raw - center) / max(spread, 1e-6)))


def compound_signal(
    engagement: np.ndarray,
    novelty: np.ndarray,
    *,
    engagement_weight: float = 0.7,
    novelty_weight: float = 0.3,
) -> np.ndarray:
    eng_norm = normalize_minmax(engagement)
    nov_norm = normalize_minmax(novelty)
    combined = engagement_weight * eng_norm + novelty_weight * nov_norm
    return np.clip(combined, 0.0, 100.0).astype(np.float64)


def valid_tr_indices(
    n_trs: int,
    tr_duration_sec: float,
    edge_cfg: dict[str, Any],
) -> list[int]:
    if n_trs <= 0:
        return []
    skip_first = int(edge_cfg.get("skip_first_trs", 1))
    skip_last = int(edge_cfg.get("skip_last_trs", 1))
    tail_buffer_sec = float(edge_cfg.get("tail_buffer_sec", 5.0))
    min_dur = float(edge_cfg.get("min_duration_for_tail_sec", 8.0))

    duration_sec = n_trs * tr_duration_sec
    tail_trs = 0
    if duration_sec > min_dur and tr_duration_sec > 0:
        tail_trs = int(np.ceil(tail_buffer_sec / tr_duration_sec))

    lower = skip_first
    upper = n_trs - skip_last
    if tail_trs > 0:
        upper = min(upper, n_trs - tail_trs)

    if lower >= upper:
        return list(range(n_trs))

    return list(range(lower, upper))


def _session_z(values: np.ndarray, indices: list[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if not indices:
        return np.zeros_like(arr)
    subset = arr[indices]
    mu = float(np.mean(subset))
    sigma = float(np.std(subset))
    if sigma < 1e-8:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def _metric_summary(key: str, score: int) -> str:
    bucket = "high" if score >= 75 else "mid" if score >= 60 else "low"
    library = {
        "opening": {
            "high": "Opening holds attention better than the rest of the session.",
            "mid": "Opening is near the session average.",
            "low": "Opening is weaker than later parts of the walkthrough.",
        },
        "sustain": {
            "high": "Engagement holds through the later sections.",
            "mid": "Late-session engagement is mixed.",
            "low": "Engagement fades in the second half.",
        },
        "transition": {
            "high": "Plenty of visual or neural change between steps.",
            "mid": "Moderate pace of change across the session.",
            "low": "Few strong transitions — the page may feel static.",
        },
        "stability": {
            "high": "Signal is smooth without erratic jumps.",
            "mid": "Mostly stable with occasional shifts.",
            "low": "Signal is jittery — check for confusing layout shifts.",
        },
        "density": {
            "high": "Solid average engagement relative to session peaks.",
            "mid": "Average engagement is moderate.",
            "low": "Mostly quiet except for isolated spikes.",
        },
    }
    return library.get(key, {}).get(bucket, "")


def compute_session_metrics(
    engagement: np.ndarray,
    novelty: np.ndarray,
    valid_indices: list[int],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    centers = cfg.get("metric_centers") or {}
    spreads = cfg.get("metric_spreads") or {}
    labels = cfg.get("metric_labels") or {}

    if not valid_indices:
        valid_indices = list(range(len(engagement)))

    eng = np.asarray(engagement, dtype=np.float64)
    nov = np.asarray(novelty, dtype=np.float64)
    valid = sorted(valid_indices)
    eng_v = eng[valid]
    nov_v = nov[valid]

    metrics: list[dict[str, Any]] = []

    def _append(key: str, raw: float | None, reason: str | None = None) -> None:
        if raw is None:
            metrics.append({
                "key": key,
                "label": labels.get(key, key.replace("_", " ").title()),
                "score": None,
                "raw_value": None,
                "summary": reason or "Not enough scored timesteps.",
            })
            return
        score = score_from_ratio(
            float(raw),
            float(centers.get(key, 1.0)),
            float(spreads.get(key, 0.2)),
        )
        metrics.append({
            "key": key,
            "label": labels.get(key, key.replace("_", " ").title()),
            "score": score,
            "raw_value": round(float(raw), 4),
            "summary": _metric_summary(key, score),
        })

    # opening: first 25% vs rest
    split = max(1, len(valid) // 4)
    early = eng_v[:split]
    rest = eng_v[split:]
    if len(rest) >= 1 and len(early) >= 1:
        opening_raw = (float(np.mean(early)) + 1e-6) / (float(np.mean(rest)) + 1e-6)
        _append("opening", opening_raw)
    else:
        _append("opening", None)

    # sustain: last third vs first third
    third = max(1, len(valid) // 3)
    if len(eng_v) >= third * 2:
        sustain_raw = (float(np.mean(eng_v[-third:])) + 1e-6) / (float(np.mean(eng_v[:third])) + 1e-6)
        _append("sustain", sustain_raw)
    else:
        _append("sustain", None)

    # transition: fraction novelty_z > 0.6
    if len(nov_v) >= 3:
        nov_z = (nov_v - float(np.mean(nov_v))) / (float(np.std(nov_v)) + 1e-8)
        transition_raw = float(np.sum(nov_z > 0.6) / len(nov_z))
        _append("transition", transition_raw)
    else:
        _append("transition", None)

    # stability
    if len(nov_v) >= 2 and float(np.mean(nov_v)) > 1e-8:
        stability_raw = 1.0 / (1.0 + float(np.std(nov_v)) / (float(np.mean(nov_v)) + 1e-8))
        _append("stability", stability_raw)
    else:
        _append("stability", None)

    # density
    if len(eng_v) >= 1:
        p90 = float(np.percentile(eng_v, 90))
        density_raw = float(np.mean(eng_v) / (p90 + 1e-8))
        _append("density", density_raw)
    else:
        _append("density", None)

    return metrics


def find_drop_moments(
    engagement: np.ndarray,
    novelty: np.ndarray,
    display_scores: np.ndarray,
    valid_indices: list[int],
    tr_duration_sec: float,
    drop_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    eng_z = _session_z(engagement, valid_indices)
    nov_z = _session_z(novelty, valid_indices)
    eng_thr = float(drop_cfg.get("engagement_z", -0.65))
    nov_thr = float(drop_cfg.get("novelty_z", -0.2))
    max_markers = int(drop_cfg.get("max_markers", 6))
    valid_set = set(valid_indices)

    drops: list[dict[str, Any]] = []
    for t in range(1, len(engagement)):
        if t not in valid_set:
            continue
        if eng_z[t] >= eng_thr or nov_z[t] >= nov_thr:
            continue
        drops.append({
            "t_idx": t,
            "seconds": round(t * tr_duration_sec, 2),
            "display_score": round(float(display_scores[t]), 1),
            "engagement_z": round(float(engagement[t]), 4),
            "novelty": round(float(novelty[t]), 4),
            "reason": "Engagement and visual change both dip relative to this session.",
        })
        if len(drops) >= max_markers:
            break
    return drops


def _pick_windows(
    display_scores: np.ndarray,
    valid_indices: list[int],
    *,
    prefer: str,
    count: int = 2,
) -> list[dict[str, Any]]:
    if not valid_indices:
        return []
    window_size = max(3, min(8, len(display_scores) // 8 or 3))
    scored: list[tuple[float, int, int, int]] = []
    for center in valid_indices:
        half = window_size // 2
        start = max(0, center - half)
        end = min(len(display_scores) - 1, center + half)
        score = float(np.mean(display_scores[start : end + 1]))
        scored.append((score, center, start, end))
    scored.sort(key=lambda item: item[0], reverse=(prefer == "high"))

    selected: list[dict[str, Any]] = []
    for score, center, start, end in scored:
        overlaps = any(abs(center - item["center_index"]) <= window_size for item in selected)
        if overlaps:
            continue
        selected.append({
            "center_index": center,
            "score": round(score, 1),
            "t_idx": center,
            "start_t_idx": start,
            "end_t_idx": end,
        })
        if len(selected) >= count:
            break
    return selected


def pick_focus_windows(
    display_scores: np.ndarray,
    valid_indices: list[int],
    tr_duration_sec: float,
) -> list[dict[str, Any]]:
    strong = _pick_windows(display_scores, valid_indices, prefer="high", count=1)
    weak = _pick_windows(display_scores, valid_indices, prefer="low", count=1)
    out: list[dict[str, Any]] = []
    for kind, windows in (("strong", strong), ("weak", weak)):
        for w in windows:
            t = int(w["t_idx"])
            out.append({
                "kind": kind,
                "t_idx": t,
                "seconds": round(t * tr_duration_sec, 2),
                "display_score": round(float(display_scores[t]), 1),
                "window_score": w["score"],
            })
    return out


def score_sections(
    section_report: list[dict[str, Any]],
    engagement: np.ndarray,
    display_scores: np.ndarray,
) -> list[dict[str, Any]]:
    if not section_report:
        return []

    section_means: list[tuple[str, float | None]] = []
    for sec in section_report:
        sid = str(sec.get("section_id", ""))
        eng = sec.get("engagement") or {}
        mean_z = eng.get("mean")
        if mean_z is not None:
            section_means.append((sid, float(mean_z)))
        else:
            t_indices = sec.get("t_indices") or []
            vals = [float(engagement[t]) for t in t_indices if 0 <= t < len(engagement)]
            section_means.append((sid, float(np.mean(vals)) if vals else None))

    numeric = [m for _, m in section_means if m is not None]
    if numeric:
        lo, hi = float(min(numeric)), float(max(numeric))
    else:
        lo, hi = 0.0, 1.0

    def _to_display(z: float | None) -> int | None:
        if z is None:
            return None
        if hi - lo < 1e-6:
            return 50
        return _clip_score(100.0 * (z - lo) / (hi - lo))

    scored: list[dict[str, Any]] = []
    for sec, (_, mean_z) in zip(section_report, section_means):
        sid = str(sec.get("section_id", ""))
        score = _to_display(mean_z)
        # Also average display_score over section TRs when available
        t_indices = sec.get("t_indices") or []
        display_mean = None
        if t_indices:
            ds = [float(display_scores[t]) for t in t_indices if 0 <= t < len(display_scores)]
            if ds:
                display_mean = round(float(np.mean(ds)), 1)
        scored.append({
            "section_id": sid,
            "score": score if score is not None else display_mean,
            "display_score_mean": display_mean,
            "engagement_z_mean": round(mean_z, 4) if mean_z is not None else None,
            "dwell_sec": sec.get("dwell_sec"),
            "rank": None,
            "label": None,
        })

    with_score = [s for s in scored if s.get("score") is not None]
    with_score.sort(key=lambda item: float(item["score"]), reverse=True)
    for rank, item in enumerate(with_score, start=1):
        item["rank"] = rank
        if rank == 1 and len(with_score) > 1:
            item["label"] = "strongest"
        elif rank == len(with_score) and len(with_score) > 1:
            item["label"] = "weakest"

    return scored


def score_sections_activation(
    section_report: list[dict[str, Any]],
    raw_activation: np.ndarray,
    display_scores: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-section 0–100 scores from mean |preds| aggregated per section."""
    if not section_report:
        return []

    section_means: list[tuple[str, float | None]] = []
    for sec in section_report:
        sid = str(sec.get("section_id", ""))
        act = sec.get("activation") or {}
        mean_raw = act.get("mean_raw")
        if mean_raw is not None:
            section_means.append((sid, float(mean_raw)))
        else:
            t_indices = sec.get("t_indices") or []
            vals = [float(raw_activation[t]) for t in t_indices if 0 <= t < len(raw_activation)]
            section_means.append((sid, float(np.mean(vals)) if vals else None))

    numeric = [m for _, m in section_means if m is not None]
    lo, hi = (float(min(numeric)), float(max(numeric))) if numeric else (0.0, 1.0)

    def _to_display(raw: float | None) -> int | None:
        if raw is None:
            return None
        if hi - lo < 1e-9:
            return 50
        return _clip_score(100.0 * (raw - lo) / (hi - lo))

    scored: list[dict[str, Any]] = []
    for sec, (_, mean_raw) in zip(section_report, section_means):
        sid = str(sec.get("section_id", ""))
        score = _to_display(mean_raw)
        t_indices = sec.get("t_indices") or []
        display_mean = None
        if t_indices:
            ds = [float(display_scores[t]) for t in t_indices if 0 <= t < len(display_scores)]
            if ds:
                display_mean = round(float(np.mean(ds)), 1)
        act = sec.get("activation") or {}
        scored.append({
            "section_id": sid,
            "score": score if score is not None else display_mean,
            "display_score_mean": display_mean,
            "mean_raw": round(mean_raw, 6) if mean_raw is not None else None,
            "mean_z": act.get("mean_z"),
            "rank": None,
            "label": None,
        })

    with_score = [s for s in scored if s.get("score") is not None]
    with_score.sort(key=lambda item: float(item["score"]), reverse=True)
    for rank, item in enumerate(with_score, start=1):
        item["rank"] = rank
        if rank == 1 and len(with_score) > 1:
            item["label"] = "strongest"
        elif rank == len(with_score) and len(with_score) > 1:
            item["label"] = "weakest"

    return scored


def _activation_from_bundle(
    bundle: dict[str, Any],
    preds: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (raw mean_abs, comparison Z, metadata) for Track 3."""
    at = bundle.get("activation_track") or {}
    meta = {
        "reducer": at.get("reducer", "mean_abs"),
        "source": at.get("source", "preds"),
        "comparison_mode": at.get("comparison_mode"),
        "baseline_flag": at.get("baseline_flag"),
    }
    raw_list = at.get("raw_scores")
    if raw_list:
        raw = np.asarray(raw_list, dtype=np.float64)
        z = np.asarray(at.get("scores") or at.get("session_z") or raw, dtype=np.float64)
        return raw, z, meta
    if preds is not None:
        raw = compute_mean_activation(preds).astype(np.float64)
        mu, sigma = float(raw.mean()), float(raw.std())
        z = ((raw - mu) / (sigma + 1e-8)).astype(np.float64)
        meta["comparison_mode"] = "session_relative"
        meta["source"] = "preds_mean_abs_computed"
        return raw, z, meta
    return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), meta


def build_activation_analysis(
    bundle: dict[str, Any],
    section_report: list[dict[str, Any]],
    *,
    tr_duration_sec: float,
    preds: np.ndarray | None,
) -> dict[str, Any] | None:
    """Track 3 analysis: mean |preds[t]| over full brain, session 0–100 display curve."""
    raw, z_scores, meta = _activation_from_bundle(bundle, preds)
    if raw.size == 0:
        return None

    display = normalize_minmax(raw)
    sections = score_sections_activation(section_report, raw, display)
    points = [
        {
            "t_idx": t,
            "seconds": round(t * tr_duration_sec, 2),
            "mean_activation": round(float(raw[t]), 6),
            "baseline_z": round(float(z_scores[t]), 4) if t < len(z_scores) else None,
            "display_score": round(float(display[t]), 1),
        }
        for t in range(len(raw))
    ]
    return {
        "schema_version": 1,
        "reducer": meta.get("reducer", "mean_abs"),
        "source": "preds_full_brain",
        "comparison_mode": meta.get("comparison_mode"),
        "baseline_flag": meta.get("baseline_flag"),
        "display_curve": {
            "tr_duration_sec": tr_duration_sec,
            "points": points,
            "avg_score": round(float(np.mean(display)), 1),
            "max_score": round(float(np.max(display)), 1),
            "min_score": round(float(np.min(display)), 1),
        },
        "session_summary": {
            "mean_raw": round(float(np.mean(raw)), 6),
            "max_raw": round(float(np.max(raw)), 6),
            "min_raw": round(float(np.min(raw)), 6),
            "mean_baseline_z": round(float(np.mean(z_scores)), 4),
        },
        "sections": sections,
        "disclaimer": (
            "Mean absolute predicted activation across all fsaverage5 vertices per TR; "
            "not eye-tracking. display_score is session-relative 0–100."
        ),
    }


def _resolve_tr_duration(bundle: dict[str, Any], session_dir: Path) -> float:
    sc = bundle.get("session_capture") or {}
    if sc.get("tr_duration_sec"):
        return float(sc["tr_duration_sec"])
    try:
        from scout_core.session_align import load_manifest, tr_duration_from_manifest

        manifest = load_manifest(session_dir)
        if manifest:
            return float(tr_duration_from_manifest(manifest))
    except Exception:
        pass
    return 1.0


def _engagement_array(bundle: dict[str, Any], preds: np.ndarray | None) -> tuple[np.ndarray, str, str | None]:
    """Return (activation trace for compound curve, source label, baseline_flag)."""
    engagement = bundle.get("engagement_track") or {}
    baseline_flag = engagement.get("baseline_flag")
    raw_scores = engagement.get("scores") or []
    n = preds.shape[0] if preds is not None else max(len(raw_scores), 1)

    numeric = [s for s in raw_scores if s is not None]
    if len(numeric) == len(raw_scores) and len(numeric) == n and n > 0:
        return np.asarray(raw_scores, dtype=np.float64), "engagement_track.scores", baseline_flag

    activation = bundle.get("activation_track") or {}
    act_raw = activation.get("raw_scores") or []
    if act_raw and len(act_raw) == n:
        return (
            np.asarray(act_raw, dtype=np.float64),
            "activation_track.raw_scores",
            activation.get("baseline_flag"),
        )

    if preds is not None:
        activation_vals = np.mean(np.abs(preds), axis=1).astype(np.float64)
        source = "preds_mean_abs_fallback"
        if baseline_flag or activation.get("baseline_flag"):
            source = "preds_mean_abs_fallback_no_baseline"
        return activation_vals, source, baseline_flag or activation.get("baseline_flag")

    if numeric:
        filled = np.zeros(n, dtype=np.float64)
        for i, s in enumerate(raw_scores[:n]):
            filled[i] = float(s) if s is not None else 0.0
        return filled, "engagement_track.partial", baseline_flag

    return np.zeros(n, dtype=np.float64), "unavailable", baseline_flag


def build_marketing_scores(
    session_dir: Path,
    bundle: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    cfg = config if config is not None else load_marketing_config(config_path)
    tr_duration_sec = _resolve_tr_duration(bundle, session_dir)
    preds = load_preds_trace(session_dir)
    engagement, activation_source, baseline_flag = _engagement_array(bundle, preds)

    n = len(engagement)
    if preds is not None and preds.shape[0] != n:
        n = min(n, preds.shape[0])
        engagement = engagement[:n]
        preds = preds[:n]

    if preds is not None:
        novelty = compute_novelty(preds)
        novelty_source = "preds_l2_diff"
    else:
        novelty = np.zeros(n, dtype=np.float64)
        novelty_source = "unavailable"

    weights = cfg.get("compound_weights") or {}
    display = compound_signal(
        engagement,
        novelty,
        engagement_weight=float(weights.get("engagement", 0.7)),
        novelty_weight=float(weights.get("novelty", 0.3)),
    )

    edge_cfg = cfg.get("edge_mask") or {}
    valid = valid_tr_indices(n, tr_duration_sec, edge_cfg)

    session_metrics = compute_session_metrics(engagement, novelty, valid, cfg)
    scored_values = [m["score"] for m in session_metrics if m.get("score") is not None]
    overall_score = int(round(float(np.mean(scored_values)))) if scored_values else None

    drop_cfg = cfg.get("drop_thresholds") or {}
    drop_moments = find_drop_moments(
        engagement, novelty, display, valid, tr_duration_sec, drop_cfg,
    )
    focus_windows = pick_focus_windows(display, valid, tr_duration_sec)

    section_report = bundle.get("section_report") or []
    sections = score_sections(section_report, engagement, display)
    activation_analysis = build_activation_analysis(
        bundle, section_report, tr_duration_sec=tr_duration_sec, preds=preds,
    )

    points = [
        {
            "t_idx": t,
            "seconds": round(t * tr_duration_sec, 2),
            "engagement_z": round(float(engagement[t]), 4),
            "novelty": round(float(novelty[t]), 4),
            "display_score": round(float(display[t]), 1),
        }
        for t in range(n)
    ]

    return {
        "schema_version": 1,
        "provenance": {
            "activation_source": activation_source,
            "novelty_source": novelty_source,
            "display_curve": "minmax_session_compound",
            "baseline_flag": baseline_flag,
            "compound_weights": {
                "engagement": float(weights.get("engagement", 0.7)),
                "novelty": float(weights.get("novelty", 0.3)),
            },
        },
        "display_curve": {
            "tr_duration_sec": tr_duration_sec,
            "points": points,
            "avg_score": round(float(np.mean(display)), 1) if n else 0.0,
            "max_score": round(float(np.max(display)), 1) if n else 0.0,
            "min_score": round(float(np.min(display)), 1) if n else 0.0,
        },
        "session_metrics": session_metrics,
        "overall_score": overall_score,
        "drop_moments": drop_moments,
        "focus_windows": focus_windows,
        "sections": sections,
        "activation_analysis": activation_analysis,
        "disclaimer": cfg.get(
            "disclaimer",
            "Session-relative display scores; not eye-tracking or guaranteed conversion.",
        ),
    }
