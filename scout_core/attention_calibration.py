"""Calibrate attention proxy scores against behavioral ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scout_core.clarity_adapter import click_rows_by_selector, parse_clarity_click_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = PROJECT_ROOT / "configs" / "attribution_calibrated.yaml"


def load_calibration_config(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_CALIBRATION
    if not p.is_file():
        return {
            "calibration_id": "default_v0",
            "attention_weight": 0.68,
            "click_weight": 0.32,
            "thresholds": {"spearman_min": 0.4, "top1_cta_hit_min": 0.6},
        }
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def behavioral_score_from_clarity(
    elements: list[dict[str, Any]],
    clarity_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Map dom_id -> normalized behavioral score from Clarity click counts."""
    by_sel = click_rows_by_selector(clarity_rows)
    max_clicks = max((int(r.get("clicks") or 0) for r in clarity_rows), default=0)
    scores: dict[str, float] = {}
    for el in elements:
        dom_id = str(el.get("dom_id") or "").lower()
        if not dom_id:
            continue
        clicks = 0
        if dom_id in by_sel:
            clicks = int(by_sel[dom_id].get("clicks") or 0)
        else:
            for sel, row in by_sel.items():
                if sel in dom_id or dom_id in sel:
                    clicks = max(clicks, int(row.get("clicks") or 0))
        scores[dom_id] = float(clicks / max_clicks) if max_clicks > 0 else 0.0
    return scores


def model_scores_from_section(
    section: dict[str, Any],
    *,
    attention_weight: float | None = None,
) -> dict[str, float]:
    """Return element model scores, optionally re-fusing raw component scores."""
    out: dict[str, float] = {}
    for el in section.get("top_elements") or []:
        dom_id = str(el.get("dom_id") or "").lower()
        if dom_id:
            if attention_weight is None:
                score = float(el.get("combined_score") or 0.0)
            else:
                attention = float(el.get("attention_score") or 0.0)
                clickability = float(el.get("clickability") or 0.0)
                score = attention_weight * attention + (1.0 - attention_weight) * clickability
            out[dom_id] = float(np.clip(score / 100.0, 0.0, 1.0))
    return out


def spearman_rho(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return float("nan")
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values with average ranks for ties, matching Spearman semantics."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def top1_hit(model: dict[str, float], behavioral: dict[str, float]) -> bool:
    if not model or not behavioral:
        return False
    top_model = max(model, key=model.get)
    top_beh = max(behavioral, key=behavioral.get)
    return top_model == top_beh


def validate_session(
    session_dir: Path,
    *,
    clarity_csv: Path | None = None,
    thresholds: dict[str, float] | None = None,
    attention_weight: float | None = None,
) -> dict[str, Any]:
    bundle_path = session_dir / "analysis_bundle.json"
    if not bundle_path.is_file():
        return {
            "session_id": session_dir.name,
            "status": "invalid",
            "reason": "missing_analysis_bundle",
            "passed": False,
        }
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("clarity_attribution"):
        return {
            "session_id": session_dir.name,
            "status": "invalid",
            "reason": "clarity_leakage_in_analysis_bundle",
            "passed": False,
        }
    clarity_path = clarity_csv or session_dir / "clarity_clicks.csv"
    if not clarity_path.is_file():
        return {"session_id": session_dir.name, "status": "skipped", "reason": "no_clarity_csv"}

    clarity_rows = parse_clarity_click_csv(clarity_path)
    sections = bundle.get("section_report") or []
    rhos: list[float] = []
    hits = 0
    total = 0
    for sec in sections:
        elements = sec.get("top_elements") or []
        beh = behavioral_score_from_clarity(elements, clarity_rows)
        mod = model_scores_from_section(sec, attention_weight=attention_weight)
        keys = sorted(set(beh) & set(mod))
        if len(keys) < 2:
            continue
        rho = spearman_rho([beh[k] for k in keys], [mod[k] for k in keys])
        if np.isfinite(rho):
            rhos.append(rho)
        total += 1
        if top1_hit(mod, beh):
            hits += 1

    agg_rho = float(np.mean(rhos)) if rhos else float("nan")
    hit_rate = hits / total if total else 0.0
    thr = thresholds or {}
    passed = bool(
        np.isfinite(agg_rho)
        and agg_rho >= float(thr.get("spearman_min", 0.4))
        and hit_rate >= float(thr.get("top1_cta_hit_min", 0.6))
    )
    return {
        "session_id": session_dir.name,
        "status": "ok",
        "spearman_mean": round(agg_rho, 4) if np.isfinite(agg_rho) else None,
        "top1_hit_rate": round(hit_rate, 4),
        "n_sections": total,
        "attention_weight": attention_weight,
        "passed": passed,
    }


def _resolve_clarity_path(validation_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    return project_path if project_path.is_file() else validation_dir / path


def _bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = 20260611,
    n_bootstrap: int = 2000,
) -> list[float] | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(n_bootstrap, finite.size))
    means = finite[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [round(float(low), 4), round(float(high), 4)]


def _corpus_summary(
    results: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    qualifying = [row for row in results if row.get("status") == "ok"]
    rhos = [
        float(row["spearman_mean"])
        for row in qualifying
        if row.get("spearman_mean") is not None
    ]
    hit_rates = [float(row.get("top1_hit_rate") or 0.0) for row in qualifying]
    rho_mean = float(np.mean(rhos)) if rhos else float("nan")
    hit_mean = float(np.mean(hit_rates)) if hit_rates else 0.0
    passed = bool(
        qualifying
        and np.isfinite(rho_mean)
        and rho_mean >= float(thresholds.get("spearman_min", 0.4))
        and hit_mean >= float(thresholds.get("top1_cta_hit_min", 0.6))
    )
    return {
        "n_qualifying": len(qualifying),
        "spearman_macro_mean": round(rho_mean, 4) if np.isfinite(rho_mean) else None,
        "spearman_macro_ci95": _bootstrap_mean_ci(rhos),
        "top1_macro_mean": round(hit_mean, 4),
        "top1_macro_ci95": _bootstrap_mean_ci(hit_rates),
        "passed": passed,
    }


def validate_corpus(
    validation_dir: Path,
    *,
    roles: set[str] | None = None,
    attention_weight: float | None = None,
) -> dict[str, Any]:
    manifest_path = validation_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "empty", "sessions": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = load_calibration_config()
    thr = cfg.get("thresholds") or {}
    results = []
    for entry in manifest.get("sessions") or []:
        sid = entry.get("session_id")
        clarity = entry.get("clarity_csv")
        if not sid:
            continue
        role = str(entry.get("role") or "unspecified")
        if roles is not None and role not in roles:
            continue
        session_dir = PROJECT_ROOT / "scout_data" / "sessions" / sid
        clarity_path = _resolve_clarity_path(validation_dir, clarity)
        result = validate_session(
            session_dir,
            clarity_csv=clarity_path,
            thresholds=thr,
            attention_weight=attention_weight,
        )
        result.update({
            "role": role,
            "property_id": entry.get("property_id"),
            "page_id": entry.get("page_id"),
            "unique_sessions": entry.get("unique_sessions"),
            "mapped_clicks": entry.get("mapped_clicks"),
        })
        results.append(result)
    passed = sum(1 for r in results if r.get("passed"))
    return {
        "calibration_id": cfg.get("calibration_id"),
        "validation_dir": str(validation_dir),
        "n_sessions": len(results),
        "n_passed": passed,
        "roles": sorted(roles) if roles is not None else None,
        "attention_weight": attention_weight,
        "summary": _corpus_summary(results, thresholds=thr),
        "sessions": results,
    }


def fit_attention_weight(
    validation_dir: Path,
    *,
    roles: set[str] | None = None,
    candidates: list[float] | None = None,
) -> dict[str, Any]:
    """Select a fusion weight without touching holdout entries."""
    fit_roles = roles or {"train", "validation"}
    if "holdout" in fit_roles:
        raise ValueError("Holdout data cannot be used to fit attention weights")
    grid = candidates or [round(value, 2) for value in np.linspace(0.0, 1.0, 21)]
    rows = []
    for weight in grid:
        report = validate_corpus(
            validation_dir,
            roles=fit_roles,
            attention_weight=float(weight),
        )
        summary = report.get("summary") or {}
        rows.append({
            "attention_weight": float(weight),
            "spearman_macro_mean": summary.get("spearman_macro_mean"),
            "top1_macro_mean": summary.get("top1_macro_mean"),
            "n_qualifying": summary.get("n_qualifying"),
        })
    eligible = [row for row in rows if row.get("spearman_macro_mean") is not None]
    if not eligible:
        raise ValueError("No qualifying train/validation sessions for weight fitting")
    best = max(
        eligible,
        key=lambda row: (
            float(row["spearman_macro_mean"]),
            float(row.get("top1_macro_mean") or 0.0),
            -abs(float(row["attention_weight"]) - 0.5),
        ),
    )
    return {
        "fit_roles": sorted(fit_roles),
        "best": best,
        "candidates": rows,
    }
