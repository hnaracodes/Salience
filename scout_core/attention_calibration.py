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


def model_scores_from_section(section: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for el in section.get("top_elements") or []:
        dom_id = str(el.get("dom_id") or "").lower()
        if dom_id:
            out[dom_id] = float(el.get("combined_score") or 0.0) / 100.0
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
) -> dict[str, Any]:
    bundle_path = session_dir / "analysis_bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
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
        mod = model_scores_from_section(sec)
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
    passed = (
        (np.isnan(agg_rho) or agg_rho >= float(thr.get("spearman_min", 0.4)))
        and hit_rate >= float(thr.get("top1_cta_hit_min", 0.6))
    )
    return {
        "session_id": session_dir.name,
        "status": "ok",
        "spearman_mean": round(agg_rho, 4) if np.isfinite(agg_rho) else None,
        "top1_hit_rate": round(hit_rate, 4),
        "n_sections": total,
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
        clarity = entry.get("clarity_csv")
        if not sid:
            continue
        session_dir = PROJECT_ROOT / "scout_data" / "sessions" / sid
        clarity_path = Path(clarity) if clarity else None
        results.append(validate_session(session_dir, clarity_csv=clarity_path, thresholds=thr))
    passed = sum(1 for r in results if r.get("passed"))
    return {
        "calibration_id": cfg.get("calibration_id"),
        "validation_dir": str(validation_dir),
        "n_sessions": len(results),
        "n_passed": passed,
        "sessions": results,
    }
