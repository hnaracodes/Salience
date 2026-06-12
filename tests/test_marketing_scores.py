"""Tests for marketing_scores display layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scout_core.marketing_scores import (
    build_activation_analysis,
    build_marketing_scores,
    compute_comparison_score,
    compute_novelty,
    compute_session_metrics,
    find_drop_moments,
    map_population_quantile,
    normalize_minmax,
    score_sections,
    score_sections_activation,
    valid_tr_indices,
)


def test_normalize_minmax_flat_is_fifty():
    arr = np.ones(5)
    out = normalize_minmax(arr)
    assert np.allclose(out, 50.0)


def test_novelty_starts_at_zero():
    preds = np.random.randn(6, 10).astype(np.float32)
    nov = compute_novelty(preds)
    assert nov[0] == 0.0
    assert nov.shape[0] == 6


def test_rising_engagement_sustain_beats_opening():
    """Linear ramp (all positive): late engagement exceeds early → sustain > opening."""
    eng = np.linspace(0.2, 1.8, 12)
    nov = np.abs(np.random.randn(12)) * 0.1 + 0.05
    valid = list(range(12))
    cfg = {
        "metric_centers": {
            "opening": 1.05,
            "sustain": 0.95,
            "transition": 0.22,
            "stability": 0.58,
            "density": 0.72,
        },
        "metric_spreads": {
            "opening": 0.35,
            "sustain": 0.30,
            "transition": 0.16,
            "stability": 0.20,
            "density": 0.18,
        },
        "metric_labels": {},
    }
    metrics = compute_session_metrics(eng, nov, valid, cfg)
    by_key = {m["key"]: m for m in metrics}
    assert by_key["sustain"]["score"] > by_key["opening"]["score"]


def test_strong_opening_beats_sustain():
    """High early engagement that fades (all positive) → opening > sustain."""
    eng = np.linspace(1.8, 0.2, 12)
    nov = np.abs(np.random.randn(12)) * 0.1 + 0.05
    valid = list(range(12))
    cfg = {
        "metric_centers": {
            "opening": 1.05,
            "sustain": 0.95,
            "transition": 0.22,
            "stability": 0.58,
            "density": 0.72,
        },
        "metric_spreads": {
            "opening": 0.35,
            "sustain": 0.30,
            "transition": 0.16,
            "stability": 0.20,
            "density": 0.18,
        },
        "metric_labels": {},
    }
    metrics = compute_session_metrics(eng, nov, valid, cfg)
    by_key = {m["key"]: m for m in metrics}
    assert by_key["opening"]["score"] > by_key["sustain"]["score"]


def test_drop_moments_respect_edge_mask():
    n = 10
    eng = np.zeros(n)
    nov = np.zeros(n)
    eng[4] = -2.0
    nov[4] = -2.0
    display = np.full(n, 50.0)
    display[4] = 20.0
    valid = valid_tr_indices(n, 1.0, {"skip_first_trs": 1, "skip_last_trs": 1, "tail_buffer_sec": 5.0})
    drops = find_drop_moments(
        eng, nov, display, valid, 1.0,
        {"engagement_z": -0.65, "novelty_z": -0.2, "max_markers": 6},
    )
    assert any(d["t_idx"] == 4 for d in drops)
    assert 0 not in {d["t_idx"] for d in drops}
    assert 9 not in {d["t_idx"] for d in drops}


def test_baseline_fallback_provenance(tmp_path: Path):
    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    preds = np.random.randn(5, 8).astype(np.float32)
    np.savez(session_dir / "preds.npz", preds=preds)
    bundle = {
        "engagement_track": {
            "scores": [None] * 5,
            "baseline_flag": "no_baseline_provided",
        },
        "section_report": [],
    }
    out = build_marketing_scores(session_dir, bundle)
    assert "preds_mean_abs" in out["provenance"]["activation_source"]
    assert out["provenance"]["baseline_flag"] == "no_baseline_provided"
    assert len(out["display_curve"]["points"]) == 5


def test_section_rollup_ranks():
    section_report = [
        {
            "section_id": "hero",
            "dwell_sec": 3.0,
            "engagement": {"mean": 0.2},
            "t_indices": [0, 1],
        },
        {
            "section_id": "pricing",
            "dwell_sec": 4.0,
            "engagement": {"mean": 1.0},
            "t_indices": [2, 3],
        },
    ]
    eng = np.array([0.2, 0.2, 1.0, 1.0])
    display = np.array([30.0, 35.0, 80.0, 85.0])
    scored = score_sections(section_report, eng, display)
    by_id = {s["section_id"]: s for s in scored}
    assert by_id["pricing"]["rank"] == 1
    assert by_id["pricing"]["label"] == "strongest"
    assert by_id["hero"]["label"] == "weakest"
    assert 0 <= by_id["hero"]["score"] <= 100
    assert by_id["pricing"]["score"] > by_id["hero"]["score"]


def test_activation_analysis_from_track():
    section_report = [
        {
            "section_id": "hero",
            "t_indices": [0, 1],
            "activation": {"mean_raw": 0.12},
        },
        {
            "section_id": "pricing",
            "t_indices": [2, 3],
            "activation": {"mean_raw": 0.18},
        },
    ]
    bundle = {
        "activation_track": {
            "raw_scores": [0.10, 0.12, 0.15, 0.18],
            "scores": [0.0, 0.5, 1.0, 1.5],
            "comparison_mode": "baseline_relative",
            "baseline_flag": None,
        },
    }
    out = build_activation_analysis(bundle, section_report, tr_duration_sec=1.0, preds=None)
    assert out is not None
    assert len(out["display_curve"]["points"]) == 4
    assert out["sections"][0]["section_id"] == "hero"
    by_id = {s["section_id"]: s for s in out["sections"]}
    assert by_id["pricing"]["score"] > by_id["hero"]["score"]


def test_map_population_quantile():
    ref = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    out = map_population_quantile(np.array([0.5, 1.5]), ref)
    assert out[0] < out[1]


def test_compute_comparison_score():
    display = np.array([80.0, 70.0, 60.0, 50.0])
    score = compute_comparison_score(display, early_fraction=0.25)
    assert 0.0 <= score <= 100.0


def test_norm_referenced_falls_back_provenance_when_population_too_small(tmp_path: Path, monkeypatch):
    from scout_core.marketing_scores import build_marketing_scores

    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    bundle = {
        "engagement_track": {"scores": [0.1, 0.2, 0.3], "baseline_flag": None},
        "section_report": [],
    }
    cfg = {
        "cross_session": {
            "enabled": True,
            "norm_id": "tiny_v1",
            "comparison_mode": "norm_referenced",
        },
        "compound_weights": {"engagement": 0.7, "novelty": 0.3},
        "edge_mask": {},
        "drop_thresholds": {},
    }
    monkeypatch.setattr(
        "scout_core.marketing_scores.load_population_engagement_samples",
        lambda _norm: np.array([0.5]),
    )
    out = build_marketing_scores(session_dir, bundle, config=cfg)
    prov = out["provenance"]
    assert prov["comparison_mode"] == "session_relative"
    assert prov["norm_fallback_reason"] == "insufficient_population_samples"
    assert prov["display_curve"] == "minmax_session_compound"
