"""Tests for section-level marketing analytics."""

from __future__ import annotations

import numpy as np
import pytest

from scout_core.section_analytics import (
    aggregate_section_metrics,
    assign_section_for_snapshot,
    assign_timesteps_to_sections,
    build_section_report,
)
from scout_core.section_pipeline import enrich_section_report_with_attention, run_section_analytics
from scout_core.section_recommendations import generate_section_recommendations
from scout_core.section_sampling import attach_sample_timesteps


def _manifest_two_sections() -> dict:
    return {
        "capture": {"width": 1000, "height": 800},
        "initial_url": "https://example.com/",
        "dom_snapshots": [
            {
                "t_idx": 0,
                "url": "https://example.com/",
                "scrollY": 0,
                "scrollX": 0,
                "elements": [
                    {"dom_id": "#hero", "tag": "SECTION", "bbox": [0, 0, 1000, 400], "is_intersecting_viewport": True},
                    {"dom_id": "#btn", "tag": "BUTTON", "bbox": [100, 100, 80, 40], "is_intersecting_viewport": True},
                ],
            },
            {
                "t_idx": 5,
                "url": "https://example.com/pricing",
                "scrollY": 0,
                "scrollX": 0,
                "elements": [
                    {"dom_id": "#pricing", "tag": "SECTION", "bbox": [0, 0, 1000, 600], "is_intersecting_viewport": True},
                ],
            },
        ],
    }


def _bundle_t6() -> dict:
    names = ["contentment", "fear", "anger"]
    z = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    return {
        "engagement_track": {
            "scores": [0.5, 0.5, 0.5, 2.5, 0.5, -2.0],
            "labels": [None, None, None, "engaging", None, "boring"],
        },
        "emotion_track": {
            "template_names": names,
            "cosine_scores": [[0.05] * 3] * 6,
            "z_scores": z,
        },
        "activation_track": {
            "raw_scores": [0.10, 0.11, 0.12, 0.20, 0.11, 0.09],
            "scores": [-1.0, -0.5, 0.0, 2.0, -0.5, -1.5],
            "labels": [None, None, None, "high_attention", None, "low_attention"],
            "comparison_mode": "baseline_relative",
        },
    }


class TestSectionAssignment:
    def test_landmark_at_viewport_center(self):
        snap = _manifest_two_sections()["dom_snapshots"][0]
        cap = _manifest_two_sections()["capture"]
        sid, method, bbox = assign_section_for_snapshot(snap, cap, None)
        assert method in ("landmark", "manual", "scroll_band")
        assert sid

    def test_assign_all_timesteps(self):
        m = _manifest_two_sections()
        assigns = assign_timesteps_to_sections(6, m)
        assert len(assigns) == 6
        assert assigns[0].t_idx == 0
        assert assigns[5].t_idx == 5

    def test_temporal_fallback_without_manifest(self):
        assigns = assign_timesteps_to_sections(9, None)
        assert len(assigns) == 9
        assert assigns[0].section_id.startswith("temporal/")


class TestSectionAggregate:
    def test_aggregate_groups_by_section(self):
        m = _manifest_two_sections()
        assigns = assign_timesteps_to_sections(6, m)
        report = aggregate_section_metrics(_bundle_t6(), assigns)
        assert len(report) >= 1
        total_trs = sum(r["dwell_trs"] for r in report)
        assert total_trs == 6

    def test_build_section_report(self):
        report = build_section_report(_bundle_t6(), _manifest_two_sections())
        assert report
        assert "emotion" in report[0]
        assert "activation" in report[0]
        assert report[0]["activation"]["mean_raw"] is not None

    def test_activation_high_at_spike_tr(self):
        assigns = assign_timesteps_to_sections(6, _manifest_two_sections())
        report = aggregate_section_metrics(_bundle_t6(), assigns)
        by_id = {r["section_id"]: r for r in report}
        # t=3 has highest raw activation in fixture
        for r in report:
            if 3 in r["t_indices"]:
                assert r["activation"]["mean_raw"] >= 0.12


class TestSectionSampling:
    def test_attach_sample_timesteps(self):
        report = build_section_report(_bundle_t6(), _manifest_two_sections())
        global_ts = attach_sample_timesteps(report, _bundle_t6(), max_per_section=3, max_total=10)
        assert global_ts
        assert all("sample_t_indices" in sec for sec in report)


class TestSectionRecommendations:
    def test_generates_at_least_one_tip(self):
        sec = {
            "dwell_sec": 5,
            "engagement": {"pct_boring": 0.5, "pct_engaging": 0.0, "mean": -1.0},
            "emotion": {"mean_z": {"fear": 1.5}, "peak": {"channel": "fear", "z_score": 2.5}},
            "flags": ["high_arousal"],
            "top_elements": [],
        }
        tips = generate_section_recommendations(sec)
        assert len(tips) >= 1


class TestSectionHeatmapEnrich:
    def test_rollup_top_elements(self, tmp_path):
        manifest = _manifest_two_sections()
        heatmaps_dir = tmp_path / "heatmaps"
        heatmaps_dir.mkdir()
        h = np.ones((800, 1000), dtype=np.float32)
        np.save(heatmaps_dir / "t_0.npy", h)

        report = build_section_report(_bundle_t6(), manifest)
        for sec in report:
            sec["sample_t_indices"] = [0]

        enrich_section_report_with_attention(tmp_path, manifest, report, min_element_area=1, top_k=3)
        assert report[0]["top_elements"]

    def test_run_section_analytics_adds_element_attribution(self, tmp_path):
        session_dir = tmp_path
        manifest = _manifest_two_sections()
        (session_dir / "session_manifest.json").write_text(
            __import__("json").dumps(manifest),
            encoding="utf-8",
        )
        heatmaps_dir = session_dir / "heatmaps"
        heatmaps_dir.mkdir()
        h = np.ones((800, 1000), dtype=np.float32)
        h[80:170, 80:220] = 4.0
        np.save(heatmaps_dir / "t_0.npy", h)
        report = run_section_analytics(session_dir, _bundle_t6(), attach_heatmaps=True)
        with_elements = [sec for sec in report if sec.get("top_elements")]
        assert with_elements
        assert "combined_score" in with_elements[0]["top_elements"][0]
        assert with_elements[0]["element_attribution"]["mode"] == "per_tr_sum"
