from __future__ import annotations

from scout_core.clarity_adapter import click_rows_by_selector, parse_clarity_click_csv


def test_parse_clarity_click_csv(tmp_path):
    csv_path = tmp_path / "clarity.csv"
    csv_path.write_text(
        "Selector,Text,Clicks,Sessions,URL\n#cta,Get started,12,100,https://example.com\n#hero,Hero,3,100,https://example.com\n",
        encoding="utf-8",
    )
    rows = parse_clarity_click_csv(csv_path)
    assert rows[0]["selector"] == "#cta"
    assert rows[0]["click_rate"] == 0.12
    index = click_rows_by_selector(rows)
    assert index["#hero"]["clicks"] == 3


def test_run_section_analytics_without_clarity_csv_unchanged(tmp_path):
    from scout_core.section_pipeline import run_section_analytics

    manifest = {
        "schema_version": 2,
        "initial_url": "http://127.0.0.1:8780/",
        "capture": {"width": 200, "height": 200, "tr_duration_sec": 1.0},
        "dom_snapshots": [
            {
                "t_idx": 0,
                "pts_sec": 0.0,
                "url": "http://127.0.0.1:8780/",
                "scrollY": 0,
                "scrollX": 0,
                "elements": [
                    {
                        "dom_id": "#cta",
                        "tag": "BUTTON",
                        "text": "Start",
                        "bbox": [10, 10, 100, 40],
                        "is_intersecting_viewport": True,
                        "visibility_ratio": 1.0,
                    },
                ],
            },
        ],
    }
    (tmp_path / "session_manifest.json").write_text(
        __import__("json").dumps(manifest),
        encoding="utf-8",
    )
    heatmaps = tmp_path / "heatmaps"
    heatmaps.mkdir()
    h = __import__("numpy").zeros((200, 200), dtype=__import__("numpy").float32)
    h[10:50, 10:110] = 1.0
    __import__("numpy").save(heatmaps / "t_0.npy", h)

    bundle = {
        "engagement_track": {"scores": [1.0], "labels": [None]},
        "emotion_track": {
            "template_names": ["anger"],
            "cosine_scores": [[0.05]],
            "z_scores": [[0.0]],
        },
        "activation_track": {"scores": [0.5], "raw_scores": [0.1], "labels": [None]},
    }
    extras: dict = {}
    report = run_section_analytics(tmp_path, bundle, attach_heatmaps=True, pipeline_extras=extras)
    assert report
    assert "clarity_attribution" not in extras
