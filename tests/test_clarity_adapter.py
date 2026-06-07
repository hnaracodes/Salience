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
