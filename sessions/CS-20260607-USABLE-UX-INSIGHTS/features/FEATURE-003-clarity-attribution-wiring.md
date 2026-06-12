---
id: FEATURE-003
session_id: CS-20260607-USABLE-UX-INSIGHTS
status: resolved
owner: agent
related:
  - CS-20260610-USABLE-UX-FOLLOWUPS
files:
  - scout_core/clarity_adapter.py
  - scout_core/attention_attribution.py
  - scout_core/section_pipeline.py
  - scripts/analyze_session.py
  - configs/section_analytics.yaml
  - tests/test_clarity_adapter.py
---

# Microsoft Clarity CSV → element attribution wiring

## Goal

When the user supplies an offline Clarity click-export CSV for an owned site, merge real click counts into `section_report[].top_elements[]` and optionally boost effective clickability — without any live Clarity API dependency.

## Context

`scout_core/clarity_adapter.py` parses CSV exports and indexes rows by selector. The June 7 session built the adapter only; attribution and `analyze_session.py` do not consume it yet. This is optional enrichment for sessions where behavioral ground truth exists.

## Implementation Notes

- CLI flag: `analyze_session.py --clarity-csv PATH` (optional; no-op if missing).
- Convention fallback: `scout_data/sessions/<id>/clarity_clicks.csv` if flag omitted and file exists.
- Match Clarity `selector`/`dom_id` to element `dom_id` (exact, then normalized CSS selector).
- Add fields: `clarity_clicks`, `clarity_click_rate`, `clarity_matched`.
- Optional blend: `effective_clickability = (1 - w) * heuristic + w * clarity_norm` with `w` from config (default 0.35 when matched).
- Write `bundle.clarity_attribution = {source, n_rows, n_matched, path}` for viewer provenance.
- Do not require Clarity for demo sessions; heuristic clickability remains default.

## Validation

- Unit test: fixture CSV + synthetic `top_elements` → correct match counts and boosted `combined_score` ordering.
- Session without CSV unchanged (backward compatible).
- `python -m pytest tests/test_clarity_adapter.py tests/test_attention_attribution.py -v`
