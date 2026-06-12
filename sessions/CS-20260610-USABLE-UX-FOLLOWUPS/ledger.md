# Ledger

## Step 1 — Planning (2026-06-10)

- Created ledger features and issue under `CS-20260607-USABLE-UX-INSIGHTS`.
- Authored `.cursor/plans/usable-ux-followups_implementation.plan.md`.
- Signature: `composer-2.5@CS-20260610-USABLE-UX-FOLLOWUPS`

## Step 2 — Implementation (2026-06-10)

- **FEATURE-001:** Added interaction event manifest tests, `interaction_t_idx()`, mocked `_run_step` click/hover tests; removed duplicate `interaction_events` init in linear scroll path.
- **FEATURE-002 / ISSUE-001:** Implemented `per_tr_sum` attribution (default per user choice) with `aggregate_per_tr_element_scores`, heatmap/frame fallback via `load_or_compute_heatmap`, `section_blend` fallback mode; config in `configs/section_analytics.yaml`.
- **FEATURE-003:** Wired `--clarity-csv` in `analyze_session.py`, `enrich_sections_with_clarity` (50/50 blend per user choice), `clarity_attribution` bundle block, viewer Clarity badge.
- **Docs:** Updated `docs/runbooks/website-session.md`.

### Files touched

- `scout_core/attention_attribution.py`
- `scout_core/section_pipeline.py`
- `scout_core/walkthrough.py`
- `configs/section_analytics.yaml`
- `scripts/analyze_session.py`
- `viewer/ux_session_viewer.html`
- `tests/test_record_website_session.py`
- `tests/test_attention_attribution.py`
- `tests/test_clarity_adapter.py`
- `tests/test_section_analytics.py`
- `docs/runbooks/website-session.md`

### Validation

```text
pytest tests/test_record_website_session.py tests/test_attention_attribution.py tests/test_clarity_adapter.py tests/test_section_analytics.py tests/test_dom_score_all.py -v
43 passed in 4.47s
```

- Signature: `composer-2.5@CS-20260610-USABLE-UX-FOLLOWUPS`
