---
id: FEATURE-001
session_id: CS-20260607-USABLE-UX-INSIGHTS
status: resolved
owner: agent
related:
  - CS-20260610-USABLE-UX-FOLLOWUPS
files:
  - scout_core/walkthrough.py
  - tests/test_record_website_session.py
  - configs/walkthrough_scripts/localhost_demo.yaml
---

# Playwright interaction event capture validation

## Goal

Prove that scripted `hover` and `click` walkthrough steps produce durable `interaction_events[]` entries in `session_manifest.json` with correct `t_idx`, `pts_sec`, `action`, `selector`, and resolved `target` metadata — without requiring a full Modal E2E run on every pytest invocation.

## Context

`localhost_demo.yaml` already includes hover/click on `#cta-hero`. Scripted capture in `record_website_session_async` appends markers when `_run_step` returns non-`None`. There is no regression test covering this contract. Future attribution and viewer overlays may rely on interaction timing.

## Implementation Notes

- Add unit tests for marker shape via `build_manifest_v2(..., interaction_events=...)`.
- Add async unit test with mocked Playwright page for `_run_step` click/hover return payloads.
- Add optional `@pytest.mark.playwright` integration test against `tests/fixtures/walkthrough_site` when Chromium is available.
- Fix duplicate `interaction_events = []` initialization in `_record_linear_scroll_async` (cosmetic; linear mode does not emit interaction events today).

## Validation

- `python -m pytest tests/test_record_website_session.py -v`
- Scripted capture on localhost demo yields `len(interaction_events) >= 2` with monotonic `pts_sec`.
