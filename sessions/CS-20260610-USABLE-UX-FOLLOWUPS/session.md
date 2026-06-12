---
session_id: CS-20260610-USABLE-UX-FOLLOWUPS
title: Usable UX follow-ups — interaction tests, per-TR attribution, Clarity wiring
date: 2026-06-10
model: composer-2.5
signature: composer-2.5@CS-20260610-USABLE-UX-FOLLOWUPS
status: completed
track: website
related:
  - CS-20260607-USABLE-UX-INSIGHTS
files_touched:
  - scout_core/attention_attribution.py
  - scout_core/section_pipeline.py
  - scout_core/walkthrough.py
  - configs/section_analytics.yaml
  - scripts/analyze_session.py
  - viewer/ux_session_viewer.html
  - tests/test_record_website_session.py
  - tests/test_attention_attribution.py
  - tests/test_clarity_adapter.py
  - tests/test_section_analytics.py
  - docs/runbooks/website-session.md
---

# Usable UX Follow-Ups

## Summary

Planning session to implement three open items from `CS-20260607-USABLE-UX-INSIGHTS`: Playwright interaction-event test coverage, per-TR attribution math, and Clarity CSV wiring into element scores. No NeuroEmo involvement; production emotion remains Kragel zero-shot.

## Scope

| ID | Item | Status |
|----|------|--------|
| FEATURE-001 | Playwright interaction events test | open |
| FEATURE-002 | Per-TR attribution | open |
| FEATURE-003 | Clarity attribution wiring | open |
| ISSUE-001 | Attribution fidelity gap (tracks FEATURE-002) | open |

## Plan

See [`.cursor/plans/usable-ux-followups_implementation.plan.md`](../../.cursor/plans/usable-ux-followups_implementation.plan.md).

## How To Continue

1. Review implementation plan with user.
2. Implement Phase 1 → Phase 3 in order (tests first, then attribution, then Clarity).
3. Update feature/issue `status` fields and append `ledger.md` as work lands.

## Signature

Signed-off-by: Composer 2.5 (composer-2.5) on 2026-06-10
