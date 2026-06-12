---
session_id: CS-20260610-PLAYWRIGHT-EXPLORATION
title: Playwright explore mode — autonomous website scouring
date: 2026-06-10
model: composer-2.5
signature: composer-2.5@CS-20260610-PLAYWRIGHT-EXPLORATION
status: in_progress
track: website
related:
  - CS-20260521-WEB-PIPELINE-FOUNDATION
  - CS-20260610-USABLE-UX-FOLLOWUPS
files_touched:
  - .cursor/plans/playwright-exploration-mode.plan.md
  - sessions/CS-20260610-PLAYWRIGHT-EXPLORATION/issues/ISSUE-001-scripted-capture-limits-real-sites.md
  - sessions/CS-20260610-PLAYWRIGHT-EXPLORATION/features/FEATURE-001-explore-scroll-mode.md
  - sessions/CS-20260610-PLAYWRIGHT-EXPLORATION/features/FEATURE-002-goal-guided-exploration.md
---

# Playwright Explore Mode

## Summary

Planning session to address scripted-capture limitations on real websites. Adds a bounded `scroll_mode: explore` path so Playwright can discover nav, CTAs, and same-origin pages without hand-authored YAML steps. Validated against the `pipeline-runner` artifact checklist and E2E ladder.

## Open items

| ID | Item | Status |
|----|------|--------|
| ISSUE-001 | Scripted capture cannot scour real sites | open |
| FEATURE-001 | Explore scroll mode (heuristic policy) | open |
| FEATURE-002 | Goal-guided exploration (Phase 2) | open |

## Plan

See [`.cursor/plans/playwright-exploration-mode.plan.md`](../../.cursor/plans/playwright-exploration-mode.plan.md).

## How To Continue

1. User reviews plan and budgets (max_tr, max_clicks, allowlist policy).
2. Implement Phase 1 per plan; run `pipeline-runner` Layer 1–3 on Threadmind explore fixture.
3. Update ledger statuses; extend `pipeline-runner.md` with explore commands.

## Signature

Signed-off-by: Composer 2.5 (composer-2.5) on 2026-06-10
