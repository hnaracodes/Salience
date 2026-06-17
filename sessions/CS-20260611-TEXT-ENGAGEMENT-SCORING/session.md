---
session_id: CS-20260611-TEXT-ENGAGEMENT-SCORING
title: Text Content Engagement Scoring
date: 2026-06-11
model: composer-2.5
signature: composer-2.5@CS-20260611-TEXT-ENGAGEMENT-SCORING
status: completed
track: website
agent_transcripts:
  - eec407df-03ee-4fc1-b11f-21abee43c744
related:
  - CS-20260521-WEB-PIPELINE-FOUNDATION
  - CS-20260528-WEBSITE-CONSOLIDATED
  - CS-20260607-USABLE-UX-INSIGHTS
  - CS-20260611-PRODUCT-CLAIMS
files_touched:
  - sessions/CS-20260611-TEXT-ENGAGEMENT-SCORING/issues/ISSUE-001-visual-only-scoring-ignores-page-copy.md
---

# Text Content Engagement Scoring

## Summary

Logged a product gap: TribeV2 ratings and attribution are driven by visual/neural signals while on-page copy is captured but not used to shape engagement scores. The existing LLM narrative layer references element text post-hoc but does not feed structured text-based engagement or copy-feel signals back into `marketing_scores` or section analytics.

## Important Files

- `scout_core/llm_narrative.py` — LLM payload includes element text; narrative-only today
- `scout_core/marketing_scores.py` — neural-only rating transforms
- `scout_core/walkthrough.py` — DOM text capture at each TR
- `scout_core/visual_saliency.py` — CTA keyword heuristics only

## Validation

- ISSUE-001 logged in transcript `eec407df-03ee-4fc1-b11f-21abee43c744`
- Implementation completed in `CS-20260612-SAAS-PRODUCTION` (`scout_core/copy_signals.py`, marketing fusion)
- ISSUE-001 status: resolved

## How To Continue

- Optional Tier B: LLM-based `copy_source: llm` override per issue proposal
- Replace heuristic weights via config if product tuning needed

## Signature

Signed-off-by: Composer 2.5 (composer-2.5) on 2026-06-11
