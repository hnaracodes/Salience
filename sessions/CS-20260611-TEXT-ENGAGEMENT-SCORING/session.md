---
session_id: CS-20260611-TEXT-ENGAGEMENT-SCORING
title: Text Content Engagement Scoring
date: 2026-06-11
model: composer-2.5
signature: composer-2.5@CS-20260611-TEXT-ENGAGEMENT-SCORING
status: in_progress
track: website
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

- Issue logged; no code changes yet.

## How To Continue

- Read `issues/ISSUE-001-visual-only-scoring-ignores-page-copy.md`
- Design `text_engagement` track and fusion weights with neural scores
- Implement structured copy scoring stage before or alongside narrative generation

## Signature

Signed-off-by: Composer 2.5 (composer-2.5) on 2026-06-11
