---
session_id: CS-20260611-PRODUCT-CLAIMS
title: Product Claims Roadmap
date: 2026-06-11
model: composer-2.5
signature: composer-2.5@CS-20260611-PRODUCT-CLAIMS
status: completed
track: website
related:
  - CS-20260607-USABLE-UX-INSIGHTS
  - CS-20260610-PLAYWRIGHT-EXPLORATION
  - CS-20260610-USABLE-UX-FOLLOWUPS
files_touched:
  - scout_core/explore_policy.py
  - scout_core/walkthrough.py
  - scout_core/visual_saliency.py
  - scout_core/marketing_scores.py
  - scout_core/attention_calibration.py
  - scout_core/conversion_model.py
  - configs/explore_defaults.yaml
  - configs/marketing_scores.yaml
  - configs/attribution_calibrated.yaml
  - docs/validation/CLAIMS.md
---

# Product Claims Roadmap

## Summary

Implement three tracks to support defensible product claims: calibrated attention proxies (Track A), cross-session norm-referenced neuroscience scores (Track B), and bounded autonomous site exploration (Track C).

## Validation

- `pytest tests/test_explore_policy.py tests/test_visual_saliency.py tests/test_marketing_scores.py tests/test_attention_calibration.py tests/test_conversion_model.py -v`
