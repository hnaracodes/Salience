---
id: FEATURE-004
session_id: CS-20260611-PRODUCT-CLAIMS
status: in_progress
owner: agent
related:
  - FEATURE-001
files:
  - scout_core/conversion_model.py
  - scripts/train_conversion_model.py
  - scripts/score_conversion.py
---

# Conversion intent model (supervised, post-labels)

## Goal

Logistic conversion probability from bundle features with Platt calibration when labels exist.
