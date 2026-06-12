---
id: ISSUE-001
session_id: CS-20260611-PRODUCT-CLAIMS
status: in_progress
owner: agent
related:
  - FEATURE-001
files:
  - scout_core/attention_calibration.py
  - scripts/validate_attention_proxy.py
---

# Spatial attention not validated vs behavioral ground truth

## Context

Element rankings fuse TRIBE temporal signals with CPU/DINOv2 spatial maps but lack calibration against Clarity/click telemetry or eye-tracking.

## Proposed Next Step

Run `validate_attention_proxy.py` on `scout_data/validation/attention_v1/` after ground-truth collection.
