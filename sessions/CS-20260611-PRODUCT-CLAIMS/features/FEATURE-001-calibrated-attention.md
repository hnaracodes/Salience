---
id: FEATURE-001
session_id: CS-20260611-PRODUCT-CLAIMS
status: in_progress
owner: agent
related:
  - ISSUE-001
files:
  - scout_core/attention_calibration.py
  - scripts/validate_attention_proxy.py
  - configs/attribution_calibrated.yaml
---

# Calibrated attention fusion + validation harness

## Goal

Learn attribution weights from behavioral ground truth and report Spearman/AUC metrics on held-out sessions.
