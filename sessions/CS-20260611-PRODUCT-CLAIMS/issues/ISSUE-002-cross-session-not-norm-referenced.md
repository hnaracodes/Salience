---
id: ISSUE-002
session_id: CS-20260611-PRODUCT-CLAIMS
status: in_progress
owner: agent
related:
  - FEATURE-002
files:
  - scout_core/marketing_scores.py
  - scripts/compute_norms.py
---

# Cross-session scores not norm-referenced

## Context

Marketing 0–100 uses per-session min–max; synthetic_bootstrap_v1 is not production-grade for absolute claims.

## Proposed Next Step

Build `naturalistic_v1` norms and wire `comparison_mode: norm_referenced` in marketing_scores.
