---
id: ISSUE-003
session_id: CS-20260611-PRODUCT-CLAIMS
status: resolved
owner: agent
related:
  - FEATURE-003
  - CS-20260610-PLAYWRIGHT-EXPLORATION
files:
  - scout_core/explore_policy.py
  - scout_core/walkthrough.py
---

# Capture requires per-site YAML authoring

## Context

Only `scripted` and `linear` scroll modes exist; real marketing sites need autonomous nav/CTA discovery.

## Proposed Next Step

Ship `scroll_mode: explore` per FEATURE-003.
