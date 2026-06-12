---
id: FEATURE-002
session_id: CS-20260610-PLAYWRIGHT-EXPLORATION
status: open
owner: agent
related:
  - FEATURE-001
files:
  - scout_core/explore_policy.py
  - scout_core/element_goals.py
  - configs/llm_narrative.yaml
  - configs/explore_defaults.yaml
---

# Goal-guided exploration (optional Phase 2)

## Goal

When `explore.guidance: goal` is set and `site_goal:` is present in the walkthrough YAML, rank candidate next actions using the existing `site_goal` + visible element metadata (and optionally a lightweight Gemini call) instead of pure heuristic scoring alone.

## Context

`pipeline-runner` already documents `site_goal:` in showcase YAMLs for the **narrative** stage. Reuse that text during **capture** so exploration prioritizes elements aligned with the stated product intent (e.g. "drive signup" → prefer `#hero-cta`, pricing, subscribe nav).

## Implementation Notes

- Default Phase 1 uses deterministic heuristic policy only (no API key required)
- Phase 2 adds `explore.guidance: heuristic | goal | gemini` with template fallback
- Explicitly out of scope for Phase 1 implementation

## Validation

- With `guidance: heuristic`, explore session identical to Phase 1 baseline
- With `guidance: goal`, Threadmind explore prefers documented CTA selectors before footer links (unit test with frozen element list)
