---
id: ISSUE-SEC-002
status: resolved
severity: high
category: accuracy
title: Marketing scores mislabeled as norm_referenced
---

# ISSUE-SEC-002 — Marketing provenance

## Problem

`comparison_mode: norm_referenced` in config while `naturalistic_v1` had only one population sample caused silent fallback to session-relative minmax while provenance still claimed norm-referenced scoring.

## Fix

`build_marketing_scores` sets `effective_comparison_mode: session_relative`, `norm_fallback_reason: insufficient_population_samples`, and clears `norm_id` when population n&lt;2.

## User impact

Scores are no longer misrepresented in the viewer KPI strip and `analysis_bundle.json` provenance.
