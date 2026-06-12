---
id: FEATURE-002
session_id: CS-20260607-USABLE-UX-INSIGHTS
status: resolved
owner: agent
related:
  - CS-20260610-USABLE-UX-FOLLOWUPS
files:
  - scout_core/attention_attribution.py
  - scout_core/section_pipeline.py
  - scout_core/dom_intersect.py
  - scout_core/visual_saliency.py
  - configs/section_analytics.yaml
  - tests/test_attention_attribution.py
---

# Per-TR element attribution (saliency × visibility × brain weight)

## Goal

Replace section-level brain-weight blending with a mathematically explicit per-timestep sum across all section dwell TRs:

```text
contribution(dom_id) = Σ_t saliency(e, t) × visibility(e, t) × brain_weight(t)
```

where `brain_weight(t)` blends normalized engagement and activation at TR `t`, and `saliency(e, t)` comes from heatmap/DOM intersection or CPU visual saliency at that TR.

## Context

`enrich_sections_with_attribution` today:

1. Averages engagement/activation over `section.t_indices` into one `brain_weight`.
2. Multiplies rolled-up `mean_attention_density` (from at most 3 **sample** TRs) by that scalar.

This understates elements visible during high-engagement TRs that were not heatmap-sampled, and overstates section-average moments. The June 7 session report flagged this as the top attribution fidelity gap.

## Implementation Notes

- Add `attribution.mode` in `configs/section_analytics.yaml`: `section_blend` (current) | `per_tr_sum` (new).
- New function `aggregate_per_tr_element_scores(...)` in `attention_attribution.py`.
- For each `t` in `section.t_indices` (not only `sample_t_indices`):
  - Skip TRs outside manifest bounds (`is_timestep_in_manifest`).
  - Load `heatmaps/t_{t}.npy` if present; else compute CPU saliency from `frames/t_{t}.jpg` + nearest snapshot.
  - Use `score_all_elements` + element `visibility_ratio` from DOM snapshot.
- Sum contributions per `dom_id`; normalize to 0–100 `attention_score`; keep `clickability` heuristic; recompute `combined_score`.
- Preserve backward-compatible `section_blend` path until golden-session comparison passes.

## Validation

- Unit test: synthetic 5-TR section where only `t=2` has high engagement — per-TR mode ranks element visible at `t=2` above element only visible at `t=0`.
- Aurora or localhost session: `section_report[].top_elements` order changes predictably when toggling `attribution.mode`.
- `python -m pytest tests/test_attention_attribution.py -v`
