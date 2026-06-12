---
id: ISSUE-001
session_id: CS-20260607-USABLE-UX-INSIGHTS
status: resolved
owner: agent
related:
  - FEATURE-002
  - CS-20260610-USABLE-UX-FOLLOWUPS
files:
  - scout_core/attention_attribution.py
  - scout_core/section_pipeline.py
---

# Attribution uses section-average brain weight, not per-TR summation

## Context

The usable UX insights upgrade shipped demo-quality element ranking, but attribution math collapses temporal brain signals before combining with spatial saliency. Elements that align with brief engagement spikes can be under-ranked; elements dominant only in sampled heatmap TRs can be over-ranked.

Documented in `report.md` §Implementation note and `FEATURE-002`.

## Proposed Next Step

Implement `FEATURE-002` behind `attribution.mode: per_tr_sum`, compare rankings on Aurora/localhost sessions against current `section_blend`, then flip default if stable.

## Links

- Parent session: `CS-20260607-USABLE-UX-INSIGHTS`
- Implementation session: `CS-20260610-USABLE-UX-FOLLOWUPS`
