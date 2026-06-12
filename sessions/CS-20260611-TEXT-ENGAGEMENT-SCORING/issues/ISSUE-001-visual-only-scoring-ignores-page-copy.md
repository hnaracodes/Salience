---
id: ISSUE-001
session_id: CS-20260611-TEXT-ENGAGEMENT-SCORING
status: open
owner: agent
related:
  - CS-20260521-WEB-PIPELINE-FOUNDATION
  - CS-20260528-WEBSITE-CONSOLIDATED
  - CS-20260607-USABLE-UX-INSIGHTS
  - CS-20260611-PRODUCT-CLAIMS
files:
  - scout_core/walkthrough.py
  - scout_core/visual_saliency.py
  - scout_core/marketing_scores.py
  - scout_core/section_pipeline.py
  - scout_core/section_recommendations.py
  - scout_core/llm_narrative.py
  - scout_core/attention_attribution.py
  - configs/llm_narrative.yaml
  - scripts/generate_session_narrative.py
---

# Ratings are visual/neural-only; on-page copy does not shape engagement scores

## Context

TribeV2's **scoring and rating spine** is driven almost entirely by **visual stimulus** processed through TRIBE (`preds.npz`) and pixel-based saliency — not by semantic analysis of the text users actually read on the page.

### What the pipeline does today

| Layer | Input | Role of text |
|-------|-------|--------------|
| **TRIBE / dual-track** | Walkthrough video frames | None — engagement Z (VAN−DMN), Kragel emotion, and activation are inferred from neural response to **pixels** only |
| **CPU / DINOv2 saliency** | Screenshot + DOM bbox | Text used only for **CTA keyword heuristics** (`"get started"`, `"sign up"`, etc.) in `visual_saliency._type_score()` — not tone, clarity, or emotional valence of copy |
| **Marketing scores** | `engagement_track` + `preds` novelty | No text fields — `opening`, `sustain`, `comparison_score`, etc. are pure neural transforms |
| **Section recommendations** | Section-level neural aggregates | Generic copy advice ("shorten copy", "strengthen CTA") triggered by **boredom/arousal proxies**, not analysis of actual headline or body text |
| **LLM narrative** (`llm_narrative.py`) | Structured `section_report` + element `text` | **Post-hoc** Gemini/template summary; references element text in prompts but does **not** feed back into ratings, attribution weights, or `marketing_scores` |

### What is captured but underused

Playwright capture **does** snapshot element text (`innerText` / `textContent`, truncated to ~160–200 chars) into `session_manifest.json` per TR. That text flows into:

- `section_report.top_elements[].text`
- LLM narrative payload (`build_narrative_payload`)
- Minor saliency type-weight boosts for CTA phrases

It does **not** currently drive:

- Per-section or per-element **text engagement** or **copy-feel** scores
- Fusion with neural engagement/emotion traces at each TR
- Marketing-facing ratings that reflect whether displayed copy is persuasive, confusing, urgent, or flat

### Why this matters for true ratings

A page can look visually strong (high contrast, centered hero, motion) while copy is vague, anxiety-inducing, or misaligned with `site_goal`. Conversely, strong messaging on a visually plain layout may be under-valued. Without a **text channel** alongside the visual/neural channel:

1. **Engagement ratings** reflect "how the brain responded to pixels," not "how the message landed."
2. **Element attribution** ranks by saliency × neural weight, not by whether the **words** on that element explain a spike or drop.
3. **Product claims** (`CS-20260611-PRODUCT-CLAIMS`) risk overstating UX quality when visuals score well but copy undermines conversion intent.
4. The existing **LLM pipeline** is positioned as narrative export, not as a **scoring modality** — stakeholders may assume the LLM "understood" the page when ratings were still visual-only.

### User intent (June 11)

Integrate **engagement rating / feelings derived from displayed text** into the analysis spine **in addition to** visuals. The LLM infrastructure (`configs/llm_narrative.yaml`, `scout_core/llm_narrative.py`, `generate_session_narrative.py`) is a likely integration point, but text signals should influence structured scores in `analysis_bundle.json`, not only the final prose summary.

## Gap summary

```
[Video pixels] ──► TRIBE ──► engagement / emotion / activation  ──► marketing_scores ✓
                                                                      section_report ✓
[DOM text]     ──► manifest only ──► LLM narrative (optional)     ──► ratings ✗
                     └─ CTA keyword heuristics in saliency only
```

## Proposed Next Step

1. **Audit** per-TR visible text in manifest vs. what `section_pipeline` and `marketing_scores` consume today.
2. **Design a `text_engagement` track** (or per-section `copy_signals`) — e.g. LLM or lightweight classifier scoring clarity, urgency, trust, friction, and goal-alignment on visible copy per section/TR, outputting structured JSON (not free prose).
3. **Fuse** text signals with neural engagement in `marketing_scores` and/or `attention_attribution` (configurable weights; preserve neural-only mode for A/B).
4. **Wire** fused scores into `section_report`, viewer, and narrative payload so ratings and copy insights stay consistent.
5. **Document** in `docs/validation/CLAIMS.md` that ratings are multimodal (visual + copy) once shipped.

Defer NeuroEmo supervised emotion for this issue — goal is **on-page copy semantics**, not fMRI-labeled affect.

## Links

- LLM narrative foundation: `CS-20260521-WEB-PIPELINE-FOUNDATION` report §10
- Triple-track scoring (visual/neural): `CS-20260528-WEBSITE-CONSOLIDATED`
- Saliency + attribution (visual-only fusion today): `CS-20260607-USABLE-UX-INSIGHTS`
- Product claims / rating defensibility: `CS-20260611-PRODUCT-CLAIMS`
- Config: `configs/llm_narrative.yaml` (prompt already asks to reference element text)
