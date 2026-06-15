# Product claims registry

Maps UI copy and bundle fields to evidence tier. Update when validation memos change.

## Tier definitions

| Tier | Meaning |
|------|---------|
| T0 | Model-assisted hypothesis; disclaimers required |
| T1 | Engineering complete; provenance stamped |
| T2 | Validated on held-out behavioral or norm corpus |
| T3 | Supervised model with documented calibration |

## Claims by feature

### Cortical engagement proxy (dual-track)

| Field | Tier | Safe language |
|-------|------|---------------|
| `engagement_track.scores` | T1–T2 | Norm-referenced or baseline-relative cortical engagement proxy |
| `activation_track.scores` | T1–T2 | Mean vertex activation Z vs gray baseline or session |

**Not claimable:** Equivalent to fMRI measurement.

### Attention / element ranking

| Field | Tier | Safe language |
|-------|------|---------------|
| `section_report[].top_elements` | T0–T1 | Model-assisted element ranking hypothesis |
| After `validate_attention_proxy.py` pass | T2 | Attention proxy correlates with click telemetry on held-out pages |
| `heatmap_source: modal` | T1 | ViT self-attention map (not eye-tracking) |

### Cross-session scores

| Field | Tier | Safe language |
|-------|------|---------------|
| `marketing_scores.comparison_score` | T2 | Cross-session comparison on `naturalistic_v1` norm corpus |
| `marketing_scores.overall_score` | T1–T2 | Display score; norm-referenced when `comparison_mode: norm_referenced` |

### Autonomous scouting

| Field | Tier | Safe language |
|-------|------|---------------|
| `capture_mode: explore` | T1 | Bounded autonomous same-origin site exploration |
| `exploration_log` | T1 | Documented agent actions with safety budgets |

**Not claimable:** Replaces real user testing.

### Conversion intent

| Field | Tier | Safe language |
|-------|------|---------------|
| `conversion_prediction.probability` | T3 | Calibrated conversion intent score on held-out labels |

**Not claimable:** Guaranteed revenue lift.

### Copy engagement (multimodal fusion)

| Field | Tier | Safe language |
|-------|------|---------------|
| `copy_signals[].clarity` | T0–T1 | Heuristic readability proxy from visible DOM text |
| `copy_signals[].urgency` | T0–T1 | CTA/imperative keyword density heuristic |
| `copy_signals[].goal_fit` | T0–T1 | Keyword overlap with user-supplied `site_goal` |
| `marketing_scores.sections[].copy_score` | T0–T1 | Fused section score when `copy_source` is present |
| `marketing_scores.provenance.copy_source: llm` | T0 | Model-assisted copy assessment (optional Tier B) |

**Not claimable:** Copy scores reflect text heuristics or optional LLM pass — not user comprehension studies. Neural TRIBE scores remain pixel-driven; copy is a parallel signal fused at analyze time (`neural_only: false` in config).

## Evidence artifacts

| Artifact | Path |
|----------|------|
| Norm corpus | `scout_norms/naturalistic_v1/` |
| Attention validation | `scout_data/validation/attention_v1/` |
| Conversion labels | `scout_data/validation/conversion_labels.json` |
| Calibration config | `configs/attribution_calibrated.yaml` |
