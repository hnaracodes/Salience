# Ledger

## Step 1 — Kickoff (2026-06-11)

- Created session `CS-20260611-PRODUCT-CLAIMS` from Product Claims Roadmap plan.
- Issues: ISSUE-001–003; Features: FEATURE-001–004.
- Signature: `composer-2.5@CS-20260611-PRODUCT-CLAIMS`

## Step 2 — Implementation (2026-06-11)

### Track C — Explore mode (FEATURE-003)
- `scout_core/explore_policy.py`, `_record_explore_async` in `walkthrough.py`
- `configs/explore_defaults.yaml`, `configs/walkthrough_scripts/explore_threadmind.yaml`
- `tests/test_explore_policy.py` (5 passed)

### Track B — Cross-session norms (FEATURE-002)
- `scripts/build_naturalistic_norms.py`, `scripts/compare_sessions.py`
- `naturalistic_v1` norm bundle + `engagement_samples.json` (1 session; expand with more captures)
- `marketing_scores.py`: `comparison_score`, `norm_referenced` display curve, schema v2

### Track A — Attention / heatmaps (FEATURE-001)
- Gaussian element-localized heatmaps in `visual_saliency.py`
- Viewer element mask toggle in `ux_session_viewer.html`
- `--production` flag on `extract_section_heatmaps.py`
- `scout_core/attention_calibration.py`, `scripts/validate_attention_proxy.py`
- `configs/attribution_calibrated.yaml`
- Validation scaffold: `scout_data/validation/attention_v1/`

### Track A — Conversion (FEATURE-004)
- `scout_core/conversion_model.py`, `train_conversion_model.py`, `score_conversion.py`
- Auto-score in `analyze_session.py` when `scout_data/models/conversion_v1.npz` exists

### Governance
- `docs/validation/CLAIMS.md`, `docs/validation/attention-calibration-protocol.md`
- Runbook updates for explore + production heatmaps

### Validation
```text
pytest tests/test_explore_policy.py tests/test_visual_saliency.py tests/test_marketing_scores.py tests/test_attention_calibration.py tests/test_conversion_model.py tests/test_record_website_session.py -v
48 passed in 14.60s
python scripts/build_naturalistic_norms.py --min-clips 1
```

- Signature: `composer-2.5@CS-20260611-PRODUCT-CLAIMS`

## Step 3 — Claim audit + T2/T3 readiness (2026-07-16)

### Governance
- Downgraded premature T2/T3 language in `docs/validation/CLAIMS.md`
- Clarified Clarity is MIT open-source instrumentation; hosted telemetry is private
- Added `docs/validation/HUMAN_INTERVENTION_CLARITY.md` for owned-site CSV collection

### Website corpus (preliminary TRIBE / norms only)
- `configs/validation_site_corpus.yaml` — 8 public archetypes × 3 planned repeats
- `configs/explore_public_smoke.yaml` — zero-click read-only explore profile
- `scripts/run_validation_site_corpus.py` — batch capture/TRIBE runner + run log

### Attention harness hardening
- NaN Spearman fails; average ranks for ties; reject Clarity-leaked bundles
- Property-grouped train/validation/holdout fit (`--fit-weights --evaluate-holdout`)
- `build_naturalistic_norms.py` default `--min-clips 15` hard fail; `compute_norms.py --exclude`

### Blocked on human
- Owned Clarity properties + heatmap CSVs required for Attention T2
- Conversion labels (real outcomes) required for T3

### Handoff
- Full day protocol + tomorrow steps: [`../CS-20260716-DAILY-HANDOFF/ledger.md`](../CS-20260716-DAILY-HANDOFF/ledger.md)
- Local captures OK (threadmind/aurora ×2); **no preds.npz yet** — next: tribe+analyze then norms

- Signature: `cursor-grok-4.5@CS-20260611-PRODUCT-CLAIMS`
