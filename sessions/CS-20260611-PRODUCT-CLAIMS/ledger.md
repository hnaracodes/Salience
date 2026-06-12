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
