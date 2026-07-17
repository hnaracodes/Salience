# Attention calibration protocol

## Goal

Validate that element `combined_score` rankings correlate with behavioral telemetry (Clarity click exports) before claiming T2 attention proxy language.

## Steps

1. Use an owned or explicitly authorized property with Microsoft Clarity installed.
2. Capture the page with Playwright (`explore` or `scripted`).
3. Run the full pipeline through `analyze_session.py --website` **before** making the
   Clarity CSV discoverable to the analyzer. A bundle containing `clarity_attribution`
   is rejected as label leakage.
4. Export the Clarity click heatmap CSV for the same URL, device filter, and traffic
   window. Store it under `scout_data/validation/attention_v1/exports/`.
5. Record the session in `attention_v1/manifest.json` with `property_id`, `page_id`,
   `role`, `unique_sessions`, `mapped_clicks`, and `clarity_csv`.
6. Freeze property-grouped `train`, `validation`, and `holdout` splits. A property
   must not occur in more than one split.
7. Fit fusion weights without holdout data, then evaluate the holdout once:

```bash
python scripts/validate_attention_proxy.py \
  --fit-weights \
  --evaluate-holdout \
  --write-config configs/attribution_calibrated.yaml \
  --write-memo docs/validation/attention_validation_memo.md
```

## Corpus floor

- At least 12 qualifying pages across at least 4 independently trafficked properties
- At least 200 unique human sessions and 50 mapped actionable-element clicks per page
- Public websites without access to their private telemetry are engineering smoke tests,
  not behavioral validation
- Describe a passing result as validated on the observed traffic corpus. Do not call it
  representative of the average population without demographic sampling and weighting.

## Pass thresholds

Configured in `configs/attribution_calibrated.yaml`:

- Spearman ρ ≥ 0.4 (aggregate across sections)
- Top-1 CTA hit rate ≥ 60%
- Report page-macro bootstrap 95% confidence intervals
- NaN or otherwise undefined Spearman values fail the gate

## Stretch: eye-tracking

Map fixation heatmaps to TR-indexed frames for true gaze validation (not required for Clarity-based T2).
