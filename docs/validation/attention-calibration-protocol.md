# Attention calibration protocol

## Goal

Validate that element `combined_score` rankings correlate with behavioral telemetry (Clarity click exports) before claiming T2 attention proxy language.

## Steps

1. Capture session with Playwright (`explore` or `scripted`).
2. Run full pipeline through `analyze_session.py --website`.
3. Export Clarity click CSV for the same URL/time window.
4. Place CSV at `scout_data/sessions/<id>/clarity_clicks.csv` or pass `--clarity-csv`.
5. Run `python scripts/validate_attention_proxy.py --write-memo docs/validation/attention_validation_memo.md`.

## Pass thresholds

Configured in `configs/attribution_calibrated.yaml`:

- Spearman ρ ≥ 0.4 (aggregate across sections)
- Top-1 CTA hit rate ≥ 60%

## Stretch: eye-tracking

Map fixation heatmaps to TR-indexed frames for true gaze validation (not required for Clarity-based T2).
