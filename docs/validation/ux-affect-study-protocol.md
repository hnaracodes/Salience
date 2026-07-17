# UX Affect Validation Study Protocol

## Purpose

Measure whether Horikawa-trained decoder outputs on website walkthroughs correlate with
human self-report (SAM + UX items). Required before setting `emotion.mode: decoder` as default.

## Participants

- Pilot: N >= 10
- Full study: N >= 20

## Task

1. Browse fixed test sites using production explore script (`configs/explore_production.yaml`).
2. After each major section (or full session), complete SAM (valence, arousal, dominance).
3. Rate UX-specific items: confusion, frustration, trust (0–1 sliders).

## Artifacts

- Standard pipeline outputs under `scout_data/sessions/<id>/` including `preds_subcortical.npz`.
- Ratings under `scout_data/validation/sessions/<id>/ratings.json`.

## Go / no-go

Run `scripts/validation/aggregate_validation_metrics.py`. If `transfer_r_valence` < 0.25
(configurable in `configs/validation_study.yaml`), keep `emotion.mode: template` as default
and label decoder output experimental in the UI.

## Ethics

Non-diagnostic research prototype. Do not present scores as literal emotion measurement.
