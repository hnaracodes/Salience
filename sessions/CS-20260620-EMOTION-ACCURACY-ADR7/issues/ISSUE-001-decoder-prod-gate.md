---
id: ISSUE-001
title: Decoder production gate — UX validation required
status: open
severity: medium
created: 2026-06-20
---

# Decoder production gate

`emotion.mode` must remain `template` until:

1. `scout_models/horikawa_ridge_v1/` trained on real ds002425 TRIBE features (not `--demo` synthetic).
2. `scripts/validation/aggregate_validation_metrics.py` reports `transfer_r_valence >= 0.25` (`validation_study.yaml`).

Until then, use `emotion.parallel_decoder: true` for shadow `emotion_decoder_track` comparison.
