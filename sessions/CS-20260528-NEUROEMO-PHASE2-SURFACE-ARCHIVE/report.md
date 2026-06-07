# NeuroEmo Phase 2 Surface-Native Postfix Matrix Report (2026-05-28)

**Session ID:** `CS-20260528-NEUROEMO-PHASE2-SURFACE-ARCHIVE`  
**Agent signature:** `Historical Cursor coding agent`  
**Context contract:** Future agents may cite this session ID when asking about the archived NeuroEmo Phase 2 surface-native postfix-equivalent model matrix and frozen contract.  
**Metadata normalized by:** `GPT-5.5, 2026-06-07`

## Purpose

Complete Phase 2 of `.cursor/plans/neuroemo_remaining_high_leverage_opportunities_2026-05-27.plan.md`: run the full surface-native postfix-equivalent model matrix under a frozen contract and compare against the May 25 projected postfix baseline.

## Frozen contract

| Setting | Value |
| --- | --- |
| Train NPZ | `scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz` |
| Atlas | Schaefer 2018 400 parcels / 7 networks (surface annot) |
| Labels | 5-class; `neutral` excluded |
| Temporal window | 10 TRs, stride 1, contiguous, reducer `mean` |
| ROI reducers | `mean,std,mean_abs` |
| CV | Subject-held-out, 5 folds |
| Vertex proof | `projection_equivalence_verified` (`25ced97a324a16f4cbf07261e4e613cbb1897ff792d1af9f92a0235bf7c58a8c`) |

Output directory: `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/`

## Surface-native results (complete matrix)

| Model | Mean Acc | Mean Bal Acc | Mean Macro F1 | Pooled Log Loss |
| --- | ---: | ---: | ---: | ---: |
| `logistic_saga_5class_10tr` | 0.4200 | 0.4200 | 0.4123 | 2.2437 |
| `mlp_5class_10tr` | 0.3675 | 0.3675 | 0.3575 | 1.6926 |
| `sgd_logistic_5class_10tr` | 0.3800 | 0.3800 | 0.3711 | 22.1602 |
| `linear_svc_5class_10tr` | 0.3600 | 0.3600 | 0.3537 | 1.6391 |
| `specialist_none_5class_10tr` | 0.3725 | 0.3725 | 0.3671 | 2.1709 |
| `specialist_sigmoid_5class_10tr` | 0.3600 | 0.3600 | 0.3514 | 1.5088 |
| `mlp_valence_binary_10tr` | 0.6938 | 0.6938 | 0.6923 | 0.7102 |

## Projected postfix baseline (May 25, from session report)

| Model | Mean Acc | Mean Bal Acc | Mean Macro F1 | Pooled Log Loss |
| --- | ---: | ---: | ---: | ---: |
| `logistic_saga_5class_10tr` | 0.4450 | 0.4450 | 0.4335 | 2.2792 |
| `mlp_5class_10tr` | 0.4350 | 0.4350 | 0.4251 | 1.6324 |
| `sgd_logistic_5class_10tr` | 0.4150 | 0.4150 | 0.4098 | 20.9230 |
| `linear_svc_5class_10tr` | 0.3625 | 0.3625 | 0.3534 | 1.6258 |
| `specialist_none_5class_10tr` | 0.3800 | 0.3800 | 0.3697 | 2.1520 |
| `specialist_sigmoid_5class_10tr` | 0.3500 | 0.3500 | 0.3181 | 1.4924 |
| `mlp_valence_binary_10tr` | 0.6594 | 0.6594 | 0.6557 | 0.7109 |

## Headline comparison

| Metric family | Surface vs projected (best 5-class) | Interpretation |
| --- | --- | --- |
| 5-class top-1 / macro F1 | Logistic: −2.5 pp acc, −2.1 pp macro F1 | Surface-native does **not** beat projected on discrimination |
| 5-class calibration | Logistic pooled log loss slightly better (2.24 vs 2.28) | Minor calibration edge; not enough to promote |
| Binary valence | +3.4 pp acc / macro F1 vs projected | Valence endpoint improves on surface atlas |
| MLP 5-class | −6.8 pp acc vs projected MLP | Neural baseline degrades more on surface |

## Conclusion

**Verdict: provenance tradeoff, not a metric win for 5-class modeling.**

- Surface-native Schaefer with `projection_equivalence_verified` is the correct default for **geometry trust** and downstream TRIBE parity.
- For **5-class emotion classification**, projected postfix remains stronger on mean accuracy and macro F1; no surface-native run in this matrix beats `logistic_saga_5class_10tr` at 0.445 / 0.433 macro F1.
- Binary valence (`mlp_valence_binary_10tr`) is the one family where surface-native clearly improves over the documented projected baseline.
- Do **not** replace the projected postfix 5-class production default until a later phase (timing grid, dynamic features, or structured decoding) shows a stable macro-F1 gain.

## Engineering notes

- Fixed `ModuleNotFoundError: No module named 'scripts'` in MLP/specialist/hierarchical trainers by inserting `PROJECT_ROOT` on `sys.path` before package imports.
- Phase 2 runner: `scripts/run_neuroemo_experiment_matrix.py --phase surface-postfix --execute --skip-existing`.

## Next steps (phases 3–6)

1. Timing grid on surface train NPZ with frozen logistic first.
2. Dynamic temporal reducer bundles vs static `mean`.
3. Raw vs preprocessed and ROI fill strategy A/B.
4. Hierarchical coarse-to-fine vs flat 5-way.
