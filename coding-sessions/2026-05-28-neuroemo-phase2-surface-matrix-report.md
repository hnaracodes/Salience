# NeuroEmo Phase 2 Surface-Native Postfix Matrix Report (2026-05-28)

## Purpose

Complete Phase 2 of `.cursor/plans/neuroemo_remaining_high_leverage_opportunities_2026-05-27.plan.md`: run the full surface-native postfix-equivalent model matrix under a frozen contract and compare against the May 25 projected postfix baseline documented in `coding-sessions/2026-05-25-neuroemo-postfix-matrix-report.md`.

## Frozen contract

| Setting | Value |
| --- | --- |
| Train NPZ | `scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz` |
| Atlas | Schaefer 2018 400 parcels / 7 networks (surface `.annot`) |
| Labels | 5-class; `neutral` excluded |
| Temporal window | 10 TRs, stride 1, contiguous, reducer `mean` |
| ROI reducers | `mean,std,mean_abs` |
| CV | Subject-held-out, 5 folds |
| Vertex proof | `projection_equivalence_verified` |
| Proof hash | `25ced97a324a16f4cbf07261e4e613cbb1897ff792d1af9f92a0235bf7c58a8c` |

Output directory (canonical): `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/`

Runner:

```powershell
.venv\Scripts\python.exe scripts/run_neuroemo_experiment_matrix.py --phase surface-postfix --execute --skip-existing
```

## Completed artifacts (7/7)

| Run | Model | Metrics |
| --- | --- | --- |
| `logistic_saga_5class_10tr` | `.../logistic_saga_5class_10tr.joblib` | `.../logistic_saga_5class_10tr_metrics.json` |
| `mlp_5class_10tr` | `.../mlp_5class_10tr.joblib` | `.../mlp_5class_10tr_metrics.json` |
| `sgd_logistic_5class_10tr` | `.../sgd_logistic_5class_10tr.joblib` | `.../sgd_logistic_5class_10tr.metrics.json` |
| `linear_svc_5class_10tr` | `.../linear_svc_5class_10tr.joblib` | `.../linear_svc_5class_10tr.metrics.json` |
| `specialist_none_5class_10tr` | `.../specialist_none_5class_10tr.joblib` | `.../specialist_none_5class_10tr.metrics.json` |
| `specialist_sigmoid_5class_10tr` | `.../specialist_sigmoid_5class_10tr.joblib` | `.../specialist_sigmoid_5class_10tr.metrics.json` |
| `mlp_valence_binary_10tr` | `.../mlp_valence_binary_10tr.joblib` | `.../mlp_valence_binary_10tr.metrics.json` |

(`...` = `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix`)

### Provenance hashes (surface matrix)

| Field | Value |
| --- | --- |
| Vertex equivalence report | `25ced97a324a16f4cbf07261e4e613cbb1897ff792d1af9f92a0235bf7c58a8c` |
| Proof status | `projection_equivalence_verified` (all 7 metrics JSON files) |
| Parcellation manifest SHA | `e5ed4fcc85590d0d8dcc6c720feb2f983da17bc943b864e58c45d33e956c8779` |
| `vertex_regions.csv` SHA | `de137b365f9de13ba814192fc04458a7ab8666cdd46d02647fe6d35192d027a3` |

The May 27 logistic anchor was retrained once after Phase 1 so its metrics embed `25ced...` instead of the retired mesh-only hash `ebcf93a9...`. Metrics are unchanged (`0.420` / `0.412` macro F1).

## Surface-native results (complete matrix)

| Model | Mean Acc | Mean Bal Acc | Mean Macro F1 | Pooled Log Loss | Δ Acc vs projected |
| --- | ---: | ---: | ---: | ---: | ---: |
| `logistic_saga_5class_10tr` | 0.4200 | 0.4200 | 0.4123 | 2.2437 | −0.0250 |
| `mlp_5class_10tr` | 0.3675 | 0.3675 | 0.3575 | 1.6926 | −0.0675 |
| `sgd_logistic_5class_10tr` | 0.3800 | 0.3800 | 0.3711 | 22.1602 | −0.0350 |
| `linear_svc_5class_10tr` | 0.3600 | 0.3600 | 0.3537 | 1.6391 | −0.0025 |
| `specialist_none_5class_10tr` | 0.3725 | 0.3725 | 0.3671 | 2.1709 | −0.0075 |
| `specialist_sigmoid_5class_10tr` | 0.3600 | 0.3600 | 0.3514 | 1.5088 | +0.0100 |
| `mlp_valence_binary_10tr` | 0.6938 | 0.6938 | 0.6923 | 0.7102 | +0.0344 |

## Projected postfix baseline (May 25, from session report)

Source: `coding-sessions/2026-05-25-neuroemo-postfix-matrix-report.md` (`scout_data/neuroemo/models/2026-05-25_postfix_matrix/`; artifacts not present in this workspace clone).

| Model | Mean Acc | Mean Bal Acc | Mean Macro F1 | Pooled Log Loss |
| --- | ---: | ---: | ---: | ---: |
| `logistic_saga_5class_10tr` | 0.4450 | 0.4450 | 0.4335 | 2.2792 |
| `mlp_5class_10tr` | 0.4350 | 0.4350 | 0.4251 | 1.6324 |
| `sgd_logistic_5class_10tr` | 0.4150 | 0.4150 | 0.4098 | 20.9230 |
| `linear_svc_5class_10tr` | 0.3625 | 0.3625 | 0.3534 | 1.6258 |
| `specialist_none_5class_10tr` | 0.3800 | 0.3800 | 0.3697 | 2.1520 |
| `specialist_sigmoid_5class_10tr` | 0.3500 | 0.3500 | 0.3181 | 1.4924 |
| `mlp_valence_binary_10tr` | 0.6594 | 0.6594 | 0.6557 | 0.7109 |

## Per-class pooled metrics (surface-native, best 5-class logistic)

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| calm | 0.392 | 0.388 | 0.390 |
| afraid | 0.316 | 0.300 | 0.308 |
| delighted | 0.509 | 0.363 | 0.423 |
| depressed | 0.439 | 0.588 | 0.503 |
| excited | 0.457 | 0.463 | 0.460 |

Hard classes on surface-native logistic remain `afraid` (lowest recall) and `delighted` (high precision, low recall). `depressed` is over-predicted (recall 0.59). Per-class tables for other surface runs are in each `*_metrics.json` under `cross_validation.classification_report`.

Projected postfix per-class breakdowns were not archived in the May 25 report; treat class-level deltas as qualitative only.

## Binary valence per-class (surface `mlp_valence_binary_10tr`)

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| negative | 0.678 | 0.738 | 0.707 |
| positive | 0.712 | 0.650 | 0.680 |

## Headline comparison

| Metric family | Surface vs projected | Interpretation |
| --- | --- | --- |
| 5-class top-1 / macro F1 | Logistic: −2.5 pp acc, −2.1 pp macro F1 | Surface-native does **not** beat projected on discrimination |
| 5-class calibration | Logistic pooled log loss slightly better (2.24 vs 2.28) | Minor calibration edge; not enough to promote |
| Binary valence | +3.4 pp acc / macro F1 vs projected | Only clear metric win in this matrix |
| MLP 5-class | −6.8 pp acc vs projected MLP | Neural baseline degrades more on surface |
| Specialist sigmoid | +1.0 pp acc vs projected | Small top-1 gain; macro F1 still below logistic on both atlases |

## Conclusion

**Verdict: provenance tradeoff, not a metric win for 5-class modeling.**

- Surface-native Schaefer with `projection_equivalence_verified` (`25ced...`) is the correct default for **geometry trust** and downstream TRIBE parity.
- For **5-class emotion classification**, projected postfix remains stronger on mean accuracy and macro F1; no surface-native run beats projected `logistic_saga_5class_10tr` at 0.445 / 0.433 macro F1.
- Binary valence (`mlp_valence_binary_10tr`) is the one family where surface-native clearly improves over the documented projected baseline.
- Do **not** replace the projected postfix 5-class production default until a later phase (timing grid, dynamic features, or structured decoding) shows a stable macro-F1 gain.

## Execution notes and failures

| Event | Outcome |
| --- | --- |
| First `run_neuroemo_experiment_matrix.py --phase surface-postfix --execute` | Completed `sgd_logistic` and `linear_svc`; **failed** on `mlp_5class_10tr` with `ModuleNotFoundError: No module named 'scripts'` before `sys.path` bootstrap was present in `train_neuroemo_mlp_model.py` |
| Resume with `--skip-existing` | All 7 runs present; MLP/specialist/valence completed in a follow-up pass |
| Logistic retrain | Deleted stale `logistic_saga_5class_10tr_metrics.json` (mesh-only proof) and reran to embed `25ced...`; metrics unchanged |
| Metrics naming | Runner accepts both `{name}_metrics.json` and `{name}.metrics.json` |

## Engineering notes

- Phase 2 runner: `scripts/run_neuroemo_experiment_matrix.py --phase surface-postfix --execute --skip-existing`.
- MLP/specialist/hierarchical trainers import `scripts.train_neuroemo_emotion_model` after inserting `PROJECT_ROOT` on `sys.path`; matrix subprocesses also set `PYTHONPATH`.

## Next steps (phases 3–6)

1. Timing grid on surface train NPZ with frozen logistic first.
2. Dynamic temporal reducer bundles vs static `mean`.
3. Raw vs preprocessed and ROI fill strategy A/B.
4. Hierarchical coarse-to-fine vs flat 5-way.
