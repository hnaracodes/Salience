# NeuroEmo Surface-Native Schaefer Atlas Evaluation (2026-05-27)

This document records **all reported statistics** from the surface-native Schaefer atlas switch and the paired **postfix-equivalent** emotion-classification rerun. Each metric is labeled, sourced, and interpreted so results can be compared to the prior volumetric-projection atlas without opening raw JSON.

---

## 1. Experiment identification

| Field | Value |
| --- | --- |
| **Evaluation date** | 2026-05-27 |
| **Primary hypothesis** | CBIG FreeSurfer5.3 surface `.annot` labels improve ROI representation vs Nilearn volumetric projection |
| **Primary model run** | `logistic_saga_5class_10tr` (subject-held-out 5-fold CV + final fit) |
| **Metrics artifact** | `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr_metrics.json` |
| **Model artifact** | `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr.joblib` |
| **Atlas outputs** | `configs/vertex_regions.csv`, `configs/parcellation_manifest.yaml` |
| **Alignment report** | `scout_data/neuroemo/schaefer_annot_alignment_report.json` |
| **Archived prior atlas** | `scout_data/neuroemo/atlas_archive/projected_2026-05-25/` |
| **Comparison baseline** | `logistic_saga_5class_10tr` from `scout_data/neuroemo/models/2026-05-25_postfix_matrix/` (projected atlas, documented in `coding-sessions/2026-05-25-neuroemo-postfix-matrix-report.md`) |

**Training command (`.venv` required):**

```powershell
.venv\Scripts\activate
python scripts/train_neuroemo_emotion_model.py `
  --train-npz scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz `
  --model-type logistic_saga `
  --temporal-window-trs 10 --temporal-contiguity contiguous `
  --roi-reducers mean,std,mean_abs `
  --exclude-labels neutral `
  --metrics-json scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr_metrics.json `
  --output-model scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr.joblib
```

---

## 2. Atlas validation statistics

These numbers confirm the **atlas pipeline** succeeded before training—not classifier performance.

### 2.1 Parcellation manifest (surface-native)

| Statistic | Projected atlas (archived 2026-05-25) | Surface-native atlas (2026-05-27) | Interpretation |
| --- | ---: | ---: | --- |
| `label_source_kind` | `volume_projection_fallback` | `surface_annot_cbig` | Training now uses CBIG `.annot` as ground truth |
| `source_space` | `MNI152_volume_projected_to_fsaverage5` | `fsaverage5_surface_annot_cbig` | No volumetric resampling step in label path |
| `filled_unassigned_vertices` | **2487** | **1743** | Fewer vertices needed nearest-neighbor fill (−744, −29.9%) |
| `n_vertices_expected` | 20484 | 20484 | Unchanged; still full fsaverage5 LH+RH |
| `n_parcels` | 400 | 400 | Schaefer 400 / 7-networks |
| `vertex_regions.csv` SHA256 | *(archived copy)* | `de137b365f9de13ba814192fc04458a7ab8666cdd46d02647fe6d35192d027a3` | Pins exact vertex→parcel table used in this run |
| `parcellation_manifest.yaml` SHA256 | *(archived copy)* | `1743adb1b8b6379332e4e4a58c2b8ca80bbb7c248e34cdac90bdc8f0e7daaf7e` | Pins manifest used in this run |

**Analysis — `filled_unassigned_vertices`:** The surface atlas still leaves **1743** vertices unlabeled in the raw `.annot` (mostly medial wall). Those are filled by nearest assigned neighbor within hemisphere before CSV write. The drop from **2487 → 1743** removes a large block of projection-induced “synthetic” parcel assignments, but medial-wall fill remains expected for FreeSurfer surface parcellations.

### 2.2 Vertex order proof (`schaefer_annot_alignment_report.json`)

| Statistic | LH | RH | Meaning |
| --- | ---: | ---: | --- |
| `vertex_order_proof.status` | — | — | `annot_matches_nilearn_projection` |
| `lh_mesh_vertex_order.identity_order` | **true** | — | CBIG `lh.pial` vertex index *i* matches Nilearn fsaverage5 index *i* (coord distance 0) |
| `rh_mesh_vertex_order.identity_order` | — | **true** | Same for right hemisphere |
| `max_distance` (mesh coord map) | 0.0 mm | 0.0 mm | No reorder permutation required |
| `reorder_applied` | — | — | **false** |
| `lh_projection_agreement_pct` | **76.67%** | — | Diagnostic: parcel ID match vs one-time volumetric projection |
| `rh_projection_agreement_pct` | — | **74.30%** | Diagnostic only |
| `mean_projection_agreement_pct` | **75.49%** | — | Below 85% threshold by design for *projection* check |
| `projection_agreement_threshold` | 0.85 | — | Not used as pass/fail when mesh coords align 1:1 |

**Analysis:** The ~**75%** overlap with volumetric projection is **not** a vertex-order failure. CBIG and Nilearn share identical pial coordinates per vertex index; disagreement reflects **different parcellation methods** (surface labels vs `vol_to_surf`), not misaligned BOLD stacks. Training NPZ vertex order (`lh_then_rh_fsaverage5`) remains valid.

### 2.3 Vertex equivalence (TribeV2 mesh contract)

| Statistic | Value |
| --- | --- |
| `proof_status` | `mesh_identity_verified` |
| `reference_kind` | `upstream_source_mirror` |
| `mesh` | `fsaverage5` |
| `vertex_order` | `lh_then_rh_fsaverage5` |
| `report_sha256` | `25ced97a324a16f4cbf07261e4e613cbb1897ff792d1af9f92a0235bf7c58a8c` |

**Analysis:** Nilearn fsaverage5 meshes match the mirrored TribeV2 reference fingerprints (pial/white, LH/RH). Surface BOLD in `neuroemo_tribev2_train.npz` is therefore aligned with ROI aggregation indices.

---

## 3. Training protocol and dataset statistics

### 3.1 Model and feature construction

| Statistic | Value | Explanation |
| --- | ---: | --- |
| `model_type` | `logistic_saga` | Multinomial logistic regression (SAGA solver) |
| `feature_mode` | `roi` | Vertices aggregated to Schaefer parcels before classification |
| `input_feature_shape` | 20484 | Surface vertices per sample (fsaverage5) |
| `n_parcels` / ROIs | 400 | Schaefer 2018 parcels |
| `roi_reducers` | `mean`, `std`, `mean_abs` | Three summaries per parcel → **1200** features (400 × 3) |
| `n_features` | **1200** | Model input dimension after ROI reduction |
| `scale` | true | Features standardized before fitting |
| `max_iter` | 3000 | Solver iteration cap |
| `c_value` | 1.0 | Inverse regularization strength |
| `random_state` | 13 | Reproducibility seed |
| `elapsed_final_fit_seconds` | **13.16** | Wall time for final model on all 400 windows |

### 3.2 Labels, subjects, and temporal windows

| Statistic | Value | Explanation |
| --- | ---: | --- |
| `train_npz` | `scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz` | Unchanged BOLD prep from prior postfix runs |
| `n_subjects` | **40** | NeuroEmo subjects in CV |
| `exclude_labels` | `neutral` | Synthetic/neutral class dropped |
| `labels` (5 classes) | calm, afraid, delighted, depressed, excited | Original discrete emotion set |
| `label_filter.input_samples` | **4000** | TR-level rows before windowing |
| `label_filter.output_samples` | **4000** | No TRs dropped by label filter |
| `temporal_aggregation.window_trs` | **10** | Each training example = 10 consecutive TRs |
| `temporal_aggregation.stride_trs` | 1 | Sliding window step |
| `temporal_aggregation.contiguity` | `contiguous` | Windows do not cross TR gaps |
| `temporal_aggregation.input_samples` | 4000 | TRs entering windowing |
| `temporal_aggregation.output_samples` | **400** | Training windows after aggregation |
| `class_counts` (per class) | **80** each | Balanced 5-way design after windowing |
| `n_samples` (final training set) | **400** | Matches 40 subjects × 10 windows/subject (balanced) |

**Analysis:** The experimental **unit of analysis** is a 10-TR contiguous window (~20 s at typical TR), not a single TR. Chance accuracy for balanced 5-class is **20%**; all reported accuracies should be read against that floor.

### 3.3 Cross-validation design

| Statistic | Value |
| --- | --- |
| `cross_validation.skipped` | false |
| `cross_validation.n_splits` | **5** |
| Split type | **Subject-held-out** (all windows from test subjects only in test fold) |
| `n_train` per fold | **320** windows (32 subjects) |
| `n_test` per fold | **80** windows (8 subjects) |

**Analysis:** Subject grouping limits inflation from within-subject temporal correlation; fold variance (σ ≈ 0.053 on accuracy) reflects subject difficulty more than random window noise.

---

## 4. Cross-validation results (by fold)

Fold-level metrics from `cross_validation.folds[]`.

| Fold | Test subjects (n=8) | Accuracy | Balanced accuracy | Macro F1 | Log loss | Fit time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | sub-05, sub-10, sub-15, sub-20, sub-25, sub-30, sub-35, sub-40 | **0.500** | 0.500 | 0.491 | 2.043 | 10.47 |
| 2 | sub-04, sub-09, sub-14, sub-19, sub-24, sub-29, sub-34, sub-39 | **0.363** | 0.363 | 0.351 | 2.644 | 8.80 |
| 3 | sub-03, sub-08, sub-13, sub-18, sub-23, sub-28, sub-33, sub-38 | **0.400** | 0.400 | 0.389 | 2.199 | 9.76 |
| 4 | sub-02, sub-07, sub-12, sub-17, sub-22, sub-27, sub-32, sub-37 | **0.375** | 0.375 | 0.372 | 2.326 | 9.84 |
| 5 | sub-01, sub-06, sub-11, sub-16, sub-21, sub-26, sub-31, sub-36 | **0.463** | 0.463 | 0.458 | 2.008 | 11.02 |

### 4.1 CV summary statistics (mean ± std across folds)

| Metric | Surface-native (2026-05-27) | Projected atlas baseline (2026-05-25) | Δ (surface − projected) |
| --- | ---: | ---: | ---: |
| **Mean accuracy** | **0.420** ± 0.053 | **0.445** ± *(not in summary)* | **−0.025** |
| **Mean balanced accuracy** | **0.420** ± 0.053 | **0.445** | **−0.025** |
| **Mean macro F1** | **0.412** ± 0.054 | **0.433** | **−0.021** |
| **Mean log loss** | **2.244** | **2.279** | **−0.035** (slightly better calibration) |
| **Pooled log loss** | **2.244** | **2.279** | **−0.035** |

**Analysis — fold spread:** Accuracy ranges **0.363–0.500** across folds (spread 0.137). Fold 2 is the weakest; fold 1 the strongest. High between-fold variance is typical when *n*=8 test subjects per fold and emotions are hard to separate at 5-way granularity.

**Analysis — vs baseline:** Surface-native labels **did not improve** top-1 accuracy vs the projected-atlas postfix best on this single matched run (−2.5 percentage points mean accuracy). Log loss improved slightly, suggesting probability mass is somewhat better calibrated even when argmax accuracy is lower.

---

## 5. Pooled CV metrics (all test predictions combined)

These aggregate predictions from all five test folds into one confusion matrix (400 test windows total).

| Metric | Value | Interpretation |
| --- | ---: | --- |
| **Pooled accuracy** | **0.420** | 168 / 400 windows correct |
| **Pooled balanced accuracy** | **0.420** | Same as accuracy here because each class has 80 test windows |
| **Pooled macro F1** | **0.417** | Harmonic mean of per-class F1, unweighted |
| **Pooled log loss** | **2.244** | Lower is better; random baseline ≈ ln(5) ≈ 1.61 for perfect calibration, higher when confused |

### 5.1 Confusion matrix (pooled, rows = true, columns = predicted)

Class order: **calm**, **afraid**, **delighted**, **depressed**, **excited**.

| True \ Pred | calm | afraid | delighted | depressed | excited | Row total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **calm** | 31 | 24 | 2 | 18 | 5 | 80 |
| **afraid** | 24 | 24 | 6 | 13 | 13 | 80 |
| **delighted** | 8 | 6 | 29 | 19 | 18 | 80 |
| **depressed** | 5 | 12 | 8 | 47 | 8 | 80 |
| **excited** | 11 | 10 | 12 | 10 | 37 | 80 |

**Analysis — confusion patterns:**

- **Depressed** is the most recognizable class (**47/80** correct, recall **58.8%**); many **afraid** and **calm** windows are called depressed (13 and 18 false positives into depressed from other rows).
- **Afraid** is the hardest class (**24/80** correct, recall **30.0%**); calm↔afraid confusion is symmetric (24 each way).
- **Delighted** has moderate recall (**36.3%**) but bleeds into **depressed** (19) and **excited** (18)—valence/arousal ambiguity.
- **Excited** is mid-tier (**46.3%** recall); confused with all classes somewhat evenly.
- **Calm** recall **38.8%**; often predicted as afraid (24) or depressed (18).

Overall, the model captures **broad negative affect (depressed)** better than **fear (afraid)** or **positive high-arousal (delighted/excited)** under 5-way Schaefer-ROI features.

---

## 6. Per-class classification report (pooled CV)

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| **calm** | 0.392 | 0.388 | 0.390 | 80 |
| **afraid** | 0.316 | 0.300 | 0.308 | 80 |
| **delighted** | 0.509 | 0.363 | 0.423 | 80 |
| **depressed** | 0.439 | 0.588 | 0.503 | 80 |
| **excited** | 0.457 | 0.463 | 0.460 | 80 |
| **Macro avg** | 0.423 | 0.420 | 0.417 | 400 |
| **Weighted avg** | 0.423 | 0.420 | 0.417 | 400 |

**Analysis:**

- **Highest F1:** depressed (0.503) — consistent with confusion matrix diagonal.
- **Lowest F1:** afraid (0.308) — fear is under-detected; many false negatives become calm or depressed.
- **Precision vs recall:** delighted has the highest precision (0.509) but low recall (0.363)—when the model says delighted it is modestly trustworthy, but it misses most delighted windows.

---

## 7. Automated test statistics (pipeline validation)

Unit tests run after the atlas switch (`.venv`):

```text
pytest tests/test_build_schaefer_surface_annot.py tests/test_neuroemo_training.py
→ 21 passed, 2 skipped
```

| Test area | What it validates |
| --- | --- |
| `test_build_schaefer_surface_annot.py` | Annot remapping, manifest gates for `surface_annot_cbig` vs legacy projection |
| `test_neuroemo_training.py` | Training contract: vertex order, manifest SHA, temporal metadata, vertex equivalence |

**Analysis:** Tests confirm **engineering correctness** (manifest rejection, remapping, contract enforcement). They do not assert minimum classification accuracy.

---

## 8. Integrated analysis and conclusions

### 8.1 What improved with the atlas switch

1. **Label provenance:** ROI features now derive from published CBIG surface parcellation, not `vol_to_surf` nearest-neighbor voting.
2. **Fewer synthetic fills:** **2487 → 1743** filled vertices (−29.9%), reducing projection artifacts on unassigned cortex.
3. **Vertex order verified:** Mesh-coordinate proof passes; TribeV2/Nilearn vertex indexing is consistent with `.annot`.
4. **Slightly better log loss:** **2.244** vs **2.279** on the matched logistic run—marginal probability-quality gain.

### 8.2 What did not improve (on this run)

1. **5-class accuracy / macro F1:** Mean accuracy **0.420** vs projected **0.445** (−0.025); macro F1 **0.412** vs **0.433**.
2. **Per-class fear detection:** Afraid remains the weakest class (F1 **0.308**).

### 8.3 Plausible explanations

| Factor | Role |
| --- | --- |
| **Representation change ≠ immediate gain** | Surface labels change parcel boundaries; linear ROI summaries may need re-tuning (reducers, regularization, or nonlinear model). |
| **Task difficulty ceiling** | 5-way emotion on 40 subjects with 10-TR windows is near chance+; small atlas gains are easily drowned by subject variance. |
| **Medial-wall fill** | 1743 filled vertices still inject neighbor parcel signal into non-cortical vertices. |
| **Same BOLD, new ROIs** | NPZ surface time series unchanged; only ROI aggregation changed—limits expected uplift to ROI-relevant signal. |

### 8.4 Recommended next steps (not part of this run)

- Rerun full **postfix matrix** (MLP, SGD, specialists) on surface-native atlas for fair multi-model comparison.
- Report **balanced accuracy and macro F1** as primary metrics (class-balanced design).
- Consider **binary valence** or **hierarchical** models if 5-way ceiling persists.
- Optionally exclude medial-wall vertices from ROI reducers instead of nearest-neighbor fill.

---

## 9. Source artifact index

| Artifact | Path |
| --- | --- |
| Training metrics | `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr_metrics.json` |
| Trained model | `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr.joblib` |
| Annot alignment | `scout_data/neuroemo/schaefer_annot_alignment_report.json` |
| Parcellation manifest | `configs/parcellation_manifest.yaml` |
| Vertex table | `configs/vertex_regions.csv` |
| Projected atlas archive | `scout_data/neuroemo/atlas_archive/projected_2026-05-25/` |
| Baseline postfix report | `coding-sessions/2026-05-25-neuroemo-postfix-matrix-report.md` |
| Session log | `coding-sessions/2026-05-27-surface-native-schaefer-atlas-switch.md` |
| Atlas setup | `docs/atlas-setup/cbig-schaefer2018-fsaverage5.md` |

---

*NeuroEmo-derived outputs are model-assisted emotion-state hypotheses, not clinical measurements.*
