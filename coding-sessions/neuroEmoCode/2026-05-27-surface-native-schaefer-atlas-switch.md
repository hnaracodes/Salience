# Surface-Native Schaefer Atlas Switch (2026-05-27)

## Summary

Switched NeuroEmo ROI training from Nilearn volumetric projection (`volume_projection_fallback`, 2487 filled vertices) to CBIG FreeSurfer5.3 surface `.annot` labels (`surface_annot_cbig`).

## Atlas artifacts

| Check | Result |
| --- | --- |
| `label_source_kind` | `surface_annot_cbig` |
| `source_space` | `fsaverage5_surface_annot_cbig` |
| `filled_unassigned_vertices` | 1743 (medial wall nearest-neighbor fill; down from 2487 projected) |
| `vertex_order_proof.status` | `annot_matches_nilearn_projection` (CBIG pial coords match Nilearn index order 1:1) |
| Archived projected atlas | `scout_data/neuroemo/atlas_archive/projected_2026-05-25/` |

Projection overlap with volumetric labels is ~77% LH / ~74% RH (diagnostic only; surface vs volume parcellation difference, not vertex-order failure).

## Commands (use `.venv`)

```powershell
.venv\Scripts\activate
python scripts/verify_schaefer_annot_alignment.py --data-dir scout_data/atlases/cbig_schaefer2018
python scripts/build_schaefer_vertex_regions.py --n-rois 400 --yeo-networks 7 --data-dir scout_data/atlases/cbig_schaefer2018
python scripts/train_neuroemo_emotion_model.py --train-npz scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz --model-type logistic_saga --temporal-window-trs 10 --temporal-contiguity contiguous --roi-reducers mean,std,mean_abs --exclude-labels neutral --metrics-json scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr_metrics.json --output-model scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr.joblib
```

## Postfix rerun (logistic_saga 5-class, 10-TR contiguous)

Compared to projected-atlas postfix best (`logistic_saga_5class_10tr` mean accuracy **0.445** in `coding-sessions/2026-05-25-neuroemo-postfix-matrix-report.md`):

| Metric | Projected (2026-05-25) | Surface annot (2026-05-27) |
| --- | ---: | ---: |
| Mean accuracy | 0.445 | 0.420 |
| Mean macro F1 | 0.433 | 0.412 |

Metrics path: `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr_metrics.json`

## Code / docs touched

- `scout_core/schaefer_surface_labels.py` — CBIG annot load, mesh coord order proof, pial auto-download
- `scripts/build_schaefer_vertex_regions.py`, `scripts/verify_schaefer_annot_alignment.py`
- `scout_core/parcellation.py`, `scout_core/roi_features.py` — strict surface-native manifest gates
- `docs/atlas-setup/cbig-schaefer2018-fsaverage5.md`
- `tests/test_build_schaefer_surface_annot.py`
