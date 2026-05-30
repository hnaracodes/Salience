# NeuroEmo code archive

Supervised emotion-classifier training on OpenNeuro **NeuroEmo** (ds005700), Schaefer ROI extraction, and experiment matrices.

**Not part of the main website E2E pipeline.** Production website sessions use:

- `scripts/record_website_session.py` → `tribe.py::record_session` → `run_dual_track.py` (Kragel template emotion Z-scores, not NeuroEmo ML)
- `analyze_session.py` → section analytics → UX viewer

## Data

Artifacts live under [`scout_data/neuroEmoCode/`](../../scout_data/neuroEmoCode/) (raw BIDS, prepared NPZ, models, vertex-equivalence report).

## Entry points

| Script | Purpose |
|--------|---------|
| `prepare_neuroemo_tribev2.py` | BIDS → TribeV2-shaped surface NPZ |
| `train_neuroemo_emotion_model.py` | Linear / logistic ROI classifiers |
| `train_neuroemo_specialist_models.py` | Per-class specialists |
| `train_neuroemo_mlp_model.py` | MLP baseline |
| `train_neuroemo_hierarchical_model.py` | Hierarchical decode |
| `run_neuroemo_experiment_matrix.py` | Batch experiment manifest |
| `build_schaefer_vertex_regions.py` | Atlas CSV + manifest |
| `verify_tribe_vertex_equivalence.py` | Vertex-order proof (also `modal run tribe.py::verify_vertex_equivalence`) |

## Tests

```powershell
python -m pytest tests/neuroEmoCode/ -v --tb=short
```

Default `pytest tests/` skips this folder (see `tests/conftest.py`).
