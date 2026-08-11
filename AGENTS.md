# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Neural-UX Scout — a neuroscience-powered UX analysis platform. Uses Facebook's Tribe v2 model to predict cortical activations from video, then parcellates, z-scores against norms, and applies calibrated threshold rules to generate UX insight hypotheses.

### Architecture

- **`scout_core/`** — Pure Python analysis library (parcellation, aggregation, z-scoring, threshold engine). No external services required.
- **`activation_store.py`** — Persistence layer: saves cortical predictions as `.npz` + summary/peak data to SQLite (`scout_data/activations.sqlite`).
- **`tribe.py`** — Modal-based GPU inference (A100). Requires Modal account + `HF_TOKEN` secret. **Not needed for local dev/testing.**
- **`scripts/`** — Offline pipeline scripts (build vertex CSV, bootstrap norms, compute norms, analyze sessions).
- **`configs/`** — YAML rules, insight catalog, vertex regions CSV (20,484 vertices demo data ships in repo).

### Running tests

```
python3 -m pytest tests/ -v
```

All tests are in `tests/test_scout_core.py` and cover parcellation, network aggregation, z-scoring, and threshold rule evaluation.

### Running the offline analysis pipeline

1. Bootstrap synthetic norms (only needed once, for pipeline validation):
   ```
   python3 scripts/bootstrap_norms.py --norm-id synthetic_bootstrap_v1
   ```
2. Create a session (requires real Tribe predictions or synthetic data via `activation_store.save_cortical_timeseries`).
3. Analyze a session:
   ```
   python3 scripts/analyze_session.py --session-id <ID> --norm-id synthetic_bootstrap_v1
   ```

### Key gotchas

- `pytest` and other scripts install to `~/.local/bin` which may not be on PATH. Use `python3 -m pytest` instead of bare `pytest`.
- Modal (`tribe.py`) requires a Modal account and `HF_TOKEN` secret — skip this for local development. The entire `scout_core` test suite and offline analysis pipeline work without Modal/GPU.
- SQLite database is auto-created at `scout_data/activations.sqlite` on first use. No manual DB setup required.
- The `configs/vertex_regions.csv` ships with 20,484 synthetic demo vertices. Replace with real atlas data for production use.
