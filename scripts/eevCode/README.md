# EEV code archive

Supervised **continuous evoked-expression** regression on Google's **EEV** dataset, using TRIBE v2 cortical predictions and Schaefer/Yeo network features.

**Not part of the main website E2E pipeline.** Production website sessions still use:

- `scripts/record_website_session.py` → `tribe.py::record_session` → `run_dual_track.py` (Kragel template emotion Z-scores)
- `analyze_session.py` → section analytics → UX viewer

NeuroEmo (`neuroEmoCode/`) and EEV (`eevCode/`) are **parallel research tracks** — neither replaces production emotion scoring unless explicitly promoted in a future plan.

## Data

Artifacts live under [`scout_data/eevCode/`](../../scout_data/eevCode/):

| Path | Contents |
|------|----------|
| `csv/` | EEV `train.csv`, `val.csv` |
| `videos/` | Downloaded `{video_id}.mp4` |
| `intermediates/` | `{video_id}_features.npz` (one-time TRIBE cache) |
| `aligned/` | Combined `X`, `Y` training matrices |
| `models/` | SVR joblib + metrics + manifest |
| `manifests/` | Pilot lists, download status |

## Entry points

| Script | Purpose |
|--------|---------|
| `download_videos.py` | yt-dlp download + pilot manifest builder |
| `generate_features.py` | Modal TRIBE → network feature NPZ |
| `build_aligned_dataset.py` | EEV labels ↔ features, lag search |
| `train_svr.py` | Multi-Output SVR training |
| `evaluate_svr.py` | LOVO metrics + optional session preds eval |

## Quick start (pilot)

```powershell
# 1. Copy EEV CSVs into scout_data/eevCode/csv/ (from Google GitHub repo)

# 2. Build pilot manifest + download videos
python scripts/eevCode/download_videos.py --create-pilot --download

# 3. TRIBE inference (Modal GPU, once per video)
python scripts/eevCode/generate_features.py --video-ids-file scout_data/eevCode/manifests/pilot_50.json --execute-tribe --skip-existing

# 4. Align + train
python scripts/eevCode/build_aligned_dataset.py --split train --search-lag
python scripts/eevCode/train_svr.py
python scripts/eevCode/evaluate_svr.py --mode lovo
```

## Tests

```powershell
python -m pytest tests/eevCode/ -v --tb=short
```

Default `pytest tests/` skips this folder (see `tests/conftest.py`).

## Caveats

- **One Modal worker only.** Never run two `generate_features.py --execute-tribe` processes at once — Modal cancels in-flight inputs when the client disconnects or a duplicate worker starts (`Successfully canceled input` in logs).
- EEV labels are **estimated viewer facial expressions**, not ground-truth emotion or engagement.
- YouTube video availability varies; download manifest tracks failures.
- TRIBE inference requires Modal + HuggingFace token (same as website pipeline).
