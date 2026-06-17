# Ledger — CS-20260616-DOCKER-RUNTIME-FIXES

## Step 1 — analyze stage FileNotFoundError

- Symptom: `roi_norms.parquet` path was `C:\Users\...\TribeV2\scout_norms\...` inside Linux worker
- Confirmed: parquet exists at `/app/scout_norms/synthetic_bootstrap_v1/roi_norms.parquet`; SQLite stored Windows absolute paths
- Cause: host `activations.sqlite` copied into image (no `.dockerignore`); `_ensure_norm_bootstrap()` skipped when row existed

## Step 2 — Fixes (norm paths)

- `resolve_storage_path()` — normalize backslashes; map legacy absolute paths via `scout_norms/<norm_id>/<file>`
- `_portable_storage_path()` — store repo-relative POSIX paths in SQLite
- `_ensure_norm_bootstrap()` — verify files exist; re-register portable paths when stale
- `.dockerignore` — exclude `*.sqlite`, `scout_data/sessions/`, etc.
- `tests/test_storage_path_resolution.py` — regression tests

## Step 3 — export_viewer NoSuchBucket

- Symptom: `PutObject` → bucket `scout` missing on MinIO
- Fix: `minio-init` service runs `mc mb local/scout --ignore-existing`
- Fix: `artifacts._ensure_bucket()` creates bucket on local MinIO endpoints only; production R2 must pre-exist

## Step 4 — FAKE_TRIBE clarification

- `docker-compose.yml` sets `FAKE_TRIBE=1` on worker → synthetic zero `preds.npz`, uniform heatmaps, fast scans
- Real Modal GPU inference requires `FAKE_TRIBE=0` + `MODAL_TOKEN_*`

## Validation

- 257+ unit tests passing after path fix
- Runtime: worker logs show norm path repair and bucket creation
