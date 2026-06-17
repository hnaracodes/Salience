---
session_id: CS-20260616-DOCKER-RUNTIME-FIXES
title: Docker local stack runtime fixes
date: 2026-06-16
model: composer-2.5
signature: composer-2.5@CS-20260616-DOCKER-RUNTIME-FIXES
status: completed
track: saas
related:
  - CS-20260612-SAAS-PRODUCTION
  - CS-20260616-CI-REQUIREMENTS
  - CS-20260613-LOCAL-PRODUCT-INTEGRATION
agent_transcripts:
  - dc0935c5-4f40-4bfa-8cf8-071bd9103c4d
files_touched:
  - scout_core/storage_migrations.py
  - services/api/jobs.py
  - services/pipeline/artifacts.py
  - scripts/bootstrap_norms.py
  - docker-compose.yml
  - .dockerignore
  - tests/test_storage_path_resolution.py
  - tests/test_artifacts_bucket.py
---

# Docker local stack runtime fixes

## Summary

Fixed two production-blocker failures when running scans via `docker compose`: analyze stage could not find norm parquet files (Windows absolute paths baked into SQLite from host), and export_viewer failed with MinIO `NoSuchBucket`. Documented that `FAKE_TRIBE=1` in compose uses synthetic TRIBE outputs for fast local dev.

## Important Files

- `scout_core/storage_migrations.py` — `resolve_storage_path()`, POSIX `_portable_storage_path()`
- `services/api/jobs.py` — `_ensure_norm_bootstrap()` repairs portable norm paths on worker startup
- `services/pipeline/artifacts.py` — `_ensure_bucket()` auto-creates MinIO bucket in dev
- `docker-compose.yml` — `minio-init` service creates `scout` bucket
- `.dockerignore` — excludes host `*.sqlite` and session artifacts from images

## Validation

- Worker container: norm paths resolve to `/app/scout_norms/.../roi_norms.parquet`
- MinIO: `scout` bucket exists; `_ensure_bucket` succeeds
- `tests/test_storage_path_resolution.py`, `tests/test_artifacts_bucket.py` pass

## Signature

Signed-off-by: Composer 2.5 (composer-2.5) on 2026-06-16
