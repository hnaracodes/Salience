# Pipeline Asset Requirements

Assets that must be present (or fetchable) before a scan can run all stages.

## Required at Runtime

| Asset | Expected Path | Notes |
|---|---|---|
| `configs/vertex_regions.csv` | `<project_root>/configs/vertex_regions.csv` | Schaefer parcellation lookup. Maps fsaverage5 vertex indices to region labels and Yeo 7-network assignments. Required by `scout_core/parcellation.py`. |
| `configs/emotion_templates/` | `<project_root>/configs/emotion_templates/` | Kragel emotion template directory. Used by `scout_core/neuroEmoCode/`. Only required when emotion decoding is enabled; pipeline continues without it. |
| `scout_norms/synthetic_bootstrap_v1/` | `<project_root>/scout_norms/synthetic_bootstrap_v1/` | Norm dataset for population-relative engagement scoring. If missing, `marketing_scores.py` falls back to session-relative (min-max) scoring. |

## Optional but Recommended

| Asset | Expected Path | Remote Key | Notes |
|---|---|---|---|
| `preds_baseline.npz` | `scout_data/preds_baseline.npz` | `assets/preds_baseline.npz` (B2) | Baseline prediction array for dual-track Track 1 comparison. If absent, `dual_track.py` logs a warning and runs Track 2 only (session-relative novelty). Fetch from B2 before first production scan: `aws s3 cp s3://<bucket>/assets/preds_baseline.npz scout_data/preds_baseline.npz --endpoint-url $S3_ENDPOINT_URL` |

## Environment Variables (production)

```
S3_ENDPOINT_URL        — Backblaze B2 S3 endpoint (https://s3.<region>.backblazeb2.com)
S3_ACCESS_KEY_ID       — B2 application key ID
S3_SECRET_ACCESS_KEY   — B2 application key secret
S3_BUCKET              — B2 bucket name for scan artifacts
S3_PUBLIC_URL          — Optional public base URL for viewer links (omit for presigned URLs)
REDIS_URL              — Upstash Redis TCP URL (rediss:// recommended)
FAKE_TRIBE=1           — Dev mode: skip Modal calls, use synthetic preds
```

Legacy `R2_*` names are still accepted for local MinIO via docker-compose.
