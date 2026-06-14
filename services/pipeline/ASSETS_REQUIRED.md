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
| `preds_baseline.npz` | `scout_data/preds_baseline.npz` | `assets/preds_baseline.npz` (R2) | Baseline prediction array for dual-track Track 1 comparison. If absent, `dual_track.py` logs a warning and runs Track 2 only (session-relative novelty). Fetch from R2 before first production scan: `aws s3 cp s3://<bucket>/assets/preds_baseline.npz scout_data/preds_baseline.npz --endpoint-url $R2_ENDPOINT_URL` |

## Environment Variables (production)

```
R2_ENDPOINT_URL        — Cloudflare R2 endpoint (https://…r2.cloudflarestorage.com)
R2_ACCESS_KEY_ID       — R2 access key
R2_SECRET_ACCESS_KEY   — R2 secret key
R2_BUCKET              — Bucket name for scan artifacts
R2_PUBLIC_URL          — Optional public base URL for viewer links
FAKE_TRIBE=1           — Dev mode: skip Modal calls, use synthetic preds
```
