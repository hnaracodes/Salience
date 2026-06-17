# Environment variables reference

All variables used across Salience production services. Set these in each host's secret manager (Vercel, Railway, Render, Modal) — never commit real values.

---

## Frontend — Vercel (`apps/web`)

| Variable | Required | Example | Notes |
|----------|----------|---------|-------|
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Yes (prod) | `pk_live_...` | From Clerk Dashboard → API Keys |
| `CLERK_SECRET_KEY` | Yes (prod) | `sk_live_...` | Server-side only; Vercel encrypts |
| `NEXT_PUBLIC_API_URL` | Yes | `https://api.yourdomain.com` | Public FastAPI base URL, no trailing slash |
| `NEXT_PUBLIC_CLERK_SIGN_IN_URL` | No | `/sign-in` | Default works for most setups |
| `NEXT_PUBLIC_CLERK_SIGN_UP_URL` | No | `/sign-up` | Default works for most setups |

Without `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, the app runs in **dev bypass mode**: landing page works, `/dashboard` and `/scans` redirect home with `?auth=required`.

---

## API — Railway / Render (`Dockerfile.api`)

| Variable | Required | Example | Notes |
|----------|----------|---------|-------|
| `DATABASE_URL` | Yes | `postgresql+asyncpg://user:pass@host:5432/scout` | SQLAlchemy URL; `+asyncpg` is stripped internally |
| `REDIS_URL` | Yes | `redis://default:pass@host:6379` | Arq job queue |
| `R2_ENDPOINT_URL` | Yes (prod) | `https://<accountid>.r2.cloudflarestorage.com` | S3-compatible endpoint |
| `R2_ACCESS_KEY_ID` | Yes (prod) | — | R2 API token access key |
| `R2_SECRET_ACCESS_KEY` | Yes (prod) | — | R2 API token secret |
| `R2_BUCKET` | Yes (prod) | `salience` | Bucket for `scans/<id>/ux_viewer/` |
| `R2_PUBLIC_URL` | Recommended | `https://artifacts.yourdomain.com` | Public base for viewer links; omit to use 7-day presigned URLs |
| `CLERK_JWKS_URL` | Yes (prod) | `https://<clerk-domain>/.well-known/jwks.json` | JWT verification |
| `CLERK_ISSUER` | Yes (prod) | `https://<clerk-domain>` | Must match token `iss` claim |
| `CORS_ORIGINS` | Yes | `["https://app.yourdomain.com"]` | JSON array string; include staging origins |

Health check path: `GET /health` → `{"status":"ok"}`

---

## Worker — Railway / Render (`Dockerfile.worker`)

| Variable | Required | Example | Notes |
|----------|----------|---------|-------|
| `DATABASE_URL` | Yes | Same as API | Worker updates scan status in Postgres |
| `REDIS_URL` | Yes | Same as API | Must point to same Redis instance as API |
| `R2_ENDPOINT_URL` | Yes (prod) | Same as API | Uploads viewer artifacts after pipeline |
| `R2_ACCESS_KEY_ID` | Yes (prod) | Same as API | |
| `R2_SECRET_ACCESS_KEY` | Yes (prod) | Same as API | |
| `R2_BUCKET` | Yes (prod) | Same as API | |
| `R2_PUBLIC_URL` | Recommended | Same as API | Stored in `scans.viewer_url` |
| `MODAL_TOKEN_ID` | Yes (prod) | — | From `modal token new` |
| `MODAL_TOKEN_SECRET` | Yes (prod) | — | Paired with token ID |
| `FAKE_TRIBE` | Dev/CI only | `1` | Skips Modal GPU; synthetic preds + uniform heatmaps |

**Do not set `FAKE_TRIBE=1` in production** unless you are running a demo environment without GPU.

Worker start command (already in Dockerfile):

```bash
python -m arq services.api.jobs.WorkerSettings
```

Recommended worker resources: **≥ 4 GB RAM**, **≥ 2 vCPU** (Playwright + video processing).

---

## Modal (`tribe.py`)

Configured via Modal dashboard and CLI, not env files in this repo.

| Secret / config | Required | Notes |
|-----------------|----------|-------|
| `huggingface-secret` | Yes | Modal secret containing `HF_TOKEN` for TRIBE weights |
| App name | Yes | `tribe-v2-brain-sim` (must match `modal.Cls.lookup` in `runner.py`) |
| Class | Yes | `TribeInference` with methods `predict_brain_npz`, `extract_frame_attention` |

Deploy from repo root:

```bash
modal deploy tribe.py
```

---

## Local development (`docker-compose.yml`)

Copy and extend for local testing:

```env
# .env.compose (gitignored)
CLERK_JWKS_URL=
CLERK_ISSUER=
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
```

Compose defaults use MinIO as R2 stand-in and `FAKE_TRIBE=1` on the worker.

---

## CORS example for multiple client domains

If you host the same product for multiple front-end domains (staging + prod + white-label):

```json
["https://app.yourdomain.com","https://staging.yourdomain.com","https://client-a.com"]
```

Pass as the `CORS_ORIGINS` env var on the API service.

---

## Clerk JWT values (how to find them)

1. Clerk Dashboard → **Configure** → **Domains**
2. **JWKS URL**: `https://<your-clerk-frontend-api>/.well-known/jwks.json`
3. **Issuer**: `https://<your-clerk-frontend-api>` (no path)

Use **Production** instance keys on Vercel production and API production.

---

## Verification commands

```bash
# API health
curl https://api.yourdomain.com/health

# R2 connectivity (from machine with AWS CLI)
aws s3 ls s3://your-bucket --endpoint-url $R2_ENDPOINT_URL

# Modal lookup (from worker env)
python -c "import modal; print(modal.Cls.lookup('tribe-v2-brain-sim', 'TribeInference'))"
```
