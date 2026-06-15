# Production deployment guide — Salience

This guide walks through deploying **Salience** so it is available to clients: sign up, submit a URL, receive a neural UX report in the interactive viewer.

> **View this guide:** open `TribeV2/tutorials/production-deployment.md` in Cursor/VS Code (Markdown preview: `Ctrl+Shift+V`). It is intentionally **not** on the public website.

**Stack**

| Layer | Service | Role |
|-------|---------|------|
| Frontend | **Vercel** | Next.js 14 app, marketing + dashboard |
| Auth | **Clerk** | Sign-up, sign-in, JWT for API |
| API | **Railway** or **Render** | FastAPI — enqueue scans, list status |
| Worker | **Railway** or **Render** | Arq worker — Playwright crawl + full pipeline |
| Queue | **Redis** | Arq job broker (Upstash, Railway, or Render) |
| Database | **PostgreSQL** | Scan metadata, users (Neon, Supabase, Railway, or Render) |
| GPU | **Modal** | TRIBE v2 inference + DINOv2 heatmaps |
| Artifacts | **Cloudflare R2** | Viewer HTML, assets, session outputs |

> **Clerk**: This guide includes Clerk steps for completeness. You can configure Clerk in a separate session; the app will run in landing-only mode until keys are set.

---

## Table of contents

1. [Architecture recap](#1-architecture-recap)
2. [Pre-flight: verify locally](#2-pre-flight-verify-locally)
3. [Provision shared infrastructure](#3-provision-shared-infrastructure)
4. [Deploy Modal (GPU)](#4-deploy-modal-gpu)
5. [Deploy API](#5-deploy-api)
6. [Deploy worker](#6-deploy-worker)
7. [Deploy frontend (Vercel)](#7-deploy-frontend-vercel)
8. [Configure Clerk](#8-configure-clerk)
9. [Wire services together](#9-wire-services-together)
10. [Database migrations](#10-database-migrations)
11. [Pipeline assets](#11-pipeline-assets)
12. [Production smoke test](#12-production-smoke-test)
13. [Making the product available to clients](#13-making-the-product-available-to-clients)
14. [Staging vs production](#14-staging-vs-production)
15. [Operations & monitoring](#15-operations--monitoring)
16. [Troubleshooting](#16-troubleshooting)
17. [Cost & scaling notes](#17-cost--scaling-notes)

---

## 1. Architecture recap

**Request flow**

1. User signs in on Vercel (Clerk).
2. User submits URL → Next.js calls `POST /v1/scans` with Clerk Bearer token.
3. API validates URL (SSRF gate), creates Postgres row, enqueues Arq job on Redis.
4. Worker picks up job, runs 8 pipeline stages locally in the container.
5. Worker calls **Modal** only for GPU: `predict_brain_npz` (TRIBE) and heatmap extraction.
6. Worker uploads `ux_viewer/` to **R2**, saves `viewer_url` on the scan row.
7. User polls `GET /v1/scans/{id}` and opens viewer in dashboard iframe.

**What stays ephemeral per job**

- Local session files under the worker filesystem (`activations.sqlite`, video, intermediate NPZ).
- Durable state: **Postgres** (metadata) + **R2** (viewer bundle).

See also: [environment-variables.md](./environment-variables.md)

---

## 2. Pre-flight: verify locally

Run from the **TribeV2 repo root** (directory containing `docker-compose.yml`).

### 2.1 Install dependencies

```bash
# Python (API tests)
pip install -r requirements.txt pytest

# Frontend
cd apps/web && npm install && cd ../..
```

### 2.2 Unit tests

```bash
set FAKE_TRIBE=1
set PYTHONPATH=.
python -m pytest tests/ -v --tb=short -k "not playwright"
```

### 2.3 Local full stack (no Modal, no Clerk)

```bash
docker compose up --build
```

In another terminal:

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

Worker runs with `FAKE_TRIBE=1` (synthetic neural preds). MinIO substitutes R2 at `http://localhost:9000`.

### 2.4 Frontend build

```bash
cd apps/web
npm run build
```

Fix any TypeScript or Next.js errors before deploying.

---

## 3. Provision shared infrastructure

Create these **before** deploying API/worker. Use managed services for production reliability.

### 3.1 PostgreSQL

**Options**: [Neon](https://neon.tech), [Supabase](https://supabase.com), Railway Postgres, Render Postgres.

1. Create a database named `scout` (or any name).
2. Copy the connection string.
3. Ensure the URL uses the async driver form for this project:

```
postgresql+asyncpg://USER:PASSWORD@HOST:5432/scout?sslmode=require
```

The API strips `+asyncpg` internally for synchronous SQLAlchemy.

### 3.2 Redis

**Options**: [Upstash Redis](https://upstash.com), Railway Redis, Render Key Value.

1. Create a Redis instance in the **same region** as your worker when possible.
2. Copy the connection URL:

```
redis://default:PASSWORD@HOST:6379
```

API and worker **must share the same Redis URL**.

### 3.3 Cloudflare R2

1. Cloudflare Dashboard → **R2** → Create bucket (e.g. `neural-ux-scout-prod`).
2. **Manage R2 API tokens** → Create token with Object Read & Write on that bucket.
3. Note:
   - `R2_ENDPOINT_URL` — `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`
   - `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET` — bucket name

4. **Public viewer access** (choose one):

   **Option A — Custom domain (recommended for clients)**

   - R2 bucket → Settings → Connect custom domain → `artifacts.yourdomain.com`
   - Set `R2_PUBLIC_URL=https://artifacts.yourdomain.com`
   - Viewer URLs become stable: `https://artifacts.yourdomain.com/scans/<scan_id>/ux_viewer/index.html`

   **Option B — Presigned URLs (default)**

   - Omit `R2_PUBLIC_URL`
   - API/worker generate 7-day presigned URLs (stored in `scans.viewer_url`)

5. Upload optional baseline asset (recommended for dual-track scoring):

```bash
aws s3 cp scout_data/preds_baseline.npz s3://YOUR_BUCKET/assets/preds_baseline.npz \
  --endpoint-url $R2_ENDPOINT_URL
```

See `services/pipeline/ASSETS_REQUIRED.md` for full asset list.

---

## 4. Deploy Modal (GPU)

Modal runs TRIBE v2 on A100 GPUs. The worker calls it via the Python SDK — no HTTP webhook from Modal to your infra.

### 4.1 Prerequisites

- [Modal account](https://modal.com)
- Hugging Face token with access to `facebook/tribev2`

### 4.2 Install CLI and authenticate

```bash
pip install modal
modal setup
```

### 4.3 Create Hugging Face secret in Modal

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxx
```

### 4.4 Deploy the app

From **TribeV2 repo root**:

```bash
modal deploy tribe.py
```

This publishes app `tribe-v2-brain-sim` with class `TribeInference`. The worker looks up exactly these names in `services/pipeline/runner.py`.

### 4.5 Create worker token

On the machine or CI that runs the worker (or copy into Railway/Render secrets):

```bash
modal token new --name scout-worker-prod
```

Save `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

### 4.6 Verify deployment

```bash
modal app list
# Should show tribe-v2-brain-sim

python -c "import modal; c = modal.Cls.lookup('tribe-v2-brain-sim', 'TribeInference'); print(c)"
```

First real scan will download TRIBE weights to the Modal volume (`tribe-weights-vol`) — expect several minutes cold start.

---

## 5. Deploy API

The API is a Docker web service exposing port **8000**.

### 5.1 Railway (recommended in original build)

1. [railway.app](https://railway.app) → New Project → **Deploy from GitHub** → select repo.
2. Add service → **Dockerfile** path: `Dockerfile.api` (repo root = TribeV2).
3. Add **PostgreSQL** and **Redis** plugins, or paste external URLs.
4. Set environment variables (see [environment-variables.md](./environment-variables.md)):

   | Variable | Value |
   |----------|-------|
   | `DATABASE_URL` | Postgres URL with `+asyncpg` |
   | `REDIS_URL` | Redis URL |
   | `R2_*` | From §3.3 |
   | `CLERK_JWKS_URL` | From Clerk (§8) |
   | `CLERK_ISSUER` | From Clerk (§8) |
   | `CORS_ORIGINS` | `["https://YOUR_VERCEL_DOMAIN"]` |

5. **Networking** → Generate domain → e.g. `scout-api-production.up.railway.app`
6. Health check: path `/health`, port 8000.

### 5.2 Render (alternative)

1. [render.com](https://render.com) → New → **Web Service** → connect repo.
2. **Environment**: Docker
3. **Dockerfile path**: `./Dockerfile.api`
4. **Instance type**: Starter (512 MB) is enough for API-only; scale if needed.
5. Add env vars (same as Railway table above).
6. Health check path: `/health`

### 5.3 API does not run migrations automatically in production

Tables are created via `Base.metadata.create_all` on startup for dev. **Production should use Alembic** (§10).

---

## 6. Deploy worker

The worker is a **long-running background process** (not an HTTP server). It must use the Playwright image (`Dockerfile.worker`).

### 6.1 Railway

1. Same project → **New Service** → Dockerfile: `Dockerfile.worker`
2. **No public HTTP port** — disable domain or do not expose port.
3. Start command (default from Dockerfile):

   ```
   python -m arq services.api.jobs.WorkerSettings
   ```

4. Environment: same `DATABASE_URL`, `REDIS_URL`, `R2_*`, `MODAL_TOKEN_*` as API.
5. **Do not set `FAKE_TRIBE=1`** in production.
6. Resources: minimum **4 GB RAM / 2 vCPU**; increase for large multi-page crawls.

### 6.2 Render

1. New → **Background Worker** (not Web Service).
2. Dockerfile: `./Dockerfile.worker`
3. Docker command: `python -m arq services.api.jobs.WorkerSettings`
4. Plan: at least **Standard** (2 GB RAM); Playwright + video often needs more.
5. Same env vars as §6.1.

### 6.3 Why two services?

- API stays lightweight and scales on HTTP traffic.
- Worker runs CPU/RAM-heavy Playwright + pipeline; scale workers independently.
- Both must share **Redis** (queue) and **Postgres** (status).

### 6.4 Scaling workers

Arq processes one job per worker instance by default. To increase throughput:

- Run multiple worker replicas (same `REDIS_URL`).
- Consider per-plan concurrency limits (API already limits 1 active scan per user on free tier).

---

## 7. Deploy frontend (Vercel)

### 7.1 Import project

1. [vercel.com](https://vercel.com) → Add New → Project → import Git repo.
2. **Root directory**: `apps/web`
3. Framework preset: **Next.js** (auto-detected).

### 7.2 Build settings

| Setting | Value |
|---------|-------|
| Build command | `npm run build` (default) |
| Output | Next.js default |
| Install | `npm install` |

### 7.3 Environment variables (Production)

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk production publishable key |
| `CLERK_SECRET_KEY` | Clerk production secret key |
| `NEXT_PUBLIC_API_URL` | `https://your-api-domain` (no trailing slash) |

Redeploy after changing any `NEXT_PUBLIC_*` variable.

### 7.4 Custom domain

Vercel → Project → Domains → add `app.yourdomain.com` (or apex).

Update API `CORS_ORIGINS` to include the final Vercel URL.

### 7.5 Preview deployments

For each preview URL, either:

- Add preview origins to `CORS_ORIGINS` on a staging API, or
- Use a staging API service with broader CORS for `*.vercel.app` (not recommended for production API).

---

## 8. Configure Clerk

Complete this when you are ready to open sign-up to clients. Skip during initial infra testing — the landing page works without keys.

### 8.1 Create application

1. [dashboard.clerk.com](https://dashboard.clerk.com) → Create application.
2. Enable **Email** and any social providers you want clients to use.

### 8.2 Get API keys

Dashboard → **API Keys**:

- Publishable key → Vercel `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
- Secret key → Vercel `CLERK_SECRET_KEY`

Use **Production** keys for production Vercel environment.

### 8.3 JWT settings for FastAPI

Dashboard → **Configure** → **Domains**:

- `CLERK_ISSUER` = `https://<your-clerk-frontend-api>` (no trailing path)
- `CLERK_JWKS_URL` = `https://<your-clerk-frontend-api>/.well-known/jwks.json`

Set both on the **API** service (Railway/Render).

### 8.4 Allowed origins

Clerk → **Configure** → **Paths** / **Domains**:

- Add your Vercel production domain.
- Add `http://localhost:3000` for local dev.

### 8.5 Next.js integration checklist

The repo already includes:

- `ClerkProvider` in `apps/web/app/layout.tsx` (wraps `<body>` content via children in `<html>`)
- `middleware.ts` protecting `/dashboard` and `/scans`
- `lib/auth.ts` — `clerkEnabled` flag for graceful degradation

After adding keys:

1. Redeploy Vercel.
2. Visit `/sign-up` or use Clerk modal from landing CTA.
3. Confirm redirect to `/dashboard` after sign-in.

### 8.6 Optional: Clerk CLI (separate session)

```bash
npm install -g @clerk/cli
clerk auth login
cd apps/web && clerk init
clerk doctor
```

Pull env files locally only — copy values into Vercel/Railway secrets, never commit `.env.local`.

---

## 9. Wire services together

Use this checklist after all services are deployed.

| Step | Action |
|------|--------|
| 1 | API `CORS_ORIGINS` includes Vercel production URL |
| 2 | Vercel `NEXT_PUBLIC_API_URL` points to API HTTPS URL |
| 3 | API + worker share `DATABASE_URL` and `REDIS_URL` |
| 4 | Worker has `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` |
| 5 | Worker has all `R2_*` vars; `FAKE_TRIBE` unset |
| 6 | Clerk JWT vars on API match production Clerk instance |
| 7 | R2 custom domain (if used) serves `index.html` over HTTPS |

**Auth test (with Clerk configured)**

```bash
# Get a session token from Clerk (Dashboard → Users → impersonate, or sign in via browser DevTools)
curl -H "Authorization: Bearer <clerk_jwt>" https://your-api/v1/scans
# → {"scans":[],"total":0}
```

---

## 10. Database migrations

For production, run Alembic before accepting traffic.

From TribeV2 root with `DATABASE_URL` set to production (sync URL without `+asyncpg` is fine for Alembic):

```bash
pip install alembic psycopg2-binary
cd services/api
alembic upgrade head
```

Migration file: `services/api/alembic/versions/001_initial.py` creates `users`, `scans`, `scan_artifacts`.

**Railway/Render one-off job**: run the same command in a release phase or manual shell with production `DATABASE_URL`.

---

## 11. Pipeline assets

Bundled in the Docker images from the repo:

| Asset | Path | Required |
|-------|------|----------|
| Explore config | `configs/explore_production.yaml` | Yes |
| Vertex regions | `configs/vertex_regions.csv` | Yes |
| Emotion templates | `configs/emotion_templates/` | Optional |
| Norms | `scout_norms/synthetic_bootstrap_v1/` | Recommended |
| Baseline preds | `scout_data/preds_baseline.npz` or R2 | Recommended |

Worker container includes repo `COPY . .` — no extra upload needed unless you maintain private assets outside git.

---

## 12. Production smoke test

### 12.1 Health

```bash
curl https://your-api/health
```

### 12.2 End-to-end scan (signed-in user)

1. Sign in on production Vercel app.
2. Go to **New scan** (`/scans/new`).
3. Submit a **public, simple URL** you own (e.g. `https://example.com` or your marketing site).
4. Watch status on scan detail page — stages: `capture` → `tribe` → … → `export_viewer`.
5. When `status: done`, open `viewer_url` in the embedded iframe.

**Expected duration**: 5–20 minutes depending on site size and Modal cold start.

### 12.3 Failure modes

| Symptom | Likely cause |
|---------|----------------|
| `401` on API | Clerk JWT mismatch; check issuer/JWKS |
| `422 URL rejected` | SSRF gate blocked private/invalid URL |
| `429` | Free tier: 1 concurrent scan per user |
| Scan `failed` at `tribe` | Modal auth, HF token, or app name mismatch |
| Scan `failed` at `capture` | Playwright timeout, site blocked bot, or RAM |
| No `viewer_url` | R2 credentials or missing `ux_viewer/` export |

Check worker logs in Railway/Render for stage stderr (runner logs last 2000 chars on failure).

---

## 13. Making the product available to clients

### 13.1 Single SaaS (default)

One Vercel app + one API + shared worker pool. Any user who signs up via Clerk gets an account. This is the intended model in the current codebase (`users.clerk_user_id`, per-user scan list).

**Go-live checklist for clients**

- [ ] Production Clerk with your branding (logo, colors — already dark-themed in `layout.tsx`)
- [ ] Custom domain on Vercel (`app.yourcompany.com`)
- [ ] Privacy policy + terms (Clerk can link from sign-up)
- [ ] Rate limits / plan tiers (API has 1 concurrent scan on free tier — extend in `main.py` for paid plans)
- [ ] Support email in CTAs

### 13.2 White-label / agency clients

To offer the same product under a client domain:

1. Deploy a **Vercel project** (or alias) per client domain, or use Clerk organizations later.
2. Add each client origin to API `CORS_ORIGINS`.
3. Optionally separate R2 bucket prefix per client (`scans/<scan_id>/` is already isolated per scan).
4. Use Clerk **Organizations** for B2B seat management (future enhancement).

### 13.3 Viewer sharing

`viewer_url` from R2 can be shared with clients who are not logged in if:

- You use a **public R2 custom domain**, and
- You accept that anyone with the link can view that report.

For private reports, keep presigned URLs (short TTL) or add a viewer auth proxy (not in current build).

### 13.4 Data retention

Scans have `expires_at` (30 days from creation in `main.py`). Add a cron job or Railway scheduled task to purge expired scans and R2 prefixes via `DELETE /v1/scans/{id}` logic.

---

## 14. Staging vs production

Recommended layout:

| Environment | Vercel | API/Worker | Clerk | Modal | R2 bucket |
|-------------|--------|------------|-------|-------|-----------|
| Staging | Preview or `staging.*` | Separate services | Development instance | Same or `modal deploy` staging app | `scout-staging` |
| Production | `app.*` | Production services | Production instance | `tribe-v2-brain-sim` | `scout-prod` |

Staging worker may use `FAKE_TRIBE=1` to avoid GPU cost; production must not.

---

## 15. Operations & monitoring

### 15.1 Logs

- **API**: HTTP access, 4xx/5xx, SSRF rejections
- **Worker**: Pipeline stage logs (`services/pipeline/runner.py`), Modal call duration
- **Vercel**: Build and runtime logs for Next.js

### 15.2 Alerts (suggested)

- API health check failing > 2 min
- Worker queue depth (Redis `LLEN arq:queue`) growing unbounded
- Modal spend threshold
- R2 storage growth

### 15.3 CI

GitHub Actions (`.github/workflows/ci.yml`) runs pytest and Docker builds on `main` / `improved`. Extend with:

- Deploy hooks to Railway/Render on tag
- Vercel Git integration for frontend (automatic)

### 15.4 Security

- SSRF validation at API intake **and** worker capture (`services/api/ssrf.py`)
- Clerk JWT on all `/v1/scans` routes
- R2 viewer upload sanitizes inline `<script>` in HTML (`artifacts.py`)
- Never expose `MODAL_TOKEN_*` or `CLERK_SECRET_KEY` to the browser

---

## 16. Troubleshooting

### Clerk: app loads but `/scans` redirects home

`NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` missing on Vercel. Add keys and redeploy.

### CORS error in browser

API `CORS_ORIGINS` must be a JSON array string including exact Vercel origin (scheme + host, no path).

### Worker not picking up jobs

- Confirm API and worker use identical `REDIS_URL`
- Confirm worker process is running (`arq` logs "Starting worker")
- Check Arq registered function: `run_scan_task`

### Modal: `LookupError` for tribe-v2-brain-sim

Run `modal deploy tribe.py` from repo root. App name must match `runner.py`.

### Playwright OOM on Render

Upgrade worker plan or reduce `configs/explore_production.yaml` page limits.

### Viewer blank in iframe

- Confirm `viewer_url` is HTTPS and allows embedding (R2 custom domain CSP)
- Check `apps/web` scan detail page sandbox attributes

---

## 17. Cost & scaling notes

| Service | Cost driver |
|---------|-------------|
| Vercel | Bandwidth, serverless invocations (low for dashboard) |
| Railway/Render API | Always-on instance hours |
| Railway/Render Worker | RAM/CPU hours; dominant cost for crawl volume |
| Modal | A100 GPU seconds per scan (TRIBE + heatmaps) |
| R2 | Storage + egress (custom domain egress may incur fees) |
| Clerk | MAU pricing |
| Postgres/Redis | Instance size |

**Cost control**

- Staging with `FAKE_TRIBE=1`
- Per-user concurrency limit (already 1 on free tier)
- Scan `expires_at` + automated purge
- Modal: warm volumes reduce weight download time, not GPU inference cost

---

## Quick reference — deploy order

```
1. Postgres + Redis + R2
2. modal deploy tribe.py + Modal token
3. API (Railway/Render) + alembic upgrade
4. Worker (Railway/Render)
5. Vercel (apps/web)
6. Clerk keys → Vercel + API JWT vars
7. CORS + smoke test scan
```

---

## Related docs

- [Environment variables](./environment-variables.md)
- [Pipeline assets](../services/pipeline/ASSETS_REQUIRED.md)
- [Website session runbook](../docs/runbooks/website-session.md)
- [Frontend env example](../apps/web/.env.local.example)

---

*Last updated: 2026-06-12 — matches `improved` branch SaaS architecture (Vercel + Railway/Render + Modal + Clerk + R2).*
