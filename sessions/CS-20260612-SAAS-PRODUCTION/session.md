---
session_id: CS-20260612-SAAS-PRODUCTION
title: Neural-UX Scout SaaS — Full Production Build
date: 2026-06-12
model: claude-sonnet-4-5
signature: claude-sonnet-4-5@CS-20260612-SAAS-PRODUCTION
status: completed
track: saas
related:
  - CS-20260612-PUSH-HARDENING
  - CS-20260610-PLAYWRIGHT-EXPLORATION
  - CS-20260611-TEXT-ENGAGEMENT-SCORING
files_touched:
  - services/pipeline/runner.py
  - services/pipeline/artifacts.py
  - services/api/main.py
  - services/api/models.py
  - services/api/jobs.py
  - services/api/ssrf.py
  - scout_core/copy_signals.py
  - scout_core/explore_policy.py
  - scout_core/marketing_scores.py
  - configs/explore_production.yaml
  - apps/web/
  - Dockerfile.api
  - Dockerfile.worker
  - docker-compose.yml
  - sessions/INDEX.md
---

# Neural-UX Scout SaaS — Full Production Build

## Summary

Transforms the TribeV2 CLI prototype into a production SaaS where any signed-in user
submits a URL, a containerized Playwright worker crawls the full site, the TRIBE pipeline
runs on Modal, copy signals fuse with neural scores, and results appear in a Next.js
dashboard with Clerk auth and Railway hosting.

Stack: Vercel (Next.js) + Railway (API + Playwright worker) + Modal (TRIBE GPU) + Clerk (auth) + Cloudflare R2 (artifacts).

## Key architectural decision

The worker container owns the entire session lifecycle (all pipeline stages run
locally within the worker). Modal is called via Python SDK `.remote()` for only two
stateless GPU operations: `TribeInference.predict_brain_npz` and
`TribeInference.extract_frame_attention`. The API is thin: enqueue, status, signed URLs.
activations.sqlite is ephemeral scratch per job; durable state is Postgres + R2.

## Prior sessions consulted

- CS-20260612-PUSH-HARDENING — security fixes (XSS, provenance, base-param) on `improved` branch; ISSUE-SEC-004 open
- CS-20260610-PLAYWRIGHT-EXPLORATION — explore policy scroll-then-nav improvements
- CS-20260611-TEXT-ENGAGEMENT-SCORING — text/copy engagement gap identified; ISSUE-001 open

## Important Files

- `services/pipeline/runner.py` — importable stage coordinator (wraps existing scripts)
- `services/pipeline/artifacts.py` — R2 upload/signed-URL client
- `services/api/main.py` — FastAPI thin API (scans CRUD, Arq enqueue, Clerk JWT)
- `services/api/ssrf.py` — SSRF gate (scheme, DNS, private-range, redirect re-check)
- `scout_core/copy_signals.py` — heuristic text scoring fused into marketing_scores
- `scout_core/explore_policy.py` — ISSUE-SEC-004 closed (role/text locators)
- `configs/explore_production.yaml` — production crawl profile for arbitrary URLs
- `apps/web/` — Next.js 14 app with Clerk, scroll animations, animated SVG logo

## Validation

- [ ] `services/pipeline/runner.py` importable; `FAKE_TRIBE=1` runs full pipeline offline
- [ ] pytest 159+ passing after changes
- [ ] `docker compose up` reaches `api` + `worker` ready state with `FAKE_TRIBE=1`
- [ ] Next.js `npm run build` succeeds
- [ ] `apps/web` Clerk sign-in flow navigates to dashboard

## How To Continue

1. Complete Phase 0 subagent: verify `services/pipeline/runner.py` exists and imports cleanly
2. Run `FAKE_TRIBE=1 python -c "from services.pipeline.runner import run_scan; print('ok')"` from project root
3. Run existing pytest suite; new tests in `tests/test_copy_signals.py` and `tests/test_ssrf.py`
4. `docker compose up --build` with `.env.compose` stub
5. Deploy `apps/web` with `vercel dev` pointing at local API

## Signature

Signed-off-by: claude-sonnet-4-5 on 2026-06-12
