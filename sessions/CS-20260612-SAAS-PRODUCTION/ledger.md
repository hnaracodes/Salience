# Ledger — CS-20260612-SAAS-PRODUCTION

## Step 1 — Context

- Agent transcripts (implementation chats):
  - `d4f00b92-9c4d-429f-bed2-47b5a528688f` — Phases 0–2 (runner, copy_signals, explore hardening)
  - `c9c94c66-b168-4010-896e-941e5eb677ea` — Phases 3 & 5 (FastAPI, Docker, CI)
- Files read: sessions/INDEX.md, CS-20260612-PUSH-HARDENING/issues/ISSUE-SEC-004, tribe.py, explore_policy.py, run_website_session.py, activation_store.py, pipeline-runner.md, frontend-design-specialist.md, code-auditor.md
- Prior sessions consulted: CS-20260612-PUSH-HARDENING (open ISSUE-SEC-004), CS-20260611-TEXT-ENGAGEMENT-SCORING (open ISSUE-001)
- Plan audited: salience_saas_581e709b.plan.md — 12 gaps found and closed before build
- Open issues going in: ISSUE-SEC-004 (explore selectors), ISSUE-001 (visual-only scoring)

## Step 2 — Phase 0: Pipeline Portability

- Files touched:
  - `services/__init__.py` (new)
  - `services/pipeline/__init__.py` (new)
  - `services/pipeline/runner.py` (new) — importable stage coordinator, FAKE_TRIBE mode
  - `services/pipeline/artifacts.py` (new) — R2 upload, signed URL generation
- What changed: pipeline stages go from CLI-only to importable functions; Modal called via Python SDK bytes-in/bytes-out; R2 used for final artifact storage; FAKE_TRIBE env skips GPU for dev/CI
- Expected effect: `docker compose up` + `FAKE_TRIBE=1` runs the full pipeline without GPU/Modal account

## Step 3 — Phase 1: Explore Policy Hardening

- Files touched:
  - `scout_core/explore_policy.py` — ISSUE-SEC-004 closed: ExploreAction gains locator fields; prefer page.get_by_role/get_by_text over raw CSS selectors; selector allowlist validation
  - `configs/explore_production.yaml` (new) — production crawl profile for arbitrary URLs
- What changed: selectors for hostile DOM are now role+name locators; CSS/XPath blocked when allow_role_locators=true (production default)
- Expected effect: ISSUE-SEC-004 resolved; mis-click on untrusted production sites prevented

## Step 4 — Phase 2: Copy Signals

- Files touched:
  - `scout_core/copy_signals.py` (new) — heuristic clarity/urgency/goal_fit scores from dom_snapshots
  - `scout_core/marketing_scores.py` — fuse copy_signals mean into compound marketing score
  - `configs/marketing_scores.yaml` — new copy weight key
  - `tests/test_copy_signals.py` (new)
- What changed: text is now a scored signal; compound score = 0.55 neural + 0.25 novelty + 0.20 copy
- Expected effect: ISSUE-001 closed; element detail panel in viewer shows copy score chip

## Step 5 — Phase 3: API Backend

- Files touched:
  - `services/api/__init__.py` (new)
  - `services/api/main.py` (new) — FastAPI app, Arq job enqueue, Clerk JWT validation
  - `services/api/models.py` (new) — SQLAlchemy models (users, scans, scan_artifacts)
  - `services/api/schemas.py` (new) — Pydantic request/response models
  - `services/api/jobs.py` (new) — Arq task: run_scan wrapping runner.run_scan
  - `services/api/ssrf.py` (new) — SSRF gate: scheme, DNS, private-range, redirect re-check
  - `services/api/alembic/` (new) — migrations env + initial schema
  - `tests/test_ssrf.py` (new)
- What changed: REST API with Clerk-authenticated endpoints; SSRF-gated URL intake; Arq async job queue
- Expected effect: POST /v1/scans enqueues a job; worker picks it up; status polling via GET /v1/scans/{id}

## Step 6 — Phase 4: Frontend

- Files touched:
  - `apps/web/` (new directory — full Next.js 14 app)
  - `apps/web/package.json`, `tsconfig.json`, `next.config.ts`
  - `apps/web/app/` — App Router pages: landing, dashboard, scan, results
  - `apps/web/components/` — Logo, ScrollScene, ScanCard, ProgressStepper, ViewerFrame
  - `apps/web/public/logo.svg` — animated SVG logo (morphing neural/cursor motif)
- What changed: full user-facing product UI; dark theme; scroll-driven animations; embedded ux_viewer in results
- Expected effect: sign-in → dashboard → new scan → results with embedded viewer

## Step 7 — Phase 5: Containerization

- Files touched:
  - `Dockerfile.api` (new)
  - `Dockerfile.worker` (new)
  - `docker-compose.yml` (new)
  - `.github/workflows/ci.yml` (new)
  - `.env.example` (updated)
- What changed: both services containerized; docker-compose brings up api+worker+redis+postgres+minio for local dev
- Expected effect: `docker compose up --build` produces a working local stack

## Step 8 — Code Audit Fixes (code-auditor findings)

Auditor verdict was "Needs revision" — 3 Critical + 2 High bugs fixed before merge.

- Files touched:
  - `services/api/jobs.py` — C-1: added missing `config={}` positional arg to `runner.run_scan()`; C-2: changed `scan.viewer_url = viewer_url` to `scan.viewer_url = result["viewer_url"]` (run_scan returns dict); H-1: `WorkerSettings.redis_settings` now set to `RedisSettings.from_dsn(REDIS_URL)` instead of `None`
  - `Dockerfile.api` — H-2: added `psycopg2-binary` to pip install line
  - `services/api/main.py` — M-1: rate-limit check corrected from `> 1` to `>= 1` (was allowing 2 concurrent scans)
  - `configs/explore_production.yaml` — M-3: `unvisited_page_bonus: 40` corrected to `0.40` (100× typo)
  - `apps/web/app/scans/[id]/page.tsx` — C-3: page now owns scan polling; `viewer_url` flows down from polled scan response instead of pointing at non-existent `/v1/scans/{id}/viewer` endpoint
  - `apps/web/components/ProgressStepper.tsx` — accepts `externalScan?` prop to avoid duplicate network call when parent polls
  - `apps/web/components/ViewerFrame.tsx` — M-2: removed `allow-same-origin` from iframe sandbox; added comment explaining decision

## Validation

- Commands run: `pytest tests/test_ssrf.py tests/test_copy_signals.py tests/test_explore_policy.py -v --tb=short`
- Results: 29 passed, 0 failed
- Import check: `from services.api.jobs import WorkerSettings` → `WorkerSettings.redis_settings` is `RedisSettings(host='localhost', port=6379, ...)`
