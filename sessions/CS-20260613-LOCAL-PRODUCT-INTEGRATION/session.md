---
session_id: CS-20260613-LOCAL-PRODUCT-INTEGRATION
title: Local product integration — Clerk, API wiring, security, E2E
date: 2026-06-13
model: composer-2.5
signature: composer-2.5@CS-20260613-LOCAL-PRODUCT-INTEGRATION
status: completed
track: saas
related:
  - CS-20260612-SAAS-PRODUCTION
  - CS-20260616-DOCKER-RUNTIME-FIXES
  - CS-20260616-CI-REQUIREMENTS
agent_transcripts:
  - 4dc6ae9f-6f8f-4e78-acf5-7b0ea97b8b9d
files_touched:
  - apps/web/middleware.ts
  - apps/web/app/layout.tsx
  - apps/web/components/ScanList.tsx
  - services/api/schemas.py
  - services/api/main.py
  - services/api/auth.py
  - services/api/ssrf.py
  - scripts/record_website_session.py
  - services/pipeline/runner.py
  - scout_core/llm_narrative.py
  - scout_core/storage_migrations.py
  - scripts/analyze_session.py
  - docker-compose.yml
  - tests/test_capture_script.py
  - tests/test_api_security.py
  - tests/test_pipeline_e2e.py
  - tests/conftest.py
---

# Local product integration — Clerk, API wiring, security, E2E

## Summary

Integrated the SaaS product for local Docker + Next.js dev: Clerk middleware/provider fixes, CORS for dev ports, API response shape fixes, capture `--url` support for production explore mode, security hardening on URL workers, and a full backend E2E integration test (`FAKE_TRIBE`) verifying all eight pipeline stages.

## Important Files

- `apps/web/` — Clerk middleware matcher, ScanList API unwrap, scan error display
- `scripts/record_website_session.py` — `--url` + SSRF re-validation for SaaS scans
- `services/pipeline/runner.py` — `_sync_sqlite_session()`, `--norm-id` on analyze
- `tests/test_pipeline_e2e.py` — `@pytest.mark.integration` full pipeline smoke
- `tests/test_capture_script.py`, `tests/test_api_security.py` — security regression

## Validation

- Security tests: 13/13 in API container
- E2E: 1/1 in worker container (~22s, all stages through export_viewer)
- `docker compose up -d --build api worker` — final bake-in rebuild

## Signature

Signed-off-by: Composer 2.5 (composer-2.5) on 2026-06-16
