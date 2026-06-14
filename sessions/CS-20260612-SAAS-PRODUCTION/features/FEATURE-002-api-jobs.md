---
id: FEATURE-002
status: completed
title: API + job queue
files:
  - services/api/
  - services/api/jobs.py
  - docker-compose.yml
---

# FEATURE-002 — API + Jobs

FastAPI thin API with Clerk JWT, Arq Redis queue, Postgres models, SSRF gate, rate limits, and idempotency.
