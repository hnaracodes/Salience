---
id: SPRINT-001
title: SaaS Full Build Sprint
status: in_progress
phases: [0, 1, 2, 3, 4, 5]
---

# SPRINT-001 — SaaS Full Build

## Phases

| Phase | Goal | Agent | Status |
|-------|------|-------|--------|
| 0 | Pipeline portability (runner.py, R2, FAKE_TRIBE) | pipeline subagent | in_progress |
| 1 | Explore policy hardening (ISSUE-SEC-004 close) | pipeline subagent | in_progress |
| 2 | Copy signals heuristic + marketing fusion | pipeline subagent | in_progress |
| 3 | FastAPI + Arq + Postgres + SSRF gate | backend subagent | in_progress |
| 4 | Next.js + Clerk + scroll animations + logo | frontend subagent | in_progress |
| 5 | Dockerfiles + docker-compose + CI | backend subagent | in_progress |

## Acceptance criteria

- [ ] FAKE_TRIBE=1 runs full pipeline without GPU
- [ ] pytest suite 159+ passing
- [ ] copy_signals.py produces clarity/urgency/goal_fit scores from dom_snapshots
- [ ] marketing_scores.py fuses copy signal at 0.20 weight
- [ ] ISSUE-SEC-004 closed (role/text locators in production config)
- [ ] POST /v1/scans rejects private-range URLs
- [ ] Next.js builds successfully
- [ ] docker compose up brings up api+worker+redis+postgres+minio
