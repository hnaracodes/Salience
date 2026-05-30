# Coding sessions

Human-readable session notes for the TribeV2 / Neural-UX Scout repo. Each file summarizes **why** changes were made, **how** the pieces connect, and **how to run** them — intended for teammates who did not author the PR.

| Date | Document | Topics |
|------|----------|--------|
| 2026-05-21 | [2026-05-21-feature-isolation-playwright-website-pipeline.md](./2026-05-21-feature-isolation-playwright-website-pipeline.md) | Session-relative emotion Z-scoring, feature isolation (ViT + DOM), Playwright capture, section analytics, website session orchestration, UX viewer, LLM narrative |
| (archived) | [neuroEmoCode/](./neuroEmoCode/) | NeuroEmo supervised training, Schaefer atlas, experiment matrices — **not** in the website E2E path |
| 2026-05-25 | [2026-05-25-playwright-pipeline-hardening-and-subagent.md](./2026-05-25-playwright-pipeline-hardening-and-subagent.md) | Timeline-correct Playwright capture, richer alignment reporting, non-anger grounding, heatmap provenance/overwrite rules, downstream export cleanup, regression tests, and the new `playwright-pipeline` Cursor subagent quickstart |
| 2026-05-28 | [2026-05-28-website-pipeline-consolidated-session.md](./2026-05-28-website-pipeline-consolidated-session.md) | **Consolidated May 25–28 handoff:** ViralAnalyser learnings, marketing score layer, NeuroEmo → `neuroEmoCode/` archive, triple-track (engagement/emotion/activation), Schaefer atlas fix, gray baseline, Aurora E2E testing ladder, priority shift to website mainline |

Add new entries as `YYYY-MM-DD-short-topic.md` when you land substantial work.
