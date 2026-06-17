---
name: two-week-descope
overview: "Refocus the repo on one repeatable Salience prototype spine: Playwright capture → TRIBEv2 inference → dual-track → heatmaps → grounding/feature isolation → viewer/report. Freeze parallel NeuroEmo R&D until the website pipeline is proven end-to-end on real sessions."
todos:
  - id: freeze-scope
    content: Declare the zero-shot website-session pipeline as the only mainline; freeze NeuroEmo phases 3–6, new classifiers, and atlas experiments unless they block pipeline parity.
    status: pending
  - id: pytest-baseline
    content: Run pipeline unit tests (record, align, spikes, heatmap provenance, feature isolation, section analytics) and fix any regressions before Modal E2E.
    status: pending
  - id: e2e-placeholder
    content: Complete one full session through capture → tribe → dual-track → uniform heatmaps → analyze --website --ground → export_viewer; verify artifact checklist.
    status: pending
  - id: e2e-modal-heatmaps
    content: Re-run heatmaps with --modal --force on the golden session; confirm real attention maps and section top_elements are not placeholder-only.
    status: pending
  - id: document-runbook
    content: Update docs/runbooks/website-session.md and run_website_session.py examples so --stage all includes --website --ground --refresh-sections and export_viewer.
    status: pending
  - id: benchmark-sessions
    content: Add 2 more walkthrough YAMLs; run 3 fixture sessions through the same documented flow; log failures by stage only.
    status: pending
  - id: end-of-sprint-decision
    content: Ship one canonical demo session folder + honest limitations doc; decide go/no-go on NeuroEmo runtime integration.
    status: pending
isProject: false
---

# Two-Week De-Scope Plan (updated 2026-05-28)

## Goal

Prove that the **website-session zero-shot pipeline** works reliably from **Playwright recording through TRIBEv2 inference, dual-track scoring, heatmap extraction, DOM-grounded feature isolation, and UX viewer export** — without starting new modeling branches.

NeuroEmo supervised work (surface-native Schaefer, vertex equivalence, postfix matrix) is **valuable but out of mainline** until this spine passes a documented end-to-end proof.

## Where we are (repository reality)

### Pipeline spine — code landed, E2E not closed

| Area | Status | Evidence |
|------|--------|----------|
| Timeline-correct Playwright capture | Landed | [coding-sessions/2026-05-25-playwright-pipeline-hardening-and-subagent.md](../../coding-sessions/2026-05-25-playwright-pipeline-hardening-and-subagent.md), [.cursor/plans/playwright-pipeline-fixes_0e4e0672.plan.md](playwright-pipeline-fixes_0e4e0672.plan.md) (todos completed) |
| Alignment reporting (`exact` / `tolerated` / `clamped` / `invalid`) | Landed | `scout_core/session_align.py` |
| Non-anger grounding + heatmap provenance | Landed | `scripts/analyze_session.py`, `scripts/extract_section_heatmaps.py` |
| Unit/regression tests | Landed | `tests/test_record_website_session.py`, `test_analyze_session_spikes.py`, `test_heatmap_provenance.py`, `test_feature_isolation.py`, `test_section_analytics.py` |
| Operator runbook | Partial | [docs/runbooks/website-session.md](../../docs/runbooks/website-session.md) exists; orchestrator flags need tightening |
| **Full E2E on a fresh session (Modal TRIBE + Modal heatmaps + viewer)** | **Not proven in sprint docs** | P0 checkpoints in [salience-phased-delivery-plan.md](../../docs/implementation-plans/salience-phased-delivery-plan.md) still open |
| Multiple benchmark walkthroughs | Gap | Only [configs/walkthrough_scripts/localhost_demo.yaml](../../configs/walkthrough_scripts/localhost_demo.yaml) |

### NeuroEmo — advanced in parallel (freeze, do not extend)

Recent work that should **not** expand during this sprint:

- Surface-native Schaefer atlas + vertex equivalence proof ([.cursor/plans/surface-native_schaefer_atlas_09315159.plan.md](surface-native_schaefer_atlas_09315159.plan.md), [.cursor/plans/vertex_equivalence_proof_eb9fafb6.plan.md](vertex_equivalence_proof_eb9fafb6.plan.md))
- Phase 2 surface postfix matrix ([coding-sessions/2026-05-28-neuroemo-phase2-surface-matrix-report.md](../../coding-sessions/2026-05-28-neuroemo-phase2-surface-matrix-report.md)): **geometry trust yes; 5-class metric win no** vs projected postfix
- Untracked/local: phases 3–6 scripts (`run_neuroemo_experiment_matrix.py`, timing grid NPZs, hierarchical/MLP trainers)

**Takeaway:** NeuroEmo is not ready for runtime integration into the website viewer. The website pipeline still uses **zero-shot dual-track** ([scout_core/dual_track.py](../../scout_core/dual_track.py)).

## Reasoning

[salience-phased-delivery-plan.md](../../docs/implementation-plans/salience-phased-delivery-plan.md) principle #1: **one runnable spine first**. May 25 hardening fixed known correctness bugs (capture-after-steps, anger-only spikes, silent placeholder heatmaps). That makes the spine *trustworthy in code* but does not replace a **repeatable operator proof** on real Modal runs.

The team has invested heavily in NeuroEmo training accuracy and atlas provenance while **P0-C1–C4** (golden session, Modal parity, artifact bundle, non-author runbook) remain unchecked. This sprint closes that gap.

## Two-week scope decision

### Keep as mainline (only shipping track)

- [scripts/record_website_session.py](../../scripts/record_website_session.py) → [scripts/run_website_session.py](../../scripts/run_website_session.py) (stages: `capture | tribe | dual_track | heatmaps | analyze | narrative | export_viewer | all`)
- Modal: `modal run tribe.py::record_session`
- [scripts/run_dual_track.py](../../scripts/run_dual_track.py) → [scout_core/dual_track.py](../../scout_core/dual_track.py)
- [scripts/extract_section_heatmaps.py](../../scripts/extract_section_heatmaps.py) → [scout_core/heatmap_extract.py](../../scout_core/heatmap_extract.py)
- [scripts/analyze_session.py](../../scripts/analyze_session.py) (`--website --ground`) → [scout_core/section_pipeline.py](../../scout_core/section_pipeline.py), [scout_core/dom_intersect.py](../../scout_core/dom_intersect.py)
- [scripts/export_ux_viewer.py](../../scripts/export_ux_viewer.py) → [viewer/ux_session_viewer.html](../../viewer/ux_session_viewer.html)
- Schemas: [scout_core/schemas.py](../../scout_core/schemas.py)

### Freeze for two weeks

- NeuroEmo phases 3–6 (timing grid, dynamic reducers, hierarchical decode) per [neuroemo_remaining_high_leverage_opportunities_2026-05-27.plan.md](neuroemo_remaining_high_leverage_opportunities_2026-05-27.plan.md)
- New classifier variants under `scripts/train_neuroemo_*.py`, `scripts/prepare_neuroemo_tribev2.py`
- Atlas/ROI research unless required to fix **website** parcellation breakage
- Demographic multiplex / cluster training ([demographic-multiplexer-implementation-plan.md](../../docs/implementation-plans/demographic-multiplexer-implementation-plan.md))
- 3D streaming viewer beyond static UX viewer
- Runtime supervised NeuroEmo inference in `analyze_session` / viewer

### Postpone until after proof point

- Wiring trained NeuroEmo models into website sessions
- `inference_mux`, cluster embeddings, demographic scaling
- Replacing projected postfix 5-class default with surface-native models (per May 28 matrix verdict)

## Success metric (two weeks)

A **non-author** can follow [docs/runbooks/website-session.md](../../docs/runbooks/website-session.md) and obtain, for **≥3 walkthroughs**, a session folder with:

- `walkthrough.webm` (or `.mp4`)
- `session_manifest.json` (schema v2, **varying** `dom_snapshots[].scrollY` across time)
- `preds.npz` with alignment `mapping` ∈ `{exact, tolerated}`
- `analysis_bundle.json` with `section_report`, `grounding_triggers`, and `events[]` when spikes qualify
- `heatmaps/manifest.json` with explicit `source` (`modal` vs `uniform_placeholder`)
- `ux_viewer/viewer_bundle.json` + `index.html` with alignment + provenance surfaced

## Next proof point

> **Not** “better NeuroEmo accuracy.” **Yes:** three repeatable website sessions where spike → heatmap → DOM winner → section summary → viewer is inspectable without reading Python.

### Acceptance criteria

- [ ] Pytest pipeline suite green (see testing ladder Layer 0)
- [ ] One **golden** session passes full Layer 3 checklist (Modal TRIBE + Modal heatmaps)
- [ ] At least one session with **non-placeholder** heatmaps driving `section_report[].top_elements`
- [ ] Viewer shows `alignment.mapping` not `invalid`; placeholder heatmaps labeled low-confidence
- [ ] Runbook matches actual `run_website_session.py` flags
- [ ] Limitations documented: synthetic norms, zero-shot emotion Z, Modal/ffmpeg prerequisites

---

## Clear-cut testing plan: Playwright → inference → heatmaps → feature analysis

Use [.cursor/agents/playwright-pipeline.md](../agents/playwright-pipeline.md) when debugging; run from repo root with `.venv` active.

### Layer 0 — Unit/regression (no Modal, no browser beyond fixtures)

```powershell
cd <repo-root>
.venv\Scripts\activate
python -m pytest tests/test_record_website_session.py tests/test_analyze_session_spikes.py `
  tests/test_heatmap_provenance.py tests/test_feature_isolation.py `
  tests/test_section_analytics.py tests/test_dual_track.py -v --tb=short
```

**Pass:** all targeted tests green. **Fail:** fix before any Modal spend.

Optional full suite (exclude heavy NeuroEmo model test):

```powershell
python -m pytest tests/ -v --tb=short --ignore=tests/test_neuroemo_model.py
```

### Layer 1 — Capture + alignment only (local, ~2 min)

**Prerequisites:** `playwright install chromium`; local fixture server.

```powershell
python -m http.server 8765 --directory tests/fixtures/walkthrough_site
# separate terminal:
python scripts/record_website_session.py --script configs/walkthrough_scripts/localhost_demo.yaml
```

**Verify (manual):**

| Check | Where |
|-------|--------|
| `dom_snapshots[0].scrollY` ≈ 0, later snapshots increase (400, 800…) | `session_manifest.json` |
| `video.path` present | manifest |
| `walkthrough.webm` exists | `scout_data/sessions/<id>/` |

### Layer 2 — CPU path with placeholder heatmaps (Modal TRIBE only)

Substitute `<ID>` from Layer 1.

```powershell
modal run tribe.py::record_session --session-id <ID>
python scripts/run_dual_track.py --session-id <ID>
python scripts/bootstrap_norms.py --norm-id synthetic_bootstrap_v1
python scripts/extract_section_heatmaps.py --session-id <ID> --uniform-heatmap --refresh-sections
python scripts/analyze_session.py --session-id <ID> --norm-id synthetic_bootstrap_v1 --website --ground
python scripts/export_ux_viewer.py --session-id <ID>
```

**Or orchestrator (must pass flags explicitly until runbook updated):**

```powershell
python scripts/run_website_session.py --stage all `
  --script configs/walkthrough_scripts/localhost_demo.yaml `
  --norm-id synthetic_bootstrap_v1 `
  --uniform-heatmap --refresh-sections --website --ground
# then if export_viewer not in your CLI defaults:
python scripts/export_ux_viewer.py --session-id <ID>
```

**Verify:**

| Check | Artifact / field |
|-------|------------------|
| `preds.shape[0]` ≈ `len(dom_snapshots)` (Δ ≤ 1) | manifest `alignment.preds_validation.mapping` |
| `grounding_triggers` non-empty when emotion Z high | `analysis_bundle.json` |
| `events[].triggers.emotion_channel` set (not anger-only) | `analysis_bundle.json` |
| `events[].heatmap_provenance.placeholder == true` OK for this layer | events |
| `section_report` populated | `analysis_bundle.json` |
| Viewer opens | `scout_data/sessions/<ID>/ux_viewer/index.html?base=.` |

### Layer 3 — Production heatmaps + feature isolation (Modal GPU)

On the **same** `<ID>` after Layer 2:

```powershell
python scripts/extract_section_heatmaps.py --session-id <ID> --modal --refresh-sections --force
python scripts/analyze_session.py --session-id <ID> --norm-id synthetic_bootstrap_v1 --website --ground
python scripts/export_ux_viewer.py --session-id <ID>
```

**Verify (feature analysis / isolation):**

| Check | Artifact / field |
|-------|------------------|
| `heatmaps/manifest.json` entries with `"source": "modal"` | heatmaps manifest |
| No silent all-placeholder after `--modal` (script prints `modal=N ...`) | stdout |
| `frames/t_<N>.jpg` exist for grounded timesteps | `frames/` |
| `events[]` non-empty when triggers exist; each has `top_elements` or grounding payload | `analysis_bundle.json` |
| `section_report[].heatmap_stats.real > 0` for at least one section | `analysis_bundle.json` |
| `section_report[].top_elements` not all low-confidence placeholder | section_report |
| `viewer_bundle.json` → `alignment`, `heatmap_provenance`, `manifest_n_timesteps` vs `analysis_n_timesteps` | ux_viewer |
| DOM overlay only on in-bounds timesteps | viewer UI |

**Failure triage order:** capture drift → TRIBE `preds` length → dual-track empty → ffmpeg frames → Modal heatmap zero yield → grounding thresholds → section_pipeline bounds → viewer export.

### Layer 4 — Benchmark repeatability (week 2)

1. Add `configs/walkthrough_scripts/` YAMLs: **local demo** (exists), **simple marketing site**, **interaction-heavy site** (deterministic steps, fixed viewport 1920×1080).
2. Run Layers 1–3 once per script; record `session_id`, stage of first failure, and alignment `mapping`.
3. Pick best session as **canonical demo**; copy path into sprint note.

---

## Week 1 — Prove the spine

### 1. Freeze scope (day 1)

- Short status note: website pipeline = mainline; NeuroEmo phases 3–6 = research-only.
- Do not merge new `scout_data/neuroemo/models/*` experiments into website runtime.

### 2. Layers 0 → 3 on localhost demo

- Execute testing ladder above; file blockers as E2E issues only (no refactors).
- Fix `run_website_session.py` / runbook if `--stage all` omits `--ground`, `--website`, or `export_viewer`.

### 3. Golden session artifact audit

- One session folder passes Layer 3 checklist end-to-end.
- Attach session ID and screenshot/note to `coding-sessions/` sprint log.

## Week 2 — Repeatability and demo package

### 4. Two additional walkthrough YAMLs + 3 benchmark runs

- Same command sequence; triage failures by stage.
- Only fix issues that improve reliability or interpretability of artifacts.

### 5. Output quality at artifact boundary (if time)

Targeted only if Layers 0–3 pass:

- [scout_core/dom_intersect.py](../../scout_core/dom_intersect.py) — winner stability
- [scout_core/section_analytics.py](../../scout_core/section_analytics.py) — section rollup clarity
- [scripts/generate_session_narrative.py](../../scripts/generate_session_narrative.py) — do not overclaim on placeholder heatmaps
- [viewer/ux_session_viewer.html](../../viewer/ux_session_viewer.html) — alignment/provenance UX

### 6. Close with demo package

Deliver:

- one canonical `scout_data/sessions/<id>/` path,
- updated [docs/runbooks/website-session.md](../../docs/runbooks/website-session.md) matching real commands,
- **What works / what’s manual / what’s zero-shot** (3–5 bullets),
- go/no-go on NeuroEmo runtime integration.

## Explicit no-work list

Unless it blocks Layer 3:

- NeuroEmo timing grid / dynamic feature bundles / hierarchical models
- MLP or specialist tuning
- New dataset-prep iterations
- Demographic clustering and `inference_mux`
- Atlas replacement for emotion classification
- 3D streaming brain viewer
- Platform abstractions beyond website-session path

## Decision at end of two weeks

### If proof point succeeds

- Harden artifacts and viewer/report UX.
- Then evaluate supervised NeuroEmo (likely **valence binary** first per May 28 surface matrix; not 5-class replacement).

### If proof point fails

- Narrow to **one** golden walkthrough.
- Drop optional stages (`narrative`, Modal heatmaps) until capture → preds → dual-track → viewer works.
- Do not resume NeuroEmo integration until spine is green.

## Practical heuristic

Every task must answer **yes** to at least one:

1. Does it advance **Layer 0–3** on a real session?
2. Does it make [scripts/run_website_session.py](../../scripts/run_website_session.py) or the runbook more reproducible?
3. Does it make exported artifacts easier for a non-author to trust?

Otherwise defer it.

## Related plans (reference only)

| Plan | Relevance this sprint |
|------|------------------------|
| [playwright-pipeline-fixes_0e4e0672.plan.md](playwright-pipeline-fixes_0e4e0672.plan.md) | Implementation done; **verification** is this sprint’s job |
| [neuroemo_remaining_high_leverage_opportunities_2026-05-27.plan.md](neuroemo_remaining_high_leverage_opportunities_2026-05-27.plan.md) | Phases 3–6 **frozen** |
| [salience-phased-delivery-plan.md](../../docs/implementation-plans/salience-phased-delivery-plan.md) | Close **P0-C1–C4** via testing ladder |
| [feature_isolation.md](../../docs/implementation-plans/feature_isolation.md) | Defines heatmap + DOM intersection semantics for Layer 3 checks |
