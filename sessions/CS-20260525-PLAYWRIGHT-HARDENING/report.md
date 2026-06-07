# Coding Session - May 25, 2026

**Session ID:** `CS-20260525-PLAYWRIGHT-HARDENING`  
**Agent signature:** `Historical Cursor coding agent`  
**Context contract:** Future agents may cite this session ID when asking about timeline-correct capture, alignment/provenance, grounding hardening, and the pipeline-runner subagent quickstart.  
**Metadata normalized by:** `GPT-5.5, 2026-06-07`

## Playwright Pipeline Hardening, Alignment/Provenance, and Cursor Subagent Quickstart

**Scope:** Timeline-correct Playwright capture, richer alignment reporting, non-anger grounding support, heatmap provenance/overwrite hardening, downstream viewer/export cleanup, regression coverage, and a new project-level Cursor subagent for running and debugging the full pipeline quickly.

---

## 1. Goal

The goal of this session was to make the website pipeline trustworthy end-to-end.

We were fixing four real failure modes discovered while testing the existing Playwright + TRIBEv2 + dual-track + feature-isolation workflow:

1. DOM snapshots were being captured after all scripted actions instead of during the walkthrough.
2. Grounding selection was effectively anger-only, so valid non-anger spikes could never ground.
3. Real Modal heatmaps could be silently blocked by old placeholder `.npy` files.
4. Downstream artifacts did not clearly expose alignment status or heatmap provenance.

This session also added a reusable Cursor subagent so the pipeline can be verified from chat without re-explaining all of the project context each time.

---

## 2. What Landed

### A. Timeline-correct Playwright capture

The capture path now samples DOM snapshots during the walkthrough instead of after the walkthrough is already over.

Primary files:

- `scout_core/walkthrough.py`
- `scripts/record_website_session.py`

Key changes:

- `scout_core.walkthrough.build_step_schedule()` now assigns each walkthrough step a cumulative wall-clock offset.
- `record_website_session_async()` now merges two event streams:
  - sample events every `interval_sec`
  - scripted step events at their scheduled offsets
- At the same timestamp, sampling runs before the action, so the snapshot reflects the page state immediately before that action fires.
- `wait_ms` no longer causes a double-sleep in the merged loop; it is represented as elapsed time in the schedule.
- `session_manifest.json` now records DOM state that is actually aligned to the recorded video time.

Why this matters:

- `dom_snapshots[t]` is now meaningful for grounding, sectioning, and later viewer overlays.
- A walkthrough like the localhost demo should now produce different `scrollY` values across snapshots instead of repeating the final state everywhere.

### B. Richer alignment reporting

Primary file:

- `scout_core/session_align.py`

Key changes:

- `validate_preds_manifest_alignment()` now reports:
  - `n_preds`
  - `n_snapshots`
  - `delta`
  - `overlap_start`
  - `overlap_end`
  - `out_of_range_count`
  - `mapping`
- Mapping is now explicit:
  - `"exact"`
  - `"tolerated"`
  - `"clamped"`
  - `"invalid"`
- `is_timestep_in_manifest()` was added so downstream code can skip invalid timesteps rather than quietly reaching for a nearest snapshot forever.

Why this matters:

- We can now tell whether a session is truly aligned or only barely acceptable.
- The viewer/export layer can surface this instead of hiding mismatch behind a warning string.

### C. Grounding is no longer anger-only

Primary files:

- `scripts/analyze_session.py`
- `configs/isolation_thresholds.yaml`

Key changes:

- `_select_spikes()` now supports any emotion channel above threshold instead of only `anger`.
- The winning emotion channel at each timestep is preserved as:
  - `emotion_channel`
  - `emotion_z`
- If the winning channel is `anger`, backward-compatible `kragel_anger_z` is still emitted.
- Config now uses `trigger.emotion_z_min` as the main threshold key.
- Deprecated anger-only config still has a compatibility path.

Why this matters:

- A valid `sadness`, `fear`, or other emotion spike can now produce a grounding event.
- The event payload keeps the actual winning channel instead of collapsing everything to an anger-specific field.

### D. Heatmap provenance and overwrite behavior are explicit

Primary files:

- `scripts/extract_section_heatmaps.py`
- `scout_core/heatmap_extract.py`
- `scout_core/schemas.py`

Key changes:

- `read_heatmaps_manifest()` was added.
- `write_heatmaps_manifest(..., merge=True)` now preserves prior provenance instead of replacing the whole manifest blindly.
- `extract_section_heatmaps.py` now supports `--force`.
- `--modal` will overwrite placeholder-derived heatmaps automatically.
- Heatmap manifest entries now clearly differentiate:
  - `source: "modal"`
  - `source: "uniform_placeholder"`
- ffmpeg failures are more actionable because `_extract_frame_ffmpeg()` now returns `(success, reason)`.
- Grounding events now carry `HeatmapProvenance` metadata:
  - `heatmap_path`
  - `source`
  - `placeholder`
  - `frame_path`
  - `sha256`

Why this matters:

- Real heatmaps are no longer silently blocked by old placeholder outputs.
- We can tell whether a grounded event or section rollup came from real DINO attention or a synthetic placeholder map.

### E. Downstream bundle/export behavior is more honest

Primary files:

- `scout_core/section_pipeline.py`
- `scripts/export_ux_viewer.py`
- `scout_core/llm_narrative.py`

Key changes:

- Section enrichment now skips timesteps outside manifest bounds instead of pretending a nearest DOM snapshot is always valid.
- Section outputs now include `heatmap_stats`:
  - `real`
  - `placeholder`
  - `low_confidence`
- Rolled-up section elements carry `heatmap_source`.
- `viewer_bundle.json` now exports:
  - `manifest_n_timesteps`
  - `analysis_n_timesteps`
  - `alignment`
  - `heatmap_provenance`
  - `in_manifest_bounds` for events
- `scout_core.llm_narrative.build_narrative_payload()` now prefers `mean_attention_density` and also carries heatmap source/stats into the narrative payload.

Why this matters:

- Reviewers can now see when a section or event is grounded by low-confidence placeholder heatmaps.
- Alignment mismatch is visible in the exported bundle instead of being silently ignored.

### F. Regression coverage was expanded

Primary and new tests:

- `tests/test_record_website_session.py`
- `tests/test_analyze_session_spikes.py`
- `tests/test_heatmap_provenance.py`

Coverage added this session includes:

- step scheduling and extended alignment mapping
- non-anger emotion triggers
- provenance on grounded events
- heatmap manifest merge behavior
- placeholder vs real heatmap semantics

---

## 3. Key Files To Know

### Capture and alignment

- `scout_core/walkthrough.py`
- `scripts/record_website_session.py`
- `scout_core/session_align.py`
- `configs/walkthrough_scripts/localhost_demo.yaml`

### Dual-track and grounding

- `scout_core/dual_track.py`
- `scripts/run_dual_track.py`
- `scripts/analyze_session.py`
- `configs/isolation_thresholds.yaml`

### Heatmaps and section enrichment

- `scripts/extract_section_heatmaps.py`
- `scout_core/heatmap_extract.py`
- `scout_core/dom_intersect.py`
- `scout_core/section_pipeline.py`
- `scout_core/section_analytics.py`

### Export and narrative

- `scripts/export_ux_viewer.py`
- `scout_core/llm_narrative.py`
- `scout_core/schemas.py`

### Cursor subagent

- `.cursor/agents/playwright-pipeline.md`

---

## 4. Quickstart

### Fastest manual verification path

From the repo root:

```powershell
.venv\Scripts\activate
python -m pytest tests/ -v --tb=short --ignore=tests/test_neuroemo_model.py
```

That is the quickest confidence check that the hardened pipeline code is still in a good state.

### Full pipeline quickstart

Use a fresh session ID or let the recorder create one for you:

```powershell
.venv\Scripts\activate

# 1. Record walkthrough + manifest
python scripts/record_website_session.py --script configs/walkthrough_scripts/localhost_demo.yaml

# 2. Run full-video TRIBEv2 inference on Modal
modal run tribe.py::record_session --session-id <ID>

# 3. Run dual-track emotion + engagement
python scripts/run_dual_track.py --session-id <ID>

# 4. Bootstrap norms if needed
python scripts/bootstrap_norms.py --norm-id synthetic_bootstrap_v1

# 5. Optional offline placeholder heatmaps
python scripts/extract_section_heatmaps.py --session-id <ID> --uniform-heatmap --refresh-sections

# 6. Real Modal heatmaps
python scripts/extract_section_heatmaps.py --session-id <ID> --modal --refresh-sections

# 7. Full website analysis + grounding
python scripts/analyze_session.py --session-id <ID> --norm-id synthetic_bootstrap_v1 --website --ground

# 8. Export viewer bundle
python scripts/export_ux_viewer.py --session-id <ID>
```

### What to inspect after a successful run

Check these files under `scout_data/sessions/<ID>/`:

- `session_manifest.json`
- `preds.npz`
- `analysis_bundle.json`
- `heatmaps/manifest.json`
- `ux_viewer/viewer_bundle.json`
- `ux_viewer/index.html`

---

## 5. How To Use The New Cursor Subagent

This session created a project-level Cursor subagent:

```text
.cursor/agents/playwright-pipeline.md
```

### What it is for

It is a repo-specific expert for the full:

```text
Playwright -> TRIBEv2 -> dual-track -> heatmap -> grounding -> viewer
```

pipeline.

It knows:

- the main files by name
- the normal command order
- where session artifacts land
- the common failure modes
- which tests to run first
- that Python and pytest commands should run inside `.venv`

### What it is expected to do when invoked

The subagent is designed to:

1. activate `.venv` for Python/test commands
2. run the pipeline-specific tests first
3. inspect session artifacts
4. diagnose failures from outputs and metadata
5. edit code if needed
6. re-run tests
7. run the full pipeline when a session ID or walkthrough target is provided

### Example prompts

Use prompts like these in Cursor:

```text
Use the playwright-pipeline subagent to verify the full website pipeline.
```

```text
Use the playwright-pipeline subagent to run the localhost demo end-to-end and tell me where it fails.
```

```text
Use the playwright-pipeline subagent to debug session e4cb5a6a3b834a639eefe2d9d17239de.
```

```text
Use the playwright-pipeline subagent to run the pipeline-specific tests and summarize any regressions.
```

### Why this is useful

Before this subagent existed, every verification run required reloading a lot of repo context into chat.

Now there is a single, reusable project-level agent prompt that already knows:

- the pipeline architecture
- the command sequence
- the artifact layout
- the known regressions and what files they usually come from

---

## 6. Artifact Semantics After Hardening

### `session_manifest.json`

Important expectations:

- `dom_snapshots[t]` should reflect the page state during capture, not after everything is done.
- `pts_sec` should correspond to actual sample timing from the merged capture loop.
- `scrollY` should vary across the walkthrough when the script scrolls.

### `analysis_bundle.json`

Important expectations:

- `grounding_triggers` may include any emotion channel.
- `events[].triggers` should now preserve `emotion_channel` and `emotion_z`.
- `events[].heatmap_provenance` should describe whether the heatmap was real or placeholder-derived.
- `section_report[].heatmap_stats` indicates whether section attention came from real vs placeholder heatmaps.

### `heatmaps/manifest.json`

Important expectations:

- each sampled timestep should have a provenance entry
- `source` should distinguish real vs placeholder heatmaps
- merged writes should preserve old entries unless a timestep is intentionally replaced

### `ux_viewer/viewer_bundle.json`

Important expectations:

- `manifest_n_timesteps` and `analysis_n_timesteps` are distinct
- `alignment` should summarize the mapping quality
- `heatmap_provenance` should be exported for viewer/debugging use
- out-of-range events should be marked with `in_manifest_bounds`

---

## 7. Current Verification Targets

These are the most important checks for this hardened pipeline.

### Check 1 - capture really changed over time

Open `session_manifest.json` and confirm that different snapshots show different `scrollY` values for a scrolling walkthrough.

### Check 2 - non-anger emotion spikes can ground

Open `analysis_bundle.json` and confirm that a qualifying non-anger trigger can produce:

- an event in `events[]`
- `emotion_channel`
- `emotion_z`

### Check 3 - placeholder heatmaps do not masquerade as real ones

Open `heatmaps/manifest.json` and confirm entries clearly use:

- `source: "uniform_placeholder"`
- `source: "modal"`

### Check 4 - viewer export shows alignment status

Open `ux_viewer/viewer_bundle.json` and inspect:

- `manifest_n_timesteps`
- `analysis_n_timesteps`
- `alignment.mapping`
- `alignment.out_of_range_count`

### Check 5 - section outputs show confidence level

Open `analysis_bundle.json` and inspect:

- `section_report[].heatmap_stats.real`
- `section_report[].heatmap_stats.placeholder`
- `section_report[].heatmap_stats.low_confidence`

---

## 8. Test Commands

### Full repo regression pass

```powershell
.venv\Scripts\activate
python -m pytest tests/ -v --tb=short --ignore=tests/test_neuroemo_model.py
```

### Pipeline-focused test pass

```powershell
.venv\Scripts\activate
python -m pytest tests/test_record_website_session.py tests/test_analyze_session_spikes.py tests/test_heatmap_provenance.py tests/test_feature_isolation.py tests/test_dual_track.py tests/test_section_analytics.py -v --tb=short
```

### Tests added or expanded in this session

- `tests/test_record_website_session.py`
- `tests/test_analyze_session_spikes.py`
- `tests/test_heatmap_provenance.py`

These are the best first tests to run if you are touching:

- capture timing
- alignment semantics
- grounding trigger policy
- heatmap provenance behavior

---

## 9. Troubleshooting Guide

### Problem: all snapshots look like the final page state

Look at:

- `scout_core/walkthrough.py`
- `session_manifest.json`

Likely cause:

- capture is not happening inside the merged timeline loop

Expected fix path:

- verify `build_step_schedule()`
- verify sample events and step events are both present
- verify sampling runs before same-timestamp actions

### Problem: a sadness/fear spike is in `grounding_triggers` but no event appears

Look at:

- `scripts/analyze_session.py`
- `configs/isolation_thresholds.yaml`

Likely cause:

- threshold not met
- cooldown or max-spikes policy filtered it out
- heatmap or DOM snapshot missing

### Problem: `--modal` says heatmaps already exist

Look at:

- `heatmaps/manifest.json`
- `scripts/extract_section_heatmaps.py`

Expected behavior now:

- placeholder entries should be overwritten automatically by `--modal`
- use `--force` if you want to overwrite everything

### Problem: section outputs exist but are low confidence

Look at:

- `analysis_bundle.json`
- `section_report[].heatmap_stats`

Likely cause:

- only placeholder heatmaps were available for those sampled timesteps

### Problem: viewer bundle looks inconsistent with preds length

Look at:

- `ux_viewer/viewer_bundle.json`
- `scout_core/session_align.py`

Expected fields:

- `manifest_n_timesteps`
- `analysis_n_timesteps`
- `alignment.mapping`

---

## 10. New Project-Level Agent

This session also introduced a reusable project asset, not just code changes:

```text
.cursor/agents/playwright-pipeline.md
```

That file is now part of the project knowledge layer for future pipeline work.

It should be treated as the fastest way to rehydrate context for:

- end-to-end verification
- debugging broken sessions
- re-running the full command chain
- checking artifacts after a run
- regression testing after pipeline edits

---

## 11. Suggested Team Workflow Going Forward

When someone touches the Playwright pipeline again, the fastest safe workflow is:

1. run the pipeline-focused tests
2. use the `playwright-pipeline` subagent for end-to-end verification
3. generate a fresh localhost demo session
4. inspect `session_manifest.json`, `analysis_bundle.json`, and `heatmaps/manifest.json`
5. export `ux_viewer/viewer_bundle.json`

That keeps future work grounded in artifacts instead of assumptions.

---

## 12. File Inventory For This Session

### Code changes

- `scout_core/walkthrough.py`
- `scout_core/session_align.py`
- `scout_core/heatmap_extract.py`
- `scout_core/section_pipeline.py`
- `scout_core/llm_narrative.py`
- `scout_core/schemas.py`
- `scripts/record_website_session.py`
- `scripts/analyze_session.py`
- `scripts/extract_section_heatmaps.py`
- `scripts/export_ux_viewer.py`
- `configs/isolation_thresholds.yaml`

### Tests

- `tests/test_record_website_session.py`
- `tests/test_analyze_session_spikes.py`
- `tests/test_heatmap_provenance.py`

### Cursor project assets

- `.cursor/agents/playwright-pipeline.md`

### Session documentation

- `coding-sessions/2026-05-25-playwright-pipeline-hardening-and-subagent.md`

---

## 13. Bottom Line

After this session, the pipeline is materially safer to trust:

- capture timing is tied to the actual walkthrough
- grounding can fire on the correct emotion channel instead of only anger
- placeholder vs real heatmaps are distinguishable
- viewer/export artifacts surface alignment and provenance
- the repo now includes a dedicated Cursor subagent for quickly verifying or debugging the full pipeline without re-teaching the assistant the whole system

This session did not change the core product idea. It hardened the operational truthfulness of the existing pipeline so later evaluation is based on aligned data and explicit provenance instead of hidden assumptions.
