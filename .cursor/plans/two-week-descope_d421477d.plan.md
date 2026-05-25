---
name: two-week-descope
overview: "Refocus the repo on one repeatable Neural-UX Scout prototype spine over the next two weeks: website capture to TRIBE inference to zero-shot analysis to grounded viewer/report, while freezing parallel R&D that is not yet integrated. The plan defines what to freeze, what to postpone, and the next proof point that would show the project is converging."
todos:
  - id: freeze-scope
    content: Declare the zero-shot website-session pipeline as the only mainline for the next two weeks and freeze NeuroEmo, multiplex, and 3D expansion work.
    status: pending
  - id: stabilize-spine
    content: Harden capture, alignment, analysis, and export around the existing website-session pipeline until 3 fixed walkthroughs run repeatably.
    status: pending
  - id: document-runbook
    content: Consolidate one clean operator runbook for the exact mainline command flow and required prerequisites.
    status: pending
  - id: run-proof-point
    content: Execute 3 to 5 benchmark sessions, fix only end-to-end blockers, and collect one canonical demo session with honest limitations.
    status: pending
  - id: end-of-sprint-decision
    content: Use the proof point result to decide whether to keep hardening the zero-shot product spine or narrow further before resuming broader R&D.
    status: pending
isProject: false
---

# Two-Week De-Scope Plan

## Goal
Use the next two weeks to prove that the current repo can reliably produce one stakeholder-ready Neural-UX Scout session artifact from a real website walkthrough, without expanding the product surface further. The mainline stays the existing zero-shot website-session pipeline, while the NeuroEmo supervised branch is treated as bounded R&D rather than an immediate integration target.

## Reasoning
The repository already has a real operator spine in [scripts/run_website_session.py](C:/Users/facebook/Desktop/TribeV2/scripts/run_website_session.py), with stages for capture, TRIBE inference, dual-track scoring, heatmaps, analysis, narrative, and viewer export. The main implementation plan in [docs/implementation-plans/neural-ux-scout-phased-delivery-plan.md](C:/Users/facebook/Desktop/TribeV2/docs/implementation-plans/neural-ux-scout-phased-delivery-plan.md) explicitly says to keep "one runnable spine first" and delay scale/complexity until that spine is stable. The recent NeuroEmo work in [coding-sessions/2026-05-25-neuroemo-training-modeling-upgrade.md](C:/Users/facebook/Desktop/TribeV2/coding-sessions/2026-05-25-neuroemo-training-modeling-upgrade.md) is valuable, but it also states that inference should be added only after the model format stabilizes. That means the current struggle is less about a bad architecture and more about too many active fronts.

## Two-Week Scope Decision
### Keep as the mainline
- The website-session prototype path built around:
  - [scripts/record_website_session.py](C:/Users/facebook/Desktop/TribeV2/scripts/record_website_session.py)
  - [scripts/run_website_session.py](C:/Users/facebook/Desktop/TribeV2/scripts/run_website_session.py)
  - [scripts/run_dual_track.py](C:/Users/facebook/Desktop/TribeV2/scripts/run_dual_track.py)
  - [scripts/analyze_session.py](C:/Users/facebook/Desktop/TribeV2/scripts/analyze_session.py)
  - [scripts/extract_section_heatmaps.py](C:/Users/facebook/Desktop/TribeV2/scripts/extract_section_heatmaps.py)
  - [scripts/export_ux_viewer.py](C:/Users/facebook/Desktop/TribeV2/scripts/export_ux_viewer.py)
  - [viewer/ux_session_viewer.html](C:/Users/facebook/Desktop/TribeV2/viewer/ux_session_viewer.html)
- The current zero-shot interpretation layer in [scout_core/dual_track.py](C:/Users/facebook/Desktop/TribeV2/scout_core/dual_track.py), because it is already part of the end-to-end runtime path.
- The manifest and bundle contracts in [scout_core/schemas.py](C:/Users/facebook/Desktop/TribeV2/scout_core/schemas.py), because convergence depends on stabilizing outputs, not adding more branches.

### Freeze for two weeks
- New NeuroEmo model variants, atlas upgrades, classifier experiments, and training-path refactors under:
  - [scripts/prepare_neuroemo_tribev2.py](C:/Users/facebook/Desktop/TribeV2/scripts/prepare_neuroemo_tribev2.py)
  - [scripts/train_neuroemo_emotion_model.py](C:/Users/facebook/Desktop/TribeV2/scripts/train_neuroemo_emotion_model.py)
  - [scripts/train_neuroemo_mlp_model.py](C:/Users/facebook/Desktop/TribeV2/scripts/train_neuroemo_mlp_model.py)
  - [scripts/train_neuroemo_specialist_models.py](C:/Users/facebook/Desktop/TribeV2/scripts/train_neuroemo_specialist_models.py)
  - [docs/implementation-plans/improved-roi-extraction-pipeline-plan.md](C:/Users/facebook/Desktop/TribeV2/docs/implementation-plans/improved-roi-extraction-pipeline-plan.md)
- New demographic multiplex / cluster-training work described in [docs/implementation-plans/demographic-multiplexer-implementation-plan.md](C:/Users/facebook/Desktop/TribeV2/docs/implementation-plans/demographic-multiplexer-implementation-plan.md) and [docs/implementation-plans/neural-ux-scout-architecture-plan.md](C:/Users/facebook/Desktop/TribeV2/docs/implementation-plans/neural-ux-scout-architecture-plan.md).
- Any new 3D streaming viewer work beyond the existing static UX viewer.

### Postpone until after the proof point
- Runtime integration of supervised NeuroEmo inference.
- Cluster embeddings, `inference_mux`, and demographic prototype scaling.
- Browser streaming 3D cortex sync.
- Further atlas/parcellation sophistication unless directly required to make the current runtime path work.

## Success Metric For The Two Weeks
By the end of the two weeks, a non-author should be able to run one documented command flow from a clean checkout and obtain a session folder containing a valid walkthrough video, `session_manifest.json`, `preds.npz`, `analysis_bundle.json`, and exported UX viewer output for at least 3 real walkthroughs that do not require hand repair.

## Next Proof Point
### Proof point statement
The next proof point is not "better model accuracy." It is:

> One repeatable website-session demo on several real sites, using the current zero-shot stack, where the viewer output is coherent enough that someone outside the implementation loop can inspect the resulting UX artifacts without reading Python internals.

### Acceptance criteria
- 3 to 5 walkthrough sessions can be run through the same documented flow.
- Each session produces the same expected artifact set and passes alignment validation.
- At least one session yields a viewer/report that feels explainable end-to-end: spike -> heatmap -> DOM candidate -> section summary -> narrative.
- The README/runbook can be followed by someone who did not build the pipeline.
- Known limitations are documented honestly: synthetic norms, zero-shot caveats, and operator prerequisites.

## Week 1
### 1. Freeze scope and establish the mainline path
- Declare the website-session zero-shot path as the only shipping track for the next two weeks in a short status note.
- Mark NeuroEmo and multiplex work as research-only for this sprint.
- Update the checklist in [docs/implementation-plans/neural-ux-scout-architecture-plan.md](C:/Users/facebook/Desktop/TribeV2/docs/implementation-plans/neural-ux-scout-architecture-plan.md) or a nearby sprint note so the current priority is explicit.

### 2. Harden the runnable spine
Focus only on issues that block repeatable sessions in:
- [scripts/run_website_session.py](C:/Users/facebook/Desktop/TribeV2/scripts/run_website_session.py)
- [scripts/record_website_session.py](C:/Users/facebook/Desktop/TribeV2/scripts/record_website_session.py)
- [scout_core/session_align.py](C:/Users/facebook/Desktop/TribeV2/scout_core/session_align.py)
- [scout_core/walkthrough.py](C:/Users/facebook/Desktop/TribeV2/scout_core/walkthrough.py)
- [scout_core/section_pipeline.py](C:/Users/facebook/Desktop/TribeV2/scout_core/section_pipeline.py)

The target is not elegance. The target is to make session capture, alignment, analysis, and export boringly repeatable.

### 3. Pick 3 fixed fixture sites / flows
- Choose 3 representative walkthrough targets: one local/demo site, one simple marketing site, and one more interaction-heavy site.
- Write or tighten YAML walkthroughs so they are deterministic enough to compare runs.
- Treat those as the only benchmark set for the sprint.

### 4. Produce a single operator runbook
Consolidate the minimum reproducible flow in [docs/runbooks/website-session.md](C:/Users/facebook/Desktop/TribeV2/docs/runbooks/website-session.md) and keep it current with the exact mainline command sequence. Avoid documenting branches that are not part of the current proof point.

## Week 2
### 5. Run the benchmark sessions and triage only end-to-end blockers
- Execute the 3 fixed sessions.
- Capture where failures occur: capture drift, TRIBE stage, baseline handling, heatmap extraction, DOM grounding, narrative generation, or viewer export.
- Fix only issues that directly improve end-to-end reliability or output coherence.

### 6. Tighten output quality at the artifact boundary
Improve only the parts that affect demo usefulness in:
- [scout_core/dom_intersect.py](C:/Users/facebook/Desktop/TribeV2/scout_core/dom_intersect.py)
- [scout_core/section_analytics.py](C:/Users/facebook/Desktop/TribeV2/scout_core/section_analytics.py)
- [scripts/generate_session_narrative.py](C:/Users/facebook/Desktop/TribeV2/scripts/generate_session_narrative.py)
- [viewer/ux_session_viewer.html](C:/Users/facebook/Desktop/TribeV2/viewer/ux_session_viewer.html)

Only make changes that improve interpretability of the existing artifact chain. Do not broaden scope.

### 7. Close with a demo package and honest limitations
At the end of week 2, produce:
- one best session folder as the canonical demo example,
- a short summary of what works reliably,
- a short list of what remains manual or weak,
- a clear go/no-go decision on whether the current zero-shot product spine is strong enough to continue hardening before returning to NeuroEmo integration.

## Explicit No-Work List For These Two Weeks
Do not spend time on these unless they block the proof point directly:
- atlas replacement or ROI research,
- MLP or specialist classifier tuning,
- new NeuroEmo dataset-prep iterations,
- demographic clustering,
- cluster-embedding training,
- runtime supervised inference integration,
- 3D streaming brain viewer,
- generalized platform abstractions beyond what the current website-session path needs.

## Decision At The End Of Two Weeks
### If the proof point succeeds
Continue investing in the mainline product spine first:
- standardize artifacts further,
- strengthen the viewer/report experience,
- then decide whether supervised NeuroEmo integration is the next best lever.

### If the proof point fails
Do not keep broadening the application. Narrow further:
- reduce to one golden walkthrough,
- remove optional stages,
- and prove a single reliable path before revisiting new modeling work.

## Practical heuristic
For the next two weeks, every task should pass this filter:

- Does it make [scripts/run_website_session.py](C:/Users/facebook/Desktop/TribeV2/scripts/run_website_session.py) more reliable, more reproducible, or more interpretable?
- Does it make the exported session artifacts easier for a non-author to trust?

If the answer is no, defer it.