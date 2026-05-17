# Neural-UX Scout — Phased delivery & checkpoints

This document breaks [`NEURAL_UX_SCOUT_IMPLEMENTATION_PLAN.md`](NEURAL_UX_SCOUT_IMPLEMENTATION_PLAN.md) into **concrete phases** with **exit criteria** you can verify independently. The overall vision is large (research-grade neuro-modeling + realtime 3D + agentic UX); these phases are deliberately **narrow** so each gate delivers a demo-able artifact before scope expands.

**How to use this file**

- Treat each **Checkpoint** as a binary pass/fail before moving on (or explicitly defer with a recorded risk).
- **Parallel tracks** are called out where research (TRIBE internals) can proceed without blocking UI skeleton work.
- Prefer **surface / fsaverage5** outputs until Phase F explicitly expands volumetrics.

---

## Guiding principles

1. **One runnable spine first:** video in → Tribe predict → artifact out → viewer consumes something (even if ugly).
2. **Contracts before scale:** freeze `session_manifest` / `analysis_bundle` shapes early; version them (`schema_version`).
3. **Multiplex second:** prove single-subject path on Modal with stable memory, then add `K` clusters with micro-batching.
4. **Interpretation last-mile:** MVPA probability traces from trained models before LLM prose; never invert that order for analytics truthfulness.
5. **Ethics & reporting floors:** define minimum cluster size and suppressed outputs before shipping comparative demographics UI.

---

## Phase map (summary)

| Id | Name | Primary outcome |
|----|------|-----------------|
| **R0** | TRIBE research spike | Know where Stage 5 subject conditioning lives; VRAM curve for one forward |
| **P0** | Baseline capture + inference | Manifest + Modal job reproduces current `tribe.py` parity |
| **P1** | Multiplex inference | `vertex_ts[K,T,V]` (or chunked) + `analysis_bundle` v0 |
| **P2A** | MVPA training data | Behavioral proxy labels + frozen TRIBE features produce validated `.pkl` classifiers |
| **P2** | Parcellation & MVPA inference | Vertex → Yeo-7 surface masking → MVPA probability classifications |
| **P3** | Streaming viewer | Browser: video + brain mesh + timeline sync (degraded mode OK) |
| **P4** | Neural barriers & grounding | Divergence windows ↔ DOM selectors / evidence bundle |
| **P5** | Data & cluster training | Demographic clustering artifacts + Modal train job v1 |
| **F** | Stretch: volume / subcortex | Only after surface path is production-stable |

Tracks **R0** and **P0–P1** can overlap only after R0 answers “can we inject batched subject/cluster embeddings without rewriting encoders?”

---

## R0 — TRIBE v2 research spike (blocking technical unknowns)

**Goal:** Replace hand-wavy multiplex assumptions with **verified** hooks in TRIBE’s forward (subject ID / embedding injection point, tensor ranks, what can be frozen vs duplicated).

**Deliverables**

- Short internal note (even a markdown section appended to the implementation plan): diagram of data flow through encoders → Stage 5 subject block → decoder/heads → surface predictions.
- Table of tensor shapes for **one** representative clip (T, V, batch dims).
- Measured **peak GPU memory** for single-subject inference on target GPU class (A100).

**Checkpoints**

- [ ] **R0-C1:** Located exact module/function where subject identity enters the graph (file + symbol name).
- [ ] **R0-C2:** Documented whether subject conditioning is **additive**, **concat**, **FiLM**, lookup table, or other — with code citation.
- [ ] **R0-C3:** Feasibility verdict for **encode-once, many subjects/clusters** without redundant encoder forward (yes / yes-with-patch / no).
- [ ] **R0-C4:** Recorded **VRAM** at batch size 1 and rough scaling hypothesis for `K` (even if linear fallback is micro-batching).

**Exit gate:** R0-C1–C4 complete → green-light multiplex design in P1 (else narrow P1 to sequential micro-batch only).

---

## P0 — Baseline pipeline: session artifacts + Modal parity

**Goal:** End-to-end **single demographic / single subject configuration** matching today’s [`tribe.py`](tribe.py): upload clip → `predict` → persisted outputs suitable for downstream tooling.

**Deliverables**

- **`session_manifest.json` v0** aligned with implementation plan (FPS, PTS mapping, URL, run id, optional DOM stub).
- Modal worker method(s) that accept video bytes and return **prediction arrays** + metadata (not only MP4).
- Frozen **`analysis_bundle.json` v0** stub (predictions + empty barriers/events) so the schema exists.

**Checkpoints**

- [ ] **P0-C1:** Playwright or manual recorder produces **manifest + mp4** for one golden URL (repeatable fixture).
- [ ] **P0-C2:** Modal inference output reproduces **within tolerance** the local/offline reference prediction for the same clip (define tolerance: e.g. MAE / correlation threshold after alignment).
- [ ] **P0-C3:** Artifact bundle written to durable storage (Volume/S3/local) with documented layout.
- [ ] **P0-C4:** README-level **runbook**: commands to reproduce P0 from clean checkout.

**Exit gate:** Non-author can run one session and inspect JSON + arrays without reading Python internals.

---

## P1 — Multiplex inference (cluster prototypes, OOM-safe)

**Goal:** Given **K cluster IDs**, produce **`vertex_ts`** shaped **[K, T, V]** or **chunked streams** along `T` or `K`, without OOM; encoder runs **once per clip** per R0 verdict.

**Deliverables**

- `infer_mux` API as in implementation plan (`cluster_ids`, optional `K_chunk`).
- **Micro-batch policy** documented with default chunk sizes per GPU.
- **`analysis_bundle.json` v1** includes per-cluster summary pointers (even if full vertex cube is file-backed).

**Checkpoints**

- [ ] **P1-C1:** Correctness: `K=1` matches P0 outputs bit-for-bit (or within agreed float tolerance).
- [ ] **P1-C2:** Stability: run `K ∈ {4, 16, 32}` on longest fixture clip without OOM; log peak VRAM.
- [ ] **P1-C3:** Performance: wall-clock vs `K` documented (linear ideal vs measured).
- [ ] **P1-C4:** Failure modes: OOM fallback strategy verified (smaller `K_chunk`, sequential fallback).

**Exit gate:** You can demo “same video, N demographic prototypes” with persisted outputs.

---

## P2A — MVPA training data (behavioral proxy labels)

**Goal:** Build the production `.pkl` models before runtime inference depends on them: collect UX recordings, create behavioral proxy labels from Playwright traces, generate frozen TRIBE features, and validate with video-held-out CV.

**Deliverables**

- Dataset manifest for **N=50 UX screen recordings** with video path, URL/session metadata, and Playwright DOM trace pointer.
- Behavioral proxy label builder for `$y`, starting with **Rage Clicks** and related DOM proxy events (`rapid repeated clicks`, `failed submit loops`, `backtrack after interaction`) mapped to Frustration/Cognitive Load labels.
- Frozen TRIBE feature builder for `$X`: `preds[T,V]` → Yeo-7 surface mask → 3-second sliding-window flattened vectors.
- `train_mvpa_model.py` outputs calibrated `scikit-learn` `.pkl` pipelines (e.g. `LinearSVC` + calibration), `label_map.json`, and LOVO metrics.

**Checkpoints**

- [ ] **P2A-C1:** Collected **N=50** UX screen recordings with matching Playwright DOM traces and usable TRIBE predictions.
- [ ] **P2A-C2:** `$y` labels generated via DOM proxy rules, including Rage Clicks, with a spot-check audit log for label quality.
- [ ] **P2A-C3:** `$X` features generated via frozen TRIBE and 3-second masked sliding windows; feature shapes documented per target mask.
- [ ] **P2A-C4:** Leave-One-Video-Out CV achieves **macro F1 > 0.75** for the target Frustration/Cognitive Load classifier, or the phase records why the model is not production-ready.

**Exit gate:** `model.pkl` is registered with metadata, LOVO macro F1 > 0.75, and runtime code can load it without retraining.

---

## P2 — Parcellation + MVPA inference (Yeo-7 surface masks, guardrails)

**Goal:** Turn vertex traces into **Yeo-7 surface-masked MVPA features** and continuous cognitive-state probability traces — **no LLM required** for core scoring.

**Deliverables**

- Vertex → **Yeo-7** mapping artifact (`csv`/`npz`) tied to **fsaverage5** vertex indexing used by Tribe.
- **`scout_core/mvpa_engine.py`**: applies Yeo-7 `SurfaceMasker`, builds 3-second flattened windows, and loads the pre-trained `.pkl` classifier.
- Removal/deprecation of **`emotion_rules.yaml`** from the production inference path.
- **`analysis_bundle.json` v2** adds `probability_traces`, `mvpa_probability_classifications`, model metadata, and confidence/calibration fields.

**Checkpoints**

- [ ] **P2-C1:** Mapping covers **100%** of mesh vertices or documents masked vertices explicitly.
- [ ] **P2-C2:** Surface masking reproducible: given fixed inputs, masked vertex order and sliding-window feature vectors are stable across runs.
- [ ] **P2-C3:** Every emitted MVPA probability classification includes **`confidence`** + **`evidence`** (model id, mask, time range, probability/slope).
- [ ] **P2-C4:** Copy review: language is **non-diagnostic** (hypothesis framing only).

**Exit gate:** System outputs a continuous probability trace for Frustration/Cognitive Load based on SVM weights, not averages.

---

## P3 — Streaming visualization (browser-first)

**Goal:** Side-by-side **walkthrough video** + **3D cortical activity** + **cluster/MVPA probability panels**, time-aligned; accepts degraded mode (lower FPS mesh updates).

**Deliverables**

- Minimal web app: video player + WebSocket client + Three/VTK canvas.
- **Transport:** mesh once + quantized per-frame colors **or** documented fallback to tiled MP4/WebRTC.
- UX: cluster legend + scrubber drives **same frame index** as inference.

**Checkpoints**

- [ ] **P3-C1:** Sync accuracy: scrubbing lands within **≤1 frame** of intended index after buffering rules applied.
- [ ] **P3-C2:** Payload budget: median/max bandwidth documented for one session at target FPS.
- [ ] **P3-C3:** Degraded mode: if WS stalls, UI shows **MVPA probability traces** without crashing.
- [ ] **P3-C4:** Accessibility pass on viewer chrome (keyboard scrubbing, contrast on legend).

**Exit gate:** Stakeholder demo without pointing them at Jupyter or Modal logs.

---

## P4 — Neural barriers & agentic grounding

**Goal:** Detect **population divergence** (“Neural Barrier”) and tie it to **specific UI elements** via Playwright traces.

**Deliverables**

- `barriers.py` logic: pairwise cluster divergence + persistence criteria + linkage to P2 MVPA probability classifications.
- DOM timeline store (selector, bbox, visibility) correlated to timestamps.
- Ranked **issue list** export suitable for design/engineering tickets.

**Checkpoints**

- [ ] **P4-C1:** Synthetic test: injected divergence produces a barrier with correct window alignment.
- [ ] **P4-C2:** Real session: top 3 barriers each point to **≥1 DOM candidate** with screenshot or bbox crop.
- [ ] **P4-C3:** False-positive review checklist completed on **N≥5** diverse sites (manual QA rubric).
- [ ] **P4-C4:** Privacy review for DOM captures (PII redaction policy documented).

**Exit gate:** “Why Senior probability trace rose but Gen‑Z didn’t” has a **UI artifact pointer**, not only a chart.

---

## P5 — Demographic data prep & Modal training (cluster embeddings)

**Goal:** Offline pipeline from TRIBE dataset metadata → **clusters** → **learned prototype embeddings** trained under frozen encoders on Modal.

**Deliverables**

- `slice_dataset.py` outputs: `cluster_members.parquet`, `cluster_demographics.json`, ethics floors.
- `train_clusters.py`: checkpointing to Volume; eval harness on held-out clips.
- Integration flag: inference can load **trained table** vs baseline defaults.

**Checkpoints**

- [ ] **P5-C1:** No **subject leakage** between splits (automated test).
- [ ] **P5-C2:** Cluster **minimum n** enforced; small clusters suppressed in UI.
- [ ] **P5-C3:** Training loss curve + simple downstream metric (e.g. prediction error vs cluster held-out) tracked per run.
- [ ] **P5-C4:** Cost estimate documented ($/hr × step time) for one full experiment.

**Exit gate:** New clusters materially change **`analysis_bundle`** probability traces vs baseline cluster prototypes — or consciously ship “v1 equals centroid-of-subjects” with documented limitation.

---

## F — Stretch: volumetric / subcortex / “70k voxels”

**Goal:** Only after **P3** stability: optional volumetric reconstruction or expanded mesh that matches scientific claims.

**Checkpoints**

- [ ] **F-C1:** Explicit forward model doc (what is being visualized).
- [ ] **F-C2:** Validation against surface MVP (consistency checks).
- [ ] **F-C3:** Performance acceptance for realtime/streaming (likely not realtime without heavy approximation).

**Exit gate:** Marketing language matches geometry (vertices vs voxels).

---

## Cross-cutting checkpoints (every phase)

Apply incrementally; don’t punt all to the end.

- **Security:** secrets only via Modal Secret / env; no tokens in manifests or browser bundles.
- **Reproducibility:** pinned Tribev2 commit, Docker/Modal image digest, seed strategy for training.
- **Observability:** structured logs per session id; timing breakdown (encode / predict / mask / MVPA / stream).
- **Documentation:** each phase updates a single “current demo script” section in README when P0+.

---

## Suggested sequencing when overwhelmed

1. **R0** (short spike, hard stop if multiplex looks infeasible without fork).
2. **P0** → **P2A** on offline fixtures **before** perfect Playwright autonomy (use manual captures plus DOM proxy labels).
3. **P2** once P2A has a registered `.pkl` model.
4. **P1** once R0+P0 stable.
5. **P3** in parallel with **P2** using mocked `analysis_bundle` JSON.
6. **P4** once P2+P3 share a timeline contract.
7. **P5** when you have dataset access rights and ethics sign-off — often last among core phases.

---

## Relationship to the implementation checklist

The checklist at the top of [`NEURAL_UX_SCOUT_IMPLEMENTATION_PLAN.md`](NEURAL_UX_SCOUT_IMPLEMENTATION_PLAN.md) maps roughly as:

| Plan checklist item | Phase |
|---------------------|-------|
| Vendor/pin tribev2; Stage 5 spike; VRAM | **R0**, **P0** |
| Manifest + `analysis_bundle` schemas | **P0** |
| Vertex→Yeo7 + MVPA probability classifications | **P2A**, **P2** |
| Modal `inference_mux` + streaming encoder | **P1**, **P3** |
| `viz_web` viewer | **P3** |
| Barrier detector + DOM intersection | **P4** |
| Data prep + clustering scripts | **P5** |
| Modal `train_clusters.py` | **P5** |

Update **this** file when phase definitions change (merge splits, add gates). Keep architectural rationale in the implementation plan.
