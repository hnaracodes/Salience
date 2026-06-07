# NeuroEmo: Modal Projection Equivalence + Phase 2 Matrix Execution (2026-05-28)

**Session ID:** `CS-20260528-VERTEX-EQUIV-PHASE2`  
**Agent signature:** `Historical Cursor coding agent`  
**Context contract:** Future agents may cite this session ID when asking about Modal projection equivalence proof, proof-hash propagation, and Phase 2 matrix execution.  
**Metadata normalized by:** `GPT-5.5, 2026-06-07`

This note consolidates work completed across:

- this chat session (Modal verifier bring-up, proof hash propagation, phase-2 matrix execution), and
- the earlier companion chat in `agent-transcripts/e08cca07-2be2-4587-a849-80737bddaf9d/e08cca07-2be2-4587-a849-80737bddaf9d.jsonl` (phase 2-end experiment machinery implementation).

The goal is to give teammates a complete handoff: what changed, what broke, what passed, what metrics moved, and what remains.

---

## 1) Objective and scope for today

Primary goals:

1. Finish Phase 1 trust/provenance gate by upgrading vertex proof from `mesh_identity_verified` to `projection_equivalence_verified` on Modal.
2. Propagate the new proof hash through manifest/dataset/docs so downstream training contracts are consistent.
3. Begin/execute Phase 2 surface-native postfix matrix runs under the frozen contract.
4. Keep this compatible with the user's local `.venv` workflow and Modal-only verification preference.

---

## 2) High-level outcomes

### 2.1 Vertex proof outcome

- **Status:** `projection_equivalence_verified` (success)
- **Report:** `scout_data/neuroemo/vertex_equivalence_report.json`
- **Proof hash:** `25ced97a324a16f4cbf07261e4e613cbb1897ff792d1af9f92a0235bf7c58a8c`
- **Projection comparison:** available + verified
  - `max_abs_diff = 0.0`
  - `mean_abs_diff = 0.0`
  - `allclose_atol = 1e-05`

### 2.2 Hash propagation outcome

- Manifest, NPZ metadata, and evaluation doc now reference the new proof hash.
- Old mesh-only hash (`ebcf...`) was removed from active references.

### 2.3 Phase 2 classical matrix runs executed

In `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix_phase2/`:

- `logistic_saga_5class_10tr`
- `sgd_logistic_5class_10tr`
- `linear_svc_5class_10tr`

All three completed and produced `.joblib` + `_metrics.json`.

---

## 3) Files added/changed today

### Verification and orchestration

- `tribe.py`
  - Added Modal local entrypoint for verifier orchestration:
    - `verify_vertex_equivalence(...)`
  - Keeps verification launched correctly as `modal run tribe.py::verify_vertex_equivalence`.

- `scripts/run_vertex_equivalence_verification.ps1`
  - Hardened runner for Windows + `.venv`.
  - Added explicit logging, UTF-8 handling, and robust Modal invocation.

- `scripts/verify_tribe_vertex_equivalence.py`
  - Modal-mode guardrail messaging to prevent incorrect `python ... --mode modal` usage and route users to the Modal entrypoint flow.

### Experiment machinery from companion chat (`e08...`)

- `scripts/run_neuroemo_experiment_matrix.py`
  - Phase 2-end matrix/timing/preproc/structured-decoding runner + command manifest generation.

- `scripts/train_neuroemo_hierarchical_model.py`
  - Coarse-to-fine (valence-style) hierarchical decoding path over existing shared data contract.

- `scripts/train_neuroemo_emotion_model.py`
  - Dynamic temporal reducer support and metadata contract expansion used by matrix workflows.

### Hash propagation + docs

- `configs/parcellation_manifest.yaml` (regenerated)
- `scout_data/neuroemo/tribev2_surface/subjects/*.npz` (40 files updated)
- `scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz` (updated)
- `docs/neuroemo-surface-native-schaefer-atlas-evaluation-2026-05-27.md` (proof hash updated)
- `scripts/run_surface_phase2_matrix.ps1` (new stable local phase-2 launcher)

---

## 4) Bug timeline and fixes

## Bug A: Modal verifier never actually launched remotely

### Symptom

- No active Modal app.
- Local log stalled after "Step 2/2".

### Root cause

- Command used `modal run scripts/verify_tribe_vertex_equivalence.py`.
- That script is a normal Python CLI, not a Modal local entrypoint target.

### Fix

- Added `@app.local_entrypoint()` in `tribe.py` (`verify_vertex_equivalence`).
- Updated runner to call:
  - `modal run tribe.py::verify_vertex_equivalence --bold-path ...`

---

## Bug B: PowerShell aborted on Modal stderr warnings

### Symptom

- Script stopped immediately with `NativeCommandError` wrappers.

### Root cause

- `$ErrorActionPreference = "Stop"` treated Modal stderr warnings (deprecations) as terminating pipeline errors.

### Fix

- Temporarily switched to `Continue` during Modal command execution, then restored prior behavior.

---

## Bug C: Windows `charmap` Unicode crash during Modal output piping

### Symptom

- `'charmap' codec can't encode character '\u2713'`.

### Root cause

- Modal CLI emits Unicode symbols (check marks/tree glyphs) not representable under default console encoding in the log pipe.

### Fix

- Forced UTF-8 environment/console in runner:
  - `PYTHONIOENCODING=utf-8`
  - `PYTHONUTF8=1`
  - `chcp 65001`
  - set console output encoding to UTF-8.

---

## Bug D: Matrix background launcher (`710971`) aborted immediately

### Symptom

- Background shell task ended in ~200ms.

### Root cause

- Fragile inline PowerShell command for long looped matrix command.

### Fix

- Added dedicated `scripts/run_surface_phase2_matrix.ps1`.
- Added same stderr handling fix used in verifier runner.
- Relaunched successfully; log reached completion.

---

## 5) Verification run interpretation

Run log: `scout_data/neuroemo/vertex_equivalence_run.log`  
Modal log: `scout_data/neuroemo/vertex_equivalence_modal.log`

### Important nuance

- During upload/runtime there were transient network exceptions (`Connection lost`, SSL/MAC, WinError 10054).
- Despite this, the run completed and emitted final report + success line:
  - `OK: proof_status='projection_equivalence_verified'`
  - `Done. Report: scout_data/neuroemo/vertex_equivalence_report.json`

### Interpretation

- The trust gap called out in plan Phase 1 is now closed for this sampled verification contract.
- Mesh identity and projector numerical equivalence both passed under the recorded parameters.

---

## 6) Hash propagation details

New canonical report hash:

- `25ced97a324a16f4cbf07261e4e613cbb1897ff792d1af9f92a0235bf7c58a8c`

Propagated to:

- `configs/parcellation_manifest.yaml` (`vertex_equivalence.report_sha256`)
- Subject NPZs + combined train NPZ (`vertex_equivalence_report_sha256`, `vertex_equivalence_json`, report path)
- `docs/neuroemo-surface-native-schaefer-atlas-evaluation-2026-05-27.md`

Old hash:

- `ebcf93a9b35fe788f7ee868f7e7a6baba30b59a030fd2b908d142fd4b172e9b9`

No longer present in active tracked references after propagation.

---

## 7) Phase 2 matrix results (classical subset executed now)

Output dir: `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix_phase2/`

| Run | Mean Acc | Mean Bal Acc | Mean Macro F1 | Mean Log Loss | Pooled Log Loss | Vertex Proof |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `logistic_saga_5class_10tr` | 0.4200 | 0.4200 | 0.4123 | 2.2437 | 2.2437 | projection verified (`25ced...`) |
| `sgd_logistic_5class_10tr` | 0.3800 | 0.3800 | 0.3711 | 22.1602 | 22.1602 | projection verified (`25ced...`) |
| `linear_svc_5class_10tr` | 0.3600 | 0.3600 | 0.3537 | 1.6391* | 1.6391* | projection verified (`25ced...`) |

\* `linear_svc` log-loss comparability should be treated carefully because probability calibration differs from logistic-style models.

### Comparison to prior surface logistic anchor

Prior file (`2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr_metrics.json`) had:

- same logistic metrics (`0.4200 / 0.4123 / 2.2437`)
- but with old `mesh_identity_verified` hash (`ebcf...`).

Interpretation: today's rerun is **contract/provenance uplift**, not a metric uplift, for logistic.

### Comparison to projected postfix baseline (documented anchor)

From plan/doc anchor:

- projected logistic mean acc `0.445`, macro F1 `0.433`, pooled log loss `~2.279`

Surface logistic remains below projected on top-1/macro-F1, but keeps slightly better log-loss trend already observed in prior evaluation notes.

---

## 8) What teammates should do next

1. Continue full Phase 2 parity matrix using `scripts/run_neuroemo_experiment_matrix.py` to include:
   - MLP 5-class, specialist variants, binary valence MLP.
2. Generate one consolidated comparison report against `2026-05-25_postfix_matrix`.
3. Keep proof hash pinned to `25ced...` in all regenerated metrics/artifacts.
4. Move to Phase 3 timing grid only after full Phase 2 comparison table is complete.
5. Keep commits split between:
   - code/docs/tests
   - generated artifacts (models/metrics/data).

---

## 9) Quick command references

### Re-run Modal verifier (now-correct path)

```powershell
.\scripts\run_vertex_equivalence_verification.ps1
```

### Run full phase 2-end manifest commands

```powershell
.venv\Scripts\python.exe scripts/run_neuroemo_experiment_matrix.py --execute --skip-existing
```

### Run local phase-2 classical subset launcher

```powershell
.\scripts\run_surface_phase2_matrix.ps1
```

---

## 10) Bottom line

Today’s work successfully moved the project from "mesh identity only" to **full projection equivalence proof** and propagated that proof through training contracts. The first executed Phase-2 classical subset does not beat the projected postfix anchor on top-line accuracy/F1, so the next leverage remains broader matrix completion + timing/feature/structured-decoding experiments rather than immediate default promotion.
