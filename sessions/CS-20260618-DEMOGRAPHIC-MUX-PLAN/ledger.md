# Ledger

Append exact changes as the session progresses. Include file paths, intent, effects, commands, and validation.

## Step 1 - Context

- Files read:
  - `docs/implementation-plans/demographic-multiplexer-implementation-plan.md` (v1 plan)
  - `README.md` (current Salience app state)
  - `tribe.py` (existing TRIBE Modal integration — `predict(events=df)` only, no hidden state)
  - `.cursor/agents/ml-training-specialist.md`
  - `.cursor/rules/ledger-protocol.mdc`
  - `sessions/INDEX.md`, `sessions/PROTOCOL.md`, `sessions/_TEMPLATE/*`
- Prior sessions consulted:
  - `CS-20260602-EEV-PIPELINE-PILOT` (LOVO eval, FeatureContract patterns)
  - `CS-20260528-WEBSITE-CONSOLIDATED` (NeuroEmo archive, atlas fix)
  - `CS-20260522-NEUROEMO-DATASET-PREP` (subject-held-out conventions)
- Open issues considered: none pre-existing for demographic mux; created ISSUE-001 for data availability blocker.
- External research: TRIBE v2 paper (arxiv 2605.04326), dataset survey (HCP 7T, CNeuroMod, NNDb, Cam-CAN, Forrest, Narratives).

## Step 2 - Plan Rework

- Files touched:
  - `docs/implementation-plans/demographic-multiplexer-implementation-plan.md`
- What changed:
  - Rewrote Intent and production path (1152-dim hidden state, low-rank delta readout, parcel/network primary target).
  - Added Scientific Preconditions table and no-hidden-state fallback.
  - Replaced MLP architecture with low-rank cluster-conditional delta on population readout; legacy MLP demoted to smoke-test only.
  - Added Loss and Metrics section (correlation loss, noise-ceiling, eval gate criteria).
  - Rewrote Data Extraction Plan with honest dataset inventory table; cut ethnicity; rescaled age bands; mandated LODO.
  - Added M0 viability experiment section (variance partition + permutation null before Modal infra).
  - Reordered milestones: M0 gates M1-M5; M3 eval gate with kill switch.
  - Expanded Risks (domain shift movie→UI, site confound, small-N, accuracy stacking).
  - Updated Definition of Done to require M0 + M3 eval gate pass and feature flag default off.
  - Updated directory structure (viability scripts, parcel-space centroids).
  - Updated `train_clusters.py` skeleton references (hidden_dim=1152, correlation loss).
- Expected effect: Future implementers have scientifically honest, gated plan; avoids building expensive infra on unvalidated hypothesis.

## Step 3 - Ledger Session

- Files touched:
  - `sessions/CS-20260618-DEMOGRAPHIC-MUX-PLAN/session.md`
  - `sessions/CS-20260618-DEMOGRAPHIC-MUX-PLAN/ledger.md`
  - `sessions/CS-20260618-DEMOGRAPHIC-MUX-PLAN/features/FEATURE-001-demographic-multiplexer.md`
  - `sessions/CS-20260618-DEMOGRAPHIC-MUX-PLAN/issues/ISSUE-001-demographic-data-availability.md`
  - `sessions/INDEX.md`
- What changed: Created ledger session folder, feature file, open issue for data availability, index row.
- Expected effect: Demographic multiplexer feature is queryable via Ledger Protocol; M0 gate is documented as next step.

## Step 4 - M0-M5 Implementation (2026-06-20)

- Files touched:
  - M0: `configs/cluster_schema.yaml`, `configs/clusters.yaml`, `data_prep/schemas.py`, `data_prep/viability_partition.py`, `scripts/run_demographic_viability.py`, `tests/test_demographic_viability.py`
  - M1: `salience/mux/tribe_mux.py`, `losses.py`, `checkpoints.py`, `latent_extractor.py`, `inference.py`, `tribe_commit.txt`, `tests/test_tribe_mux.py`
  - M2: `data_prep/metadata_harmonize.py`, `segment_clusters.py`, `align_timeseries.py`, `aggregate_centroids.py`, `package_webdataset.py`, `scripts/run_demographic_data_prep.py`, `tests/test_demographic_data_prep.py`
  - M3: `salience/mux/eval_metrics.py`, `scripts/train_demographic_mux_local.py`, `salience/modal_app/train_clusters.py`
  - M4: `salience/modal_app/inference_mux.py`, `stream_protocol.py`, `image_defs.py`, `scripts/run_demographic_mux_session.py`, `services/pipeline/runner.py` (DEMOGRAPHIC_MUX stage)
  - M5: `salience/mux/neural_barrier.py`, `tests/test_neural_barrier.py`
  - `requirements.txt` (torch optional marker)
- What changed: Full phase-by-phase implementation per revised plan; M0 audit fixes applied (go criterion, is_synthetic flag, subject dedup, CLI exit code).
- Expected effect: Codebase supports gated demographic mux pipeline; production blocked until real-data M0 pass and trained checkpoint.
- Commands: `pytest tests/test_demographic_viability.py tests/test_demographic_data_prep.py tests/test_neural_barrier.py tests/test_tribe_mux.py -v` → 17 passed.
- M0 audit follow-ups applied from code-auditor (criterion_beats_site, is_synthetic, dedup, validation).

## Step 5 - Cloud M0 via Modal + HCP S3 (2026-06-22)

- Files touched:
  - `configs/hcp_7t_movie_manifest.yaml`, `configs/hcp_fslr_schaefer400_yeo7.npz`
  - `scripts/build_hcp_parcellation_artifact.py`, `scripts/run_m0_hcp_cloud.py`
  - `data_prep/hcp_s3.py`, `hcp_cifti_network.py`, `hcp_demographics.py`, `m0_cloud_camcan.py`
  - `salience/modal_app/m0_hcp_cloud.py`, `image_defs.py` (`m0_image`)
  - `data_prep/schemas.py`, `viability_partition.py` (multi-site gate hardening)
  - `tests/test_hcp_cifti_network.py`, `tests/test_hcp_s3_paths.py`, extended `test_demographic_viability.py`
- What changed: Diskless M0 pipeline streams HCP 7T Movie FIX dtseries from `s3://hcp-openaccess` on Modal, extracts 7 net_* scalars per subject, optional Cam-CAN merge via OpenNeuro HTTPS. HCP-only phase is informational (not production gate); `m0_passed_for_training` requires multi-site.
- User action: BALSA AWS credentials → `modal secret create hcp-aws-secret`; run `python scripts/run_m0_hcp_cloud.py --preflight`.
- Validation: `pytest tests/test_hcp_cifti_network.py tests/test_hcp_s3_paths.py tests/test_demographic_viability.py` (7 passed).

## Step 6 - M0 verification phase (2026-06-22)

- Files touched:
  - `scripts/run_m0_hcp_cloud.py` — resolve Modal CLI via `.venv311/Scripts/modal.exe` on Windows
  - `data_prep/viability_partition.py` — single-site HCP pilot: empty site one-hot → intercept column
  - `tests/test_demographic_viability.py` — `test_single_site_hcp_pilot_does_not_crash`
- Commands:
  - `python scripts/run_m0_hcp_cloud.py --preflight` → **ok: true** (184 7T subjects, S3 prefix `HCP_1200`)
  - `python scripts/run_m0_hcp_cloud.py --phase hcp --subjects 5 --n-permutations 500 --download`
- Results:
  - Pilot extracted 5 real HCP 7T movie subjects from S3; `scout_data/demographic/viability/m0_report.json` downloaded
  - `passed: false` (expected — HCP-only single-site probe, n=5, not production gate)
  - `go_criteria.variance_in_two_priority_networks: true` but `permutation_significant_in_two_networks: false` (underpowered)
  - Infrastructure verified: S3 streaming, CIFTI parcellation, demographics merge, M0 report write
- Next: `--phase hcp_camcan` with larger N for production `m0_passed_for_training` gate

## Step 7 - Production gate attempt (50 subjects, 2026-06-22)

- Command: `python scripts/run_m0_hcp_cloud.py --phase hcp_camcan --subjects 50 --n-permutations 1000 --download`
- HCP: 50/50 subjects extracted from S3
- Cam-CAN: **0 subjects** — `list_camcan_movie_subjects` finds 0 on OpenNeuro HTML scrape (316 demo rows loaded)
- Report: `scout_data/demographic/viability/m0_report.json`
- `passed: true` (all 3 go criteria met on HCP-only n=50)
- `multi_site_validated: false`, `n_sites: 1`
- `m0_passed_for_training()`: **false** (production gate blocked)
- Interpretation: HCP-only stats look strong (Vis/Default/SalVentAttn p<0.01) but site confound untestable; Cam-CAN ingest must be fixed before real gate

## Step 8 - Verification + Cam-CAN ingest fix (2026-06-22)

- Confirmed prior agent transcript (`71a20e5e`): **5 HCP pilot** succeeded; **production gate did NOT pass**
- `m0_passed_for_training()` → **false** on `m0_report.json` (`n_sites=1`, `multi_site_validated=false`)
- Root cause: OpenNeuro ds000221 mirror has **rest fMRI only** (no movie BOLD); HTML directory scrape returned 0 subjects
- Fix: `m0_cloud_camcan.py` — colon-path OpenNeuro URLs + rest task discovery + MNI152 resample before `vol_to_surf`
- Local verify: 5/5 Cam-CAN rest subjects discoverable; `sub-010001` yields finite `net_*` features
- Caveat: Cam-CAN uses rest vs HCP 7T movie (task mismatch); note added to pipeline report
- Next: re-run `--phase hcp_camcan --subjects 50` and confirm `n_sites >= 2` + `m0_passed_for_training()`

## Step 9 - Multi-site smoke test + production gate (PAUSED, 2026-06-22)

**Status: PAUSED** — user stopped before 50-subject production gate completed.

### Files touched this step

- `data_prep/m0_cloud_camcan.py` — colon-path rest BOLD URLs; MNI152 resample; Yeo-7-only `net_*` columns (fixes HCP row drop)
- `salience/modal_app/m0_hcp_cloud.py` — Yeo-7 `dropna`; 8 GB / 6 h timeout; no `SystemExit(1)` on gate fail; Cam-CAN rest note
- `scripts/run_m0_hcp_cloud.py` — download artifacts even when gate fails; `--force` on volume get

### Completed

1. **Smoke test** (`--phase hcp_camcan --subjects 5 --n-permutations 500`)
   - Modal: **10 subjects** (5 HCP + 5 Cam-CAN), **`n_sites: 2`**, **`multi_site_validated: true`**
   - `passed: false` (expected at n=10 — underpowered permutations)
   - `m0_passed_for_training()`: **false** (needs `passed: true` at scale)
   - Artifacts: `scout_data/demographic/viability/m0_report_smoke5.json`, `subjects_smoke5.csv`
   - Modal run: https://modal.com/apps/hrudayiitb/main/ap-ITh3pUHeVbevjLGTnOSVOI

2. **Bug fixed:** Cam-CAN extra `net_net_*` columns caused HCP rows to be dropped by `dropna` → single-site false positive on first smoke attempt.

### Not completed

- **50-subject production gate** — started then **interrupted by user** (~22 min in); no fresh local `m0_report.json` from that run.
- Do **not** treat prior Step 7 HCP-only report (`n_sites=1`, `passed=true`) as production gate pass.

### How to continue (resume protocol)

```powershell
cd TribeV2
.\.venv311\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"

# 1. Production gate (expect ~1–3 h on Modal: 50 HCP S3 + 50 Cam-CAN rest downloads)
python scripts/run_m0_hcp_cloud.py --phase hcp_camcan --subjects 50 --n-permutations 1000 --download

# 2. Verify gate
python -c "
from pathlib import Path
import pandas as pd
from data_prep.viability_partition import m0_passed_for_training, load_m0_report
p = Path('scout_data/demographic/viability/m0_report.json')
df = pd.read_csv('scout_data/demographic/viability/subjects.csv')
print('subjects', len(df), df['dataset'].value_counts().to_dict())
r = load_m0_report(p)
print('n_sites', r.n_sites, 'multi_site', r.multi_site_validated, 'passed', r.passed)
print('m0_passed_for_training', m0_passed_for_training(p))
"
```

**Success criteria for unblocking M1+ training:**

| Check | Required |
|-------|----------|
| `subjects.csv` | ≥50 HCP + ≥50 Cam-CAN rows |
| `n_sites` | ≥ 2 |
| `multi_site_validated` | `true` |
| `passed` | `true` (all 3 go criteria) |
| `m0_passed_for_training()` | **`true`** |

**If gate fails at n=50:** inspect `m0_report.json` go_criteria; check Modal volume `viability/skipped_subjects.json`; re-run smoke (`--subjects 5`) to confirm infra before scaling.

**Scientific caveats (unchanged):**

- Cam-CAN on OpenNeuro = **rest** fMRI; HCP = **7T movie** (task mismatch noted in report)
- Parcellation npz is test fixture, not production Schaefer fsLR
- Only proceed to M1 (TRIBE hidden-state spike) after `m0_passed_for_training()` is true

## Validation

- Commands: pytest demographic + HCP cloud suite + single-site fix
- Results: 17+ unit tests green; cloud M0 preflight + 5-subject pilot succeeded on Modal
