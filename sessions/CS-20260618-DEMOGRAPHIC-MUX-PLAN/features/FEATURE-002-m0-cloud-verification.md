---
id: FEATURE-002
title: M0 cloud verification (preflight + pilot)
status: in_progress
date: 2026-06-22
---

# M0 Cloud Verification

## What ran

1. **Preflight** (`run_m0_hcp_cloud.py --preflight`)
   - S3 bucket `hcp-openaccess`, prefix `HCP_1200`
   - 184 subjects with 7T movie data
   - Demographics from bundled `hcp1200_age_gender.csv`
   - Movie runs 1–2 available per probe subject (runs 3–4 absent on probes — acceptable)

2. **Pilot pipeline** (`--phase hcp --subjects 5`)
   - Real CIFTI dtseries streamed from S3 on Modal
   - 5 subjects × 7 Yeo-7 network features extracted
   - M0 variance partition + 500 permutations
   - Artifacts: `scout_data/demographic/viability/subjects.csv`, `m0_report.json`

3. **Multi-site smoke test** (`--phase hcp_camcan --subjects 5`, Step 9)
   - 5 HCP + 5 Cam-CAN → **10 subjects, 2 sites, multi_site_validated: true**
   - Cam-CAN ingest via OpenNeuro rest BOLD (colon-path URLs + MNI resample)
   - `passed: false` at n=10 (underpowered — expected)
   - Artifacts: `m0_report_smoke5.json`, `subjects_smoke5.csv`

## Outcome

| Check | Result |
|-------|--------|
| S3 + Modal infra | Pass |
| HCP feature extraction | Pass |
| Cam-CAN feature extraction | Pass (rest fMRI, OpenNeuro ds000221) |
| Multi-site merge (smoke) | Pass (n=10, 2 sites) |
| Production gate (`m0_passed_for_training`) | **Not yet** — 50-subject run interrupted |

## Bugs fixed during verification

- Windows: `modal` not on PATH → wrapper uses `.venv311/Scripts/modal.exe`
- Single-site sklearn crash → `_one_hot` returns intercept when drop-first yields 0 columns
- Cam-CAN HTML scrape → colon-path rest BOLD discovery
- Cam-CAN NaN features → MNI152 resample before `vol_to_surf`
- HCP rows dropped on merge → Yeo-7-only `net_*` columns + Yeo-7-only `dropna`
- CLI skipped download on gate fail → download before exit check

## Next step (resume)

See ledger **Step 9 — How to continue**. Run:

```powershell
python scripts/run_m0_hcp_cloud.py --phase hcp_camcan --subjects 50 --n-permutations 1000 --download
```

Then confirm `m0_passed_for_training()` → `true` before M1.
