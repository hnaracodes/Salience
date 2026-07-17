---
id: ISSUE-001
session_id: CS-20260618-DEMOGRAPHIC-MUX-PLAN
status: open
owner: agent
files:
  - data_prep/metadata_harmonize.py
  - configs/clusters.yaml
  - docs/implementation-plans/demographic-multiplexer-implementation-plan.md
---

# Demographic Data Availability and Site Confound

## Problem

The demographic multiplexer requires cluster-average fMRI targets from naturalistic video-watching datasets with demographic metadata. **No available public dataset satisfies the v1 plan's marketed cluster grid** (genz/millennial/genx/boomer × female/male × ethnicity at N≥20 per cell).

Concrete blockers:

1. **Insufficient N per cell:** CNeuroMod has 6 subjects total. StudyForrest 7T has 20. NNDb has ~8 subjects per movie. Only HCP 7T (184 subjects, age 22-36) and Cam-CAN (~650, single 8-min clip) offer partial coverage.
2. **Age range gap:** HCP 7T is young-adult only (22-36). Cam-CAN spans 18-88 but uses one short clip. No dataset provides genz/millennial/genx/boomer cells at scale.
3. **Site/scanner confound:** HCP, Cam-CAN, CNeuroMod, NNDb use different scanners and protocols. Age and sex correlate with dataset identity. Demographic effects may be indistinguishable from site effects without careful LODO validation.
4. **Ethnicity unavailable:** No movie-fMRI dataset provides ethnicity at scale with matched stimuli and consent for demographic modeling.
5. **Access gates:** CNeuroMod requires registered access + DTA. HCP requires ConnectomeDB credentials.
6. **Domain mismatch:** All training data is Hollywood movies / short clips. Salience inference runs on UI screen-capture walkthroughs — severe OOD even if demographic signal exists in movies.

## Impact

- Cannot train or validate the multiplexer as originally specified without rescoping clusters and accepting site confound.
- Product claims about "how Gen Z vs. Boomers respond to this landing page" are **not scientifically supportable** with current data.
- M0 viability experiment may fail, triggering feature kill switch.

## Proposed Resolution

1. **Rescope clusters** to coarse age bands (young/middle/older adult) with sex where powered; pool sex when N insufficient. Ethnicity removed permanently unless new consented dataset appears.
2. **Run M0** variance partition on HCP 7T + Cam-CAN before any training infra investment.
3. **Mandate LODO** (leave-one-dataset-out) in all eval to quantify site confound.
4. **Suppress underpowered clusters** in UI; never show clusters with N < 20 or failed eval gate.
5. **Product disclaimer:** All cluster outputs labeled "model-relative prototype; not measured demographic emotion."
6. **Long-term:** Collect proprietary paired UI-walkthrough + demographic survey data, or partner for web-UX fMRI study (not feasible near-term).

## Validation

- M0 report documents whether demographic variance exceeds site variance and permutation null.
- If M0 fails: close FEATURE-001, resolve this issue as `cancelled` with rationale logged.
