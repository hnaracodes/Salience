# Ledger — CS-20260716-DAILY-HANDOFF

Daily progress + seamless tomorrow protocol for a fresh agent.

**Working directory for all commands:**  
`C:\Users\kragh\OneDrive\Documents\HrudayCodingProjects\TribeV2WebAnalyzer\TribeV2`

**Preferred Python:** `.venv311\Scripts\python.exe` when present (Horikawa); otherwise `py` / system Python 3.14 for website capture tools.

---

## Step 0 — Context loaded (2026-07-16)

### Track A: Product claims / T2–T3 (this agent)

Source plan: `C:\Users\kragh\.cursor\plans\earn_t2_t3_claims_b00893cc.plan.md`  
Parent session: `sessions/CS-20260611-PRODUCT-CLAIMS/`

### Track B: Horikawa reframing (prior agent)

Source transcript: [7ad97d18-93d3-496e-a82c-5106220a359e](7ad97d18-93d3-496e-a82c-5106220a359e)  
Parent session: `sessions/CS-20260620-EMOTION-ACCURACY-ADR7/`  
Also see: `AGENT-HANDOFF-EMOTION-PIPELINE.md` (may be stale relative to today’s NPZ restore)

---

## Step 1 — What was completed today

### 1A. Claims governance (DONE)

- Downgraded premature T2/T3 language in `docs/validation/CLAIMS.md`
  - `comparison_score` → **T1** until N≥15 norms + memo
  - `conversion_prediction` → **Unearned T3**
  - Engagement/activation → **T1** (not T1–T2)
- Wrote `docs/validation/HUMAN_INTERVENTION_CLARITY.md`
- Updated `docs/validation/attention-calibration-protocol.md` (property splits, fit-then-holdout, no Clarity leakage)

**Clarity facts (verified):**
- Client SDK is open source MIT (`microsoft/clarity`)
- Hosted click/heatmap data is **private** to project owners
- Public sites (Stripe, Wikipedia, etc.) **cannot** supply Clarity CSVs for T2
- Heatmap CSV = Clarity UI download; Data Export API = dashboard aggregates only

### 1B. Attention calibration harness (DONE — code)

Files:
- `scout_core/attention_calibration.py`
- `scripts/validate_attention_proxy.py`
- `tests/test_attention_calibration.py`
- `scout_data/validation/attention_v1/manifest.json` (schema v2, empty sessions)

Behavior now:
- NaN Spearman **fails** (not passes)
- Average ranks for ties
- Rejects bundles with `clarity_attribution` (label leakage)
- Fit weights on train/validation only; `--evaluate-holdout` once
- Weight search over raw `attention_score` + `clickability` (not fused `combined_score`)
- Bootstrap CI on page-macro metrics
- `--write-config` refused unless holdout gate passes

Tests: `py -m pytest tests/test_attention_calibration.py` → **pass** (after `bool()` fix)

Empty corpus validator correctly reports `n_sessions: 0`, `passed: false`.

### 1C. Explore / capture bugs fixed (DONE — code)

Critical bug: `ExploreBudget.at_limit()` treated `max_clicks=0` and `max_pages=1` as “session done,” producing **0 snapshots**.

Fix in `scout_core/explore_policy.py`:
- `at_limit()` → only `t_idx >= max_tr`
- Added `clicks_exhausted()` / `pages_exhausted()` for action gating
- Block nav to new pages when page budget exhausted; still allow scroll capture

Other fixes:
- Explore `page.goto` uses `load` then `domcontentloaded` (not `networkidle`) in `walkthrough.py`
- Windows cp1252 crash: replaced `→` with `->` in `run_website_session.py`
- `run_website_session.py` forwards `--url` to capture stage
- Installed Playwright Chromium via `py -m playwright install chromium`

Tests: `tests/test_explore_policy.py` includes `test_zero_click_budget_still_allows_scroll_capture` — **pass**

### 1D. Preliminary website corpus (PARTIAL)

Configs/scripts:
- `configs/validation_site_corpus.yaml` — 8 public archetypes + 2 local fixtures
- `configs/explore_public_smoke.yaml` — zero-click public smoke
- `configs/walkthrough_scripts/explore_threadmind_smoke.yaml`
- `configs/walkthrough_scripts/explore_aurora_smoke.yaml`
- `scripts/run_validation_site_corpus.py` — `--local-only`, run log, fail on gate failure
- `tests/test_validation_site_corpus.py`

**Local SSRF note:** `--url http://127.0.0.1/...` is rejected by design. Local sites must use baked-in `script:` YAML (no `--url`).

**Successful local captures (exit 0, promoted):**

| Session ID | Site | Snapshots |
|------------|------|-----------|
| `validation_threadmind_20260716T125318Z_r1` | threadmind | 23 |
| `validation_threadmind_20260716T125318Z_r2` | threadmind | 16 |
| `validation_aurora_20260716T125318Z_r1` | aurora | 16 |
| `validation_aurora_20260716T125318Z_r2` | aurora | 77 |

Batch command completed with `exit_code: 0` (~8h wall; video pad inflates snapshot counts).

**Also promoted (public, capture-only):** github_features, shopify, linear, govuk (see `scout_data/validation/preliminary_web_corpus/runs.jsonl`).

**NONE of the validation_* sessions have `preds.npz` or `analysis_bundle.json` yet.** Capture-only.

### 1E. Norm tooling (DONE — code, not rebuilt)

- `build_naturalistic_norms.py`: default `--min-clips 15`, hard fail if below floor
- `compute_norms.py`: `--exclude` session ids
- Pipeline still defaults to `synthetic_bootstrap_v1` (do **not** switch until N≥15)

### 1F. Horikawa / emotion track (prior agent — DONE / PARTIAL)

Completed in [7ad97d18](7ad97d18-93d3-496e-a82c-5106220a359e):

| Item | Status | Evidence |
|------|--------|----------|
| Numeric per-target ordering (`class_2` before `class_10`) | DONE | `scout_core/horikawaCode/cv.py` |
| Bootstrap CI written into eval reports | DONE | `evaluate_horikawa_decoder.py` |
| Real subcortical map fail-closed | DONE | `scout_core/subcortical/atlas.py` |
| Feature-spec honored in prepare | DONE | `prepare_horikawa_tribev2.py` |
| Ablation subset CLI / frozen runner | DONE | `run_ablation_matrix.py`, `run_ablation_subset_frozen.py` |
| Demo test no longer overwrites real `train.npz` | DONE | `tests/horikawaCode/test_horikawa_training.py` |
| Phase 0 consistency | DONE | `reports/phase0_consistency.json` → `consistent: true`, **371** samples, corpus **`ready_all`**, `cv_mean_r ≈ 0.242` |
| Product eval report | DONE | `reports/eval_ablation_v1.json` (matches meta) |
| Dims eval report | DONE | `reports/eval_ablation_dims_v1.json` |
| Shadow inference smoke | DONE | `reports/shadow_inference_smoke.json` → `inference_ok: true` |
| Frozen ablation subset report | **MISSING** | no `*ablation*subset*` under reports/ |
| `reframing_findings.md` | **MISSING** | `write_reframing_findings.py` exists; output not written |
| Full 28-cell ablation matrix | NOT RUN | deferred; use subset first |

Current model meta (`scout_models/horikawa_ridge_v1/meta.json`):
- `corpus: ready_all`
- `n_samples: 371`
- `cv_mean_r: 0.24197771308531232`
- `feature_spec: fused_schaefer400_subcortical_v1`

---

## Step 2 — Explicitly NOT done (do not claim)

1. Attention T2 — no Clarity CSVs; manifest sessions empty
2. Cross-session T2 — naturalistic norms still N≪15; no `preds.npz` on new captures
3. Conversion T3 — no labels / model
4. Horikawa Modal go/no-go memo (`reframing_findings.md`)
5. Frozen ablation subset JSON not on disk
6. Pipeline `DEFAULT_NORM_ID` still synthetic (correct until norms ready)

---

## Step 3 — Known pitfalls (read before coding)

1. **Leaked `chrome-headless-shell` processes** hang Playwright. Kill before capture batches:  
   `Get-Process chrome-headless-shell -ErrorAction SilentlyContinue | Stop-Process -Force`
2. **`pad_manifest_snapshots_to_video_duration`** inflates `dom_snapshots` beyond explore `max_tr`. Promotion gates should key off `exploration_log` length or min snapshots, not treat padded count as explore TR count.
3. **Localhost capture:** never pass `--url 127.0.0.1` (SSRF). Use `script:` with baked `initial_url`.
4. **Public sites:** Wikipedia 403; Stripe/SPAs may timeout; Clarity unavailable. Public captures ≠ attention ground truth.
5. **Horikawa:** never run `prepare_horikawa_tribev2.py --demo` without a temp config — it previously overwrote real `train.npz`. Tests fixed; still be careful.
6. Use **`.venv311`** for Horikawa training/eval. Website capture used system `py` + Playwright.
7. Set `$env:PYTHONUNBUFFERED='1'; $env:PYTHONIOENCODING='utf-8'` on Windows.

---

## Tomorrow — exact execution protocol

Copy this block into the new agent’s first message or follow top-to-bottom.

### Morning bootstrap (5 min)

```powershell
cd C:\Users\kragh\OneDrive\Documents\HrudayCodingProjects\TribeV2WebAnalyzer\TribeV2
Get-Process chrome-headless-shell -ErrorAction SilentlyContinue | Stop-Process -Force
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
```

Read in order:
1. `sessions/CS-20260716-DAILY-HANDOFF/session.md`
2. **This file** (`ledger.md`)
3. `docs/validation/CLAIMS.md`
4. `C:\Users\kragh\.cursor\plans\earn_t2_t3_claims_b00893cc.plan.md`
5. `sessions/CS-20260620-EMOTION-ACCURACY-ADR7/session.md`

Verify artifacts still intact:

```powershell
.\.venv311\Scripts\python.exe -c "import json,numpy as np; from pathlib import Path; m=json.loads(Path('scout_models/horikawa_ridge_v1/meta.json').read_text()); d=np.load('scout_data/horikawaCode/tribev2_fused/train.npz', allow_pickle=True); print('model', m['cv_mean_r'], m['n_samples'], m['corpus']); print('npz', d['X'].shape, str(d['corpus']))"
py -c "import json; from pathlib import Path; print('phase0', json.loads(Path('scout_data/horikawaCode/reports/phase0_consistency.json').read_text())['consistent']); print('shadow', json.loads(Path('scout_data/horikawaCode/reports/shadow_inference_smoke.json').read_text())['inference_ok'])"
```

Expect: model/npz ~371 `ready_all`, phase0 `True`, shadow `True`.

---

### Priority 1 — Finish Horikawa day (unblock Modal go/no-go) [~1–3 h]

**P1.1** Run frozen ablation subset (if report missing):

```powershell
.\.venv311\Scripts\python.exe scripts\horikawaCode\run_ablation_subset_frozen.py --n-permutations 25 --n-bootstrap 50
```

Default write path: `scout_data/horikawaCode/reports/ablation_matrix.json`  
(`write_reframing_findings.py` reads that exact filename.)

**P1.2** Write findings memo:

```powershell
.\.venv311\Scripts\python.exe scripts\horikawaCode\write_reframing_findings.py
```

Reads: `phase0_consistency.json`, `alignment_report.json` (optional), `ablation_matrix.json`  
Writes: `scout_data/horikawaCode/reports/reframing_findings.md`

**P1.3** Update `sessions/CS-20260620-EMOTION-ACCURACY-ADR7/ledger.md` with:
- Phase 0 result (371 / ready_all / r≈0.242)
- Ablation subset summary + go/no-go for Modal
- Link to findings memo

**P1.4** Only if findings recommend it: do **not** start a full 28-cell / 100-perm matrix during Modal GPU jobs; schedule offline.

---

### Priority 2 — Promote capture corpus to TRIBE preds [~2–4 h, Modal]

Pick **promoted** sessions with good manifests (start with local fixtures):

```
validation_threadmind_20260716T125318Z_r1
validation_threadmind_20260716T125318Z_r2
validation_aurora_20260716T125318Z_r1
validation_aurora_20260716T125318Z_r2
```

Plus any public promoted IDs from `scout_data/validation/preliminary_web_corpus/runs.jsonl` where `"promoted": true`.

For each `SESSION_ID`:

```powershell
py scripts\run_website_session.py --session-id SESSION_ID --stage tribe
py scripts\run_website_session.py --session-id SESSION_ID --stage dual_track
py scripts\run_website_session.py --session-id SESSION_ID --stage heatmaps --no-modal
# or --modal / production heatmaps when GPU budget allows
py scripts\run_website_session.py --session-id SESSION_ID --stage analyze --norm-id synthetic_bootstrap_v1 --website --with-heatmaps --skip-narrative
```

Or batch via:

```powershell
py scripts\run_validation_site_corpus.py --stage all --local-only --repeats 1
```

**Done when:** ≥4 sessions have `preds.npz` + `analysis_bundle.json`. Target toward ≥15 for norms (may need more captures).

**Do not** switch `DEFAULT_NORM_ID` yet.

---

### Priority 3 — Expand capture corpus if under 15 preds [~2–6 h]

```powershell
# Terminal A
py -m http.server 8780 --directory chatbot_product_site
# Terminal B
py -m http.server 8765 --directory tests/fixtures/walkthrough_site
# Terminal C
Get-Process chrome-headless-shell -ErrorAction SilentlyContinue | Stop-Process -Force
py scripts\run_validation_site_corpus.py --stage capture --local-only --repeats 3
```

Then run tribe/analyze on new IDs (Priority 2).

Optional public retries (engineering only): `--site govuk --site mozilla_firefox --site github_features` (not for Clarity T2).

---

### Priority 4 — Rebuild naturalistic norms (only if ≥15 preds)

```powershell
# Hold out at least one entire site (e.g. both aurora sessions)
py scripts\build_naturalistic_norms.py --norm-id naturalistic_v1 --min-clips 15 --exclude validation_aurora_20260716T125318Z_r1 validation_aurora_20260716T125318Z_r2
```

Verify `scout_norms/naturalistic_v1/meta.json` → `n_engagement_sessions >= 15`.

Populate `scout_data/validation/engagement_v1/manifest.json` with corpus vs holdout lists.

Re-analyze holdouts with `--norm-id naturalistic_v1` and confirm bundle provenance:
- `comparison_mode: norm_referenced`
- **no** `norm_fallback_reason`

Only then consider changing `services/pipeline/runner.py` / `jobs.py` `DEFAULT_NORM_ID`.

---

### Priority 5 — Attention T2 (BLOCKED on human)

**Interrupt user** if Clarity data still missing. Required inputs (see `docs/validation/HUMAN_INTERVENTION_CLARITY.md`):

1. ≥4 owned properties with Clarity installed  
2. ≥12 pages, each ≥200 unique sessions + ≥50 mapped clicks  
3. CSVs under `scout_data/validation/attention_v1/exports/`  
4. Matching TRIBE sessions analyzed **without** `--clarity-csv` on claim bundles  

Then:

```powershell
# Edit scout_data/validation/attention_v1/manifest.json with property_id, role train|validation|holdout, clarity_csv paths
py scripts\validate_attention_proxy.py --fit-weights --evaluate-holdout --write-config configs\attribution_calibrated.yaml --write-memo docs\validation\attention_validation_memo.md
```

Promote CLAIMS.md attention row to T2 **only** if memo gate passes.

---

### Priority 6 — Conversion T3 (deferred)

Do not invent CTA-click → conversion labels. Wait for real signup/purchase/form outcomes (≥140 variants). See plan Phase 3.

---

### Priority 7 — Explore E2E test + claims sync

```powershell
# After Threadmind explore works reliably:
# add tests/test_explore_capture.py asserting capture_mode, pages_visited>=2 OR exploration_log non-empty under a click-enabled profile
py -m pytest tests\test_explore_policy.py tests\test_attention_calibration.py tests\test_validation_site_corpus.py tests\horikawaCode\test_horikawa_training.py -q
```

Update `docs/validation/CLAIMS.md` only for tiers that passed gates. Append Step entry to this ledger + parent sessions.

---

## Suggested tomorrow order (if time-boxed)

| Block | Work | Outcome |
|-------|------|---------|
| 0–30m | Bootstrap + verify Horikawa artifacts | Safe to proceed |
| 30m–2h | P1 ablation subset + findings | Modal go/no-go written |
| 2–5h | P2 tribe+analyze on 4 local sessions | First preds for norms |
| 5h+ | P3 more captures OR wait on Clarity human | Path to N≥15 / T2 |

If only one block: **do Priority 1 first** (closes the Horikawa transcript cleanly), then Priority 2.

---

## Human intervention checklist (ask user)

- [ ] Owned domains + Clarity project ready?  
- [ ] Heatmap CSVs exported for ≥12 pages?  
- [ ] Modal GPU budget OK for tribe on N validation sessions?  
- [ ] Conversion outcome labels available (yes/no/later)?  

Do **not** ask user to paste Clarity API tokens in chat; use env vars / local files.

---

## File checklist for tomorrow agent

| Path | Role |
|------|------|
| `sessions/CS-20260716-DAILY-HANDOFF/ledger.md` | This protocol |
| `docs/validation/CLAIMS.md` | Claim language truth |
| `docs/validation/HUMAN_INTERVENTION_CLARITY.md` | Clarity collection |
| `configs/validation_site_corpus.yaml` | Site list |
| `scout_data/validation/preliminary_web_corpus/runs.jsonl` | Capture outcomes |
| `scout_data/horikawaCode/reports/phase0_consistency.json` | Horikawa gate |
| `scout_models/horikawa_ridge_v1/meta.json` | Current decoder |
| `scripts/horikawaCode/run_ablation_subset_frozen.py` | Next Horikawa command |
| `scripts/horikawaCode/write_reframing_findings.py` | Findings writer |
| `C:\Users\kragh\.cursor\plans\earn_t2_t3_claims_b00893cc.plan.md` | T2/T3 plan |

---

## Signature

- Signature: `cursor-grok-4.5@CS-20260716-DAILY-HANDOFF`
- Date: 2026-07-16
- Consolidates: this chat (T2/T3 + corpus) + transcript `7ad97d18-93d3-496e-a82c-5106220a359e` (Horikawa)
