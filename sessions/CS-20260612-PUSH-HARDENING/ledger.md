# Ledger — CS-20260612-PUSH-HARDENING

## Audit

- Full `git diff` review via code-auditor agent (29 modified + ~15 new source files).
- Verdict before fixes: **Needs revision** (2 High).
- Verdict after fixes: **Approve with caveats** (explore selector hardening deferred).

## Fixes applied

| Issue | File(s) | Change |
|-------|---------|--------|
| ISSUE-SEC-002 | `scout_core/marketing_scores.py` | Honest `comparison_mode` + `norm_fallback_reason` when population n&lt;2 |
| ISSUE-SEC-001 | `viewer/ux_session_viewer.html` | `escapeHtml()` on DOM/LLM text sinks |
| ISSUE-SEC-003 | `viewer/ux_session_viewer.html` | `resolveBase()` blocks `..` and absolute URLs |
| ISSUE-SEC-005 | `scout_norms/*/meta.json` | Repo-relative `parcellation_csv` paths |
| ISSUE-SEC-004 | `scout_core/explore_policy.py` | Subdomain check no longer allows parent domain |

## Validation

```powershell
pytest tests/test_explore_policy.py tests/test_marketing_scores.py tests/test_attention_attribution.py -q
```

## Commit

Branch: `improved` (from `main`).
