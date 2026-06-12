---
id: ISSUE-SEC-001
status: resolved
severity: high
category: security
title: UX viewer stored XSS via innerHTML
---

# ISSUE-SEC-001 — Viewer XSS

## Problem

`ux_session_viewer.html` interpolated captured DOM ids/text and LLM narrative fields into `innerHTML` without escaping. Malicious page content or a crafted `viewer_bundle.json` could execute script in the analyst browser.

## Fix

Added `escapeHtml()` and applied to element detail, grounding panel, executive summary, sidebar lists, and section tips. Attribute values use escaped `data-dom-id`.

## Residual risk

Low for trusted localhost fixture bundles. Re-audit if adding new `innerHTML` sinks.
