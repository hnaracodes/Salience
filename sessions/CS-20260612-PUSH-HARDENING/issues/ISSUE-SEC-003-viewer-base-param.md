---
id: ISSUE-SEC-003
status: resolved
severity: medium
category: security
title: Viewer base query param open redirect / arbitrary fetch
---

# ISSUE-SEC-003 — Viewer `?base=` parameter

## Problem

Unvalidated `?base=` URL parameter allowed `fetch()` against attacker-controlled origins, compounding XSS if bundle JSON is hostile.

## Fix

`resolveBase()` rejects `http(s)://`, `..`, and non-relative path characters; defaults to `.`.
