---
id: ISSUE-SEC-004
status: open
severity: medium
category: security
title: Explore mode trusts captured CSS selectors
---

# ISSUE-SEC-004 — Explore selector hardening (open)

## Problem

`explore_policy.element_selector` passes through `#`, `.`, `[` selectors from hostile DOM. Explore on untrusted production URLs could mis-click or error.

## Mitigation today

`same_origin_only`, deny href patterns, localhost fixture default.

## Follow-up

Prefer Playwright role/text locators; validate selector against allowlist before `page.click()`.
