"""Upstream DeepGaze license status (must be surfaced before production use)."""

from __future__ import annotations

LICENSE_STATUS = "no_active_license_upstream"

LICENSE_BANNER = """\
========================================================================
DEEPGAZE MSDB - NO ACTIVE UPSTREAM LICENSE
========================================================================
Upstream repository: https://github.com/matthias-k/DeepGaze
- setup.py MIT license classifiers/fields are commented out
- No LICENSE file grants rights for commercial use
- Commercial-use request remains unanswered:
  https://github.com/matthias-k/DeepGaze/issues/15

This repository's noncommercial/open-source intent is NOT a license grant
from the DeepGaze authors. The pipeline integration remains evaluation and
research only. Do not redistribute weights, use commercially, or ship
customer-facing outputs until written permission or an explicit upstream
license exists.
========================================================================
"""


def license_banner() -> str:
    return LICENSE_BANNER
