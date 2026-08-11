"""DeepGaze MSDB evaluation-only saliency baseline.

Not part of the default website Playwright / dual-track / Horikawa E2E pipeline.
Do not import from ``tribe``, ``dual_track``, or ``horikawaCode``.

LICENSE: Upstream DeepGaze currently ships with no active license. Evaluation use
only until written permission or an explicit license is obtained. See
``scout_core.deepgaze_msdb.license_notice``.
"""

from scout_core.deepgaze_msdb.inference import (
    SaliencyResult,
    predict_saliency,
    predict_saliency_path,
)
from scout_core.deepgaze_msdb.license_notice import LICENSE_STATUS, license_banner

__all__ = [
    "LICENSE_STATUS",
    "SaliencyResult",
    "license_banner",
    "predict_saliency",
    "predict_saliency_path",
]
