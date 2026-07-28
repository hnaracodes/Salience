"""Opt-in integration test against real DeepGaze MSDB weights.

Run explicitly:
  DEEPGAZE_MSDB_INTEGRATION=1 .\\.venv-deepgaze-msdb\\Scripts\\python.exe -m pytest \\
      tests/deepgazeCode/test_integration_real_model.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.integration

RUN = os.environ.get("DEEPGAZE_MSDB_INTEGRATION", "").strip() in {"1", "true", "yes"}


@pytest.mark.skipif(not RUN, reason="Set DEEPGAZE_MSDB_INTEGRATION=1 to run")
def test_real_deepgaze_msdb_smoke(tmp_path):
    deepgaze_pytorch = pytest.importorskip("deepgaze_pytorch")
    torch = pytest.importorskip("torch")
    _ = deepgaze_pytorch  # package presence is the gate

    from scout_core.deepgaze_msdb.inference import clear_model_cache, predict_saliency
    from scout_core.deepgaze_msdb.io import write_prediction_artifacts

    clear_model_cache()
    # MSDB multi-scale CLIP needs reasonably large inputs; tiny images collapse
    # spatial dims inside RN50x64 avg-pool (RuntimeError: output size 0x0).
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[180:420, 320:960] = [240, 200, 40]
    img[300:380, 700:820] = [20, 20, 220]
    path = tmp_path / "synthetic.png"
    Image.fromarray(img).save(path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = predict_saliency(
        path,
        pixels_per_degree=35.0,
        device=device,
        centerbias="mit1003",
        allow_default_ppd=True,
        image_id="synthetic",
    )
    assert result.density.shape == (720, 1280)
    assert result.density.sum() == pytest.approx(1.0, rel=1e-3)
    assert np.isfinite(result.density).all()
    assert result.timings["model_seconds"] > 0
    paths = write_prediction_artifacts(
        result, tmp_path / "out", stem="synthetic", source_image=path
    )
    assert Path(paths["heatmap"]).is_file()
