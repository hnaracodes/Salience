"""Inference contract tests using an injectable fake model (no weights)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from scout_core.deepgaze_msdb.inference import predict_saliency  # noqa: E402
from scout_core.deepgaze_msdb.io import write_prediction_artifacts  # noqa: E402


class _FakeMSDB(torch.nn.Module):
    """Returns a peaked log-density matching the input spatial size."""

    def forward(self, image, centerbias, pixel_per_dva=35.0, dataset=None):
        b, _, h, w = image.shape
        # Mild center preference + respect centerbias mass.
        yy = torch.linspace(-1, 1, h, device=image.device).view(1, h, 1)
        xx = torch.linspace(-1, 1, w, device=image.device).view(1, 1, w)
        prior = -((yy**2) + (xx**2))
        out = prior.expand(b, h, w).to(torch.float32) + 0.1 * centerbias
        out = out - torch.logsumexp(out.view(b, -1), dim=1).view(b, 1, 1)
        return out


def test_predict_saliency_with_fake_model(tmp_path):
    img = np.zeros((64, 96, 3), dtype=np.uint8)
    img[20:40, 30:60] = 255
    path = tmp_path / "frame.jpg"
    Image.fromarray(img).save(path)

    result = predict_saliency(
        path,
        pixels_per_degree=35.0,
        device="cpu",
        centerbias="uniform",
        model=_FakeMSDB(),
        image_id="frame",
    )

    assert result.density.shape == (64, 96)
    assert result.density.dtype == np.float32
    assert np.isfinite(result.density).all()
    assert result.density.min() >= 0
    assert result.density.sum() == pytest.approx(1.0, rel=1e-4)
    assert result.log_density.shape == (64, 96)
    assert result.provenance["dataset_mode"] is None
    assert result.provenance["pixels_per_degree"] == 35.0
    assert result.timings["model_seconds"] >= 0
    assert "entropy" in result.summary
    assert "AUC" not in result.summary

    paths = write_prediction_artifacts(
        result,
        tmp_path / "out",
        stem="frame",
        source_image=path,
    )
    assert (tmp_path / "out" / "frame_density.npy").is_file()
    assert (tmp_path / "out" / "frame_heatmap.png").is_file()
    assert (tmp_path / "out" / "frame_overlay.png").is_file()
    assert (tmp_path / "out" / "frame_meta.json").is_file()
    assert "density" in paths


def test_default_ppd_warning():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.warns(UserWarning, match="pixels_per_degree unset"):
        result = predict_saliency(
            img,
            pixels_per_degree=None,
            allow_default_ppd=False,
            device="cpu",
            centerbias="uniform",
            model=_FakeMSDB(),
        )
    assert result.provenance["used_default_ppd"] is True
