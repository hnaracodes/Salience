"""CLI smoke tests with a monkeypatched model loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from scripts.deepgazeCode import run_frame_saliency as cli  # noqa: E402


class _FakeMSDB(torch.nn.Module):
    def forward(self, image, centerbias, pixel_per_dva=35.0, dataset=None):
        b, _, h, w = image.shape
        out = torch.zeros((b, h, w), dtype=torch.float32, device=image.device)
        out[:, h // 2, w // 2] = 5.0
        out = out - torch.logsumexp(out.view(b, -1), dim=1).view(b, 1, 1)
        return out


def test_cli_writes_artifacts(tmp_path, monkeypatch):
    img_path = tmp_path / "shot.png"
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(img_path)
    out_dir = tmp_path / "out"

    def _fake_load_model(device="auto", pretrained=True, model_factory=None):
        return _FakeMSDB(), "cpu", {"source": "fake"}

    monkeypatch.setattr(cli, "load_model", _fake_load_model)

    rc = cli.main(
        [
            "--input",
            str(img_path),
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--pixels-per-degree",
            "35",
            "--centerbias",
            "uniform",
            "--quiet-license",
        ]
    )
    assert rc == 0
    assert (out_dir / "shot_density.npy").is_file()
    assert (out_dir / "shot_heatmap.png").is_file()
    assert (out_dir / "shot_overlay.png").is_file()
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_images"] == 1
    dens = np.load(out_dir / "shot_density.npy")
    assert dens.shape == (48, 64)
    assert dens.sum() == pytest.approx(1.0, rel=1e-4)


def test_cli_requires_ppd_flag(tmp_path):
    img_path = tmp_path / "shot.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(img_path)
    rc = cli.main(
        [
            "--input",
            str(img_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--quiet-license",
        ]
    )
    assert rc == 2
