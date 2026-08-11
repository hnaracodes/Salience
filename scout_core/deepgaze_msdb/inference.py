"""Standalone DeepGaze MSDB inference (evaluation-only)."""

from __future__ import annotations

import platform
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from scout_core.deepgaze_msdb.assets import ensure_centerbias
from scout_core.deepgaze_msdb.constants import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_PIXELS_PER_DEGREE,
    MODEL_ID,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    WEIGHTS_RELEASE,
    WEIGHTS_URL,
)
from scout_core.deepgaze_msdb.license_notice import LICENSE_STATUS
from scout_core.deepgaze_msdb.preprocess import (
    ImageInput,
    density_from_log_density,
    load_rgb_image,
    maybe_resize_long_side,
    prepare_centerbias,
    resize_density,
)
from scout_core.deepgaze_msdb.summaries import summarize_density

# Optional injectable model factory for unit tests (no real weights).
ModelFactory = Callable[[str], Any]

_MODEL_CACHE: dict[str, Any] = {}


@dataclass
class SaliencyResult:
    """Normalized gaze-density prediction plus provenance and timings."""

    density: np.ndarray
    log_density: np.ndarray
    summary: dict[str, Any]
    provenance: dict[str, Any]
    timings: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "provenance": self.provenance,
            "timings": self.timings,
            "warnings": list(self.warnings),
            "license_status": LICENSE_STATUS,
        }


def resolve_device(device: str | None = "auto") -> str:
    import torch

    requested = (device or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device}'. Use auto|cpu|cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return requested


def _import_deepgaze():
    try:
        import deepgaze_pytorch
        import torch
    except ImportError as exc:
        raise ImportError(
            "DeepGaze MSDB dependencies are missing. Install the isolated env "
            "from requirements-deepgaze-msdb.txt into .venv-deepgaze-msdb."
        ) from exc
    return deepgaze_pytorch, torch


def load_model(
    device: str = "auto",
    pretrained: bool = True,
    model_factory: Optional[ModelFactory] = None,
) -> tuple[Any, str, dict[str, Any]]:
    """Load (and cache) DeepGazeMSDB on the resolved device."""
    resolved = resolve_device(device)
    if model_factory is not None:
        model = model_factory(resolved)
        return model, resolved, {"source": "injected_factory", "pretrained": pretrained}

    cache_key = f"{resolved}:pretrained={pretrained}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], resolved, {"source": "cache", "pretrained": pretrained}

    deepgaze_pytorch, torch = _import_deepgaze()
    t0 = time.perf_counter()
    model = deepgaze_pytorch.DeepGazeMSDB(pretrained=pretrained)
    model = model.to(resolved)
    model.eval()
    load_s = time.perf_counter() - t0
    meta = {
        "source": "deepgaze_pytorch.DeepGazeMSDB",
        "pretrained": pretrained,
        "load_seconds": load_s,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": (
            torch.cuda.get_device_name(0) if resolved == "cuda" else platform.processor()
        ),
    }
    _MODEL_CACHE[cache_key] = model
    return model, resolved, meta


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def predict_saliency(
    image: ImageInput,
    *,
    pixels_per_degree: float | None = None,
    device: str = "auto",
    centerbias: str = "mit1003",
    checkpoint_dir: Path | str | None = None,
    max_long_side: int | None = None,
    allow_default_ppd: bool = False,
    model: Any | None = None,
    model_factory: Optional[ModelFactory] = None,
    image_id: str | None = None,
) -> SaliencyResult:
    """Predict a fixation probability density for one screenshot.

    Uses MSDB unknown-domain mode (``dataset=None``). AUC-style metrics are not
    produced because they require ground-truth fixations.
    """
    warn_list: list[str] = []
    used_default_ppd = False
    if pixels_per_degree is None:
        used_default_ppd = True
        pixels_per_degree = DEFAULT_PIXELS_PER_DEGREE
        msg = (
            f"pixels_per_degree unset; using MIT1003 default "
            f"{DEFAULT_PIXELS_PER_DEGREE}. Screenshots do not encode viewing "
            "geometry — set --pixels-per-degree explicitly for product work."
        )
        warn_list.append(msg)
        if not allow_default_ppd:
            warnings.warn(msg, UserWarning, stacklevel=2)
    if pixels_per_degree <= 0:
        raise ValueError("pixels_per_degree must be positive")

    t_all = time.perf_counter()
    t0 = time.perf_counter()
    rgb = load_rgb_image(image)
    rgb_infer, orig_hw = maybe_resize_long_side(rgb, max_long_side)
    if rgb_infer.shape[:2] != orig_hw:
        warn_list.append(
            f"Resized long side for inference: {orig_hw} -> {rgb_infer.shape[:2]}"
        )

    uniform = centerbias.lower() == "uniform"
    cb_prov: dict[str, Any]
    if uniform:
        template = None
        cb_prov = {"mode": "uniform"}
    else:
        template, cb_prov = ensure_centerbias(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
        cb_prov = {"mode": "mit1003", **cb_prov}

    cb = prepare_centerbias(
        rgb_infer.shape[0],
        rgb_infer.shape[1],
        centerbias_template=template,
        uniform=uniform,
    )
    preprocess_s = time.perf_counter() - t0

    if model is None:
        model, resolved, model_meta = load_model(
            device=device, pretrained=True, model_factory=model_factory
        )
        _, torch = _import_deepgaze()
    else:
        import torch

        resolved = resolve_device(device)
        # Move injectable/test models onto the requested device.
        if hasattr(model, "to"):
            model = model.to(resolved)
        if hasattr(model, "eval"):
            model.eval()
        model_meta = {"source": "provided_model"}

    image_tensor = torch.tensor(
        np.ascontiguousarray(rgb_infer.transpose(2, 0, 1))[None],
        dtype=torch.float32,
        device=resolved,
    )
    centerbias_tensor = torch.tensor(
        cb[None],
        dtype=torch.float32,
        device=resolved,
    )

    if resolved == "cuda":
        torch.cuda.synchronize()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()

    t1 = time.perf_counter()
    with torch.inference_mode():
        log_density_t = model(
            image_tensor,
            centerbias_tensor,
            pixel_per_dva=float(pixels_per_degree),
            dataset=None,
        )
    if resolved == "cuda":
        torch.cuda.synchronize()
    model_s = time.perf_counter() - t1

    # Upstream returns (B, H, W); tolerate accidental channel dim.
    log_np = log_density_t.detach().float().cpu().numpy()
    if log_np.ndim == 4:
        log_np = log_np[0, 0]
    elif log_np.ndim == 3:
        log_np = log_np[0]
    density = density_from_log_density(log_np)
    if density.shape != orig_hw:
        density = resize_density(density, orig_hw[0], orig_hw[1])
        # Keep log_density aligned to final density resolution for IO.
        log_np = np.log(np.clip(density.astype(np.float64), 1e-32, None))
        log_np -= log_np.max()

    peak_mem = None
    if resolved == "cuda" and hasattr(torch.cuda, "max_memory_allocated"):
        peak_mem = int(torch.cuda.max_memory_allocated())

    summary = summarize_density(density)
    total_s = time.perf_counter() - t_all
    source_path = str(image) if isinstance(image, (str, Path)) else None
    provenance = {
        "model_id": MODEL_ID,
        "family": "deepgaze_msdb",
        "license_status": LICENSE_STATUS,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "weights_release": WEIGHTS_RELEASE,
        "weights_url": WEIGHTS_URL,
        "dataset_mode": None,
        "pixels_per_degree": float(pixels_per_degree),
        "used_default_ppd": used_default_ppd,
        "allow_default_ppd": bool(allow_default_ppd),
        "centerbias": cb_prov,
        "device": resolved,
        "image_id": image_id,
        "source_path": source_path,
        "original_hw": list(orig_hw),
        "inference_hw": [int(rgb_infer.shape[0]), int(rgb_infer.shape[1])],
        "model": model_meta,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "peak_cuda_memory_bytes": peak_mem,
    }
    timings = {
        "preprocess_seconds": float(preprocess_s),
        "model_seconds": float(model_s),
        "total_seconds": float(total_s),
    }
    return SaliencyResult(
        density=density.astype(np.float32, copy=False),
        log_density=np.asarray(log_np, dtype=np.float32),
        summary=summary,
        provenance=provenance,
        timings=timings,
        warnings=warn_list,
    )


def predict_saliency_path(
    image_path: str | Path,
    **kwargs: Any,
) -> SaliencyResult:
    path = Path(image_path)
    kwargs.setdefault("image_id", path.stem)
    return predict_saliency(path, **kwargs)


def result_as_dict(result: SaliencyResult) -> dict[str, Any]:
    """JSON-serializable metadata (arrays excluded)."""
    payload = result.to_metadata()
    # dataclasses asdict would try to serialize arrays; keep explicit.
    payload["density_sha256_hint"] = {
        "shape": list(result.density.shape),
        "dtype": str(result.density.dtype),
        "sum": float(result.density.sum()),
    }
    return payload
