"""Adapter for TRIBE v2 subcortical checkpoint inference."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from scout_core.subcortical.atlas import N_SUBCORTICAL_VOXELS, load_subcortical_manifest

DEFAULT_SUBCORTICAL_CHECKPOINT = os.environ.get(
    "TRIBEV2_SUBCORTICAL_CHECKPOINT", "facebook/tribev2-subcortical"
)
CORTICAL_CHECKPOINT_FOR_BUILD_ARGS = os.environ.get(
    "TRIBEV2_CORTICAL_CHECKPOINT", "facebook/tribev2"
)


def _infer_n_outputs_from_state_dict(state_dict: dict[str, Any], default: int) -> int:
    """Guess fMRI output dimension from checkpoint tensors."""
    for key, tensor in state_dict.items():
        if not hasattr(tensor, "shape") or len(tensor.shape) < 1:
            continue
        lowered = key.lower()
        if "predictor.bias" in lowered:
            return int(tensor.shape[-1])
        if "predictor.weights" in lowered:
            return int(tensor.shape[-1])
    for key, tensor in state_dict.items():
        if not hasattr(tensor, "shape") or len(tensor.shape) < 1:
            continue
        lowered = key.lower()
        if any(token in lowered for token in ("readout", "head", "decoder", "output")):
            return int(tensor.shape[0])
    return default


def load_tribev2_checkpoint_model(
    checkpoint: str,
    *,
    cache_folder: str = "/cache",
    fallback_build_args_from: str = CORTICAL_CHECKPOINT_FOR_BUILD_ARGS,
) -> Any:
    """Load TribeModel from HF; patch subcortical exports missing model_build_args."""
    import torch
    import yaml
    from exca import ConfDict
    from huggingface_hub import hf_hub_download
    from tribev2.demo_utils import TribeModel

    repo_id = str(checkpoint)
    config_path = hf_hub_download(repo_id, "config.yaml")
    ckpt_path = hf_hub_download(repo_id, "best.ckpt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, "r", encoding="utf-8") as handle:
        config = ConfDict(yaml.load(handle, Loader=yaml.UnsafeLoader))
    for modality in ("text", "audio", "video"):
        config[f"data.{modality}_feature.infra.folder"] = cache_folder
        config[f"data.{modality}_feature.infra.cluster"] = None
    for param in (
        "infra.workdir",
        "data.study.infra_timelines",
        "data.neuro.infra",
        "data.image_feature.infra",
    ):
        config.pop(param, None)
    config["data.study.path"] = "."
    config["average_subjects"] = True
    config["checkpoint_path"] = ckpt_path
    config["cache_folder"] = cache_folder

    xp = TribeModel(**config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    build_args = ckpt.get("model_build_args")
    if build_args is None:
        cortical_ckpt_path = hf_hub_download(fallback_build_args_from, "best.ckpt")
        cortical_ckpt = torch.load(
            cortical_ckpt_path, map_location="cpu", weights_only=True, mmap=True
        )
        build_args = dict(cortical_ckpt["model_build_args"])
        build_args["n_outputs"] = _infer_n_outputs_from_state_dict(
            ckpt["state_dict"],
            N_SUBCORTICAL_VOXELS,
        )
    state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model = xp.brain_model_config.build(**build_args)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.to(device)
    model.eval()
    xp._model = model
    return xp


def fake_subcortical_preds(n_timesteps: int) -> np.ndarray:
    """Synthetic (T, 8802) zeros for FAKE_TRIBE / CI."""
    return np.zeros((max(1, n_timesteps), N_SUBCORTICAL_VOXELS), dtype=np.float32)


def predict_subcortical(
    events_df: Any,
    *,
    cache_folder: str = "/cache",
    checkpoint: str | None = None,
    cortical_model: Any | None = None,
    sub_model: Any | None = None,
    allow_proxy_fallback: bool = False,
) -> np.ndarray:
    """Predict subcortical BOLD from tribev2 events dataframe.

    Uses a dedicated subcortical TribeModel when available. Cortical-derived proxy
    is opt-in only (``allow_proxy_fallback=True``) for dev/CI.
    """
    preds, _meta = predict_subcortical_with_meta(
        events_df,
        cache_folder=cache_folder,
        checkpoint=checkpoint,
        cortical_model=cortical_model,
        sub_model=sub_model,
        allow_proxy_fallback=allow_proxy_fallback,
    )
    return preds


def predict_subcortical_with_meta(
    events_df: Any,
    *,
    cache_folder: str = "/cache",
    checkpoint: str | None = None,
    cortical_model: Any | None = None,
    sub_model: Any | None = None,
    allow_proxy_fallback: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict subcortical BOLD and report whether output came from the real checkpoint."""
    checkpoint = checkpoint or DEFAULT_SUBCORTICAL_CHECKPOINT
    manifest = load_subcortical_manifest()
    load_error: Exception | None = None

    try:
        model = sub_model
        if model is None:
            model = load_tribev2_checkpoint_model(
                checkpoint,
                cache_folder=cache_folder,
            )
        preds, _segments = model.predict(events=events_df)
        arr = np.asarray(preds, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] > N_SUBCORTICAL_VOXELS:
            arr = arr[:, :N_SUBCORTICAL_VOXELS]
        raw_shape = list(arr.shape)
        if arr.ndim == 2 and arr.shape[1] == N_SUBCORTICAL_VOXELS:
            return arr, {
                "prediction_source": "tribev2_subcortical_checkpoint",
                "checkpoint": checkpoint,
                "raw_output_shape": raw_shape,
            }
        n_regions = len(manifest.get("region_labels", []))
        if arr.ndim == 2 and arr.shape[1] == n_regions:
            return _expand_region_preds(arr, manifest), {
                "prediction_source": "tribev2_subcortical_checkpoint_regions",
                "checkpoint": checkpoint,
                "raw_output_shape": raw_shape,
            }
        raise ValueError(
            f"Unexpected subcortical output shape {arr.shape}; "
            f"expected (T, {N_SUBCORTICAL_VOXELS}) or (T, {n_regions})"
        )
    except Exception as exc:
        load_error = exc

    if allow_proxy_fallback and cortical_model is not None:
        cortical_preds, _ = cortical_model.predict(events=events_df)
        proxy = _cortical_to_subcortical_proxy(np.asarray(cortical_preds, dtype=np.float32))
        return proxy, {
            "prediction_source": "cortical_proxy_fallback",
            "checkpoint": checkpoint,
            "load_error": str(load_error) if load_error else None,
        }

    msg = f"Subcortical prediction failed for checkpoint {checkpoint!r}"
    if load_error is not None:
        raise RuntimeError(msg) from load_error
    raise RuntimeError(msg)


def _expand_region_preds(region_preds: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    """Broadcast (T, 8) region outputs to (T, 8802) by repeating within each region."""
    labels = manifest.get("region_labels") or []
    counts = manifest.get("region_voxel_counts") or {}
    t_count = region_preds.shape[0]
    cols: list[np.ndarray] = []
    for i, lbl in enumerate(labels):
        n = int(counts.get(lbl, N_SUBCORTICAL_VOXELS // max(len(labels), 1)))
        col = np.repeat(region_preds[:, i : i + 1], n, axis=1)
        cols.append(col)
    out = np.concatenate(cols, axis=1)
    if out.shape[1] != N_SUBCORTICAL_VOXELS:
        # Pad or trim to exact size
        if out.shape[1] < N_SUBCORTICAL_VOXELS:
            pad = np.zeros((t_count, N_SUBCORTICAL_VOXELS - out.shape[1]), dtype=np.float32)
            out = np.concatenate([out, pad], axis=1)
        else:
            out = out[:, :N_SUBCORTICAL_VOXELS]
    return out.astype(np.float32)


def _cortical_to_subcortical_proxy(cortical_preds: np.ndarray) -> np.ndarray:
    """Map cortical network means to 8 subcortical ROI proxies (deterministic dev fallback)."""
    from scout_core.subcortical.atlas import build_region_index, roi_means_from_voxels

    t_count = cortical_preds.shape[0]
    # Use cortical summary stats to seed region-varying proxies
    cortical_mean = cortical_preds.mean(axis=1, keepdims=True)
    cortical_std = cortical_preds.std(axis=1, keepdims=True)
    region_index = build_region_index()
    n_regions = int(region_index.max()) + 1
    region_ts = np.zeros((t_count, n_regions), dtype=np.float32)
    for rid in range(n_regions):
        weight = 0.85 + 0.03 * rid
        region_ts[:, rid] = (cortical_mean[:, 0] * weight + cortical_std[:, 0] * 0.1).astype(
            np.float32
        )
    # Expand to voxel space
    voxels = np.zeros((t_count, N_SUBCORTICAL_VOXELS), dtype=np.float32)
    for rid in range(n_regions):
        mask = region_index == rid
        voxels[:, mask] = region_ts[:, rid : rid + 1]
    # Sanity: ROI means round-trip
    _ = roi_means_from_voxels(voxels, region_index)
    return voxels


def subcortical_npz_bytes(
    preds: np.ndarray,
    *,
    tribe_checkpoint: str | None = None,
    prediction_source: str | None = None,
) -> bytes:
    """Serialise subcortical preds to compressed NPZ bytes (Modal return)."""
    import io

    manifest = load_subcortical_manifest()
    labels = manifest.get("region_labels") or []
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        preds=np.asarray(preds, dtype=np.float32),
        atlas_id=np.array(manifest.get("atlas_id", "harvard_oxford_subcortical_tribev2")),
        region_labels=np.array(json.dumps(labels)),
        tribe_checkpoint=np.array(
            tribe_checkpoint or manifest.get("tribe_subcortical_checkpoint", DEFAULT_SUBCORTICAL_CHECKPOINT)
        ),
        prediction_source=np.array(prediction_source or "unknown"),
    )
    return buf.getvalue()
