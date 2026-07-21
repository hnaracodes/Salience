from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

# Pin target TRIBE commit for R1 hidden-state spike (update after verification).
TRIBE_COMMIT_PIN = "main"


class LatentMode(str, Enum):
    HIDDEN_STATE = "hidden_state"
    PREDS_FALLBACK = "preds_fallback"


@dataclass
class LatentExtractResult:
    x_univ: torch.Tensor | None
    preds: torch.Tensor | None
    mode: LatentMode
    meta: dict[str, Any]


class TribeLatentExtractor:
    """
    Adapter around TRIBE v2 for demographic mux training/inference.

    R1: expose X_univ[T, 1152] from Transformer output after 2Hz->1Hz pool.
    Fallback: use average-subject preds[T, 20484] via PredsFallbackMux.
    """

    def __init__(
        self,
        *,
        cache_folder: str | None = None,
        model_id: str = "facebook/tribev2",
        device: str = "cuda",
    ) -> None:
        self.cache_folder = cache_folder
        self.model_id = model_id
        self.device = device
        self._model = None
        self._hidden_hook_available = False

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from tribev2.demo_utils import TribeModel

        kwargs: dict[str, Any] = {}
        if self.cache_folder:
            kwargs["cache_folder"] = self.cache_folder
        self._model = TribeModel.from_pretrained(self.model_id, **kwargs)
        self._hidden_hook_available = hasattr(self._model, "encode_universal")

        for maybe_name in ("model", "net", "module"):
            module = getattr(self._model, maybe_name, None)
            if module is not None and hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad_(False)
                module.eval()

    def encode_from_video_path(self, video_path: str) -> LatentExtractResult:
        self._ensure_model()
        df = self._model.get_events_dataframe(video_path=video_path)
        return self.encode_from_events(df)

    def encode_from_events(self, events_df: Any) -> LatentExtractResult:
        self._ensure_model()
        preds, _segments = self._model.predict(events=events_df)
        preds_t = torch.as_tensor(np.asarray(preds, dtype=np.float32))

        if self._hidden_hook_available:
            x = self._model.encode_universal(events=events_df)
            x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
            return LatentExtractResult(
                x_univ=x_t,
                preds=preds_t,
                mode=LatentMode.HIDDEN_STATE,
                meta={"tribe_commit": TRIBE_COMMIT_PIN},
            )

        return LatentExtractResult(
            x_univ=None,
            preds=preds_t,
            mode=LatentMode.PREDS_FALLBACK,
            meta={
                "tribe_commit": TRIBE_COMMIT_PIN,
                "fallback_reason": "encode_universal not exposed on TribeModel",
            },
        )

    def encode_video_batch(self, video_bytes_batch: list[bytes]) -> LatentExtractResult:
        import tempfile
        from pathlib import Path

        paths: list[str] = []
        for vb in video_bytes_batch:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(vb)
                paths.append(tmp.name)

        results = [self.encode_from_video_path(p) for p in paths]
        for p in paths:
            Path(p).unlink(missing_ok=True)

        if len(results) == 1:
            return results[0]

        # Batch stack for training loader (same mode required).
        mode = results[0].mode
        if any(r.mode != mode for r in results):
            raise RuntimeError("Mixed latent modes in batch")

        if mode == LatentMode.HIDDEN_STATE:
            x = torch.stack([r.x_univ for r in results if r.x_univ is not None], dim=0)
            preds = torch.stack([r.preds for r in results if r.preds is not None], dim=0)
            return LatentExtractResult(x_univ=x, preds=preds, mode=mode, meta=results[0].meta)

        preds = torch.stack([r.preds for r in results if r.preds is not None], dim=0)
        return LatentExtractResult(x_univ=None, preds=preds, mode=mode, meta=results[0].meta)
