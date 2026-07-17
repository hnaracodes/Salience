from __future__ import annotations

from pathlib import Path
from typing import Any

import modal
import numpy as np

from data_prep.viability_partition import load_m0_report, m0_passed_for_training
from salience.modal_app.image_defs import M0_REPORT_PATH, app, mux_image, volume
from salience.modal_app.stream_protocol import build_stream_payload, save_mux_parcel_npz
from salience.mux.inference import DISCLAIMER, infer_mux_arrays, load_mux_for_inference
from salience.mux.latent_extractor import LatentMode, TribeLatentExtractor


@app.cls(
    gpu="A100",
    image=mux_image,
    volumes={"/mnt/mux": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
class DemographicMuxInference:
    @modal.enter()
    def setup(self) -> None:
        ckpt = Path("/mnt/mux/checkpoints/latest_mux.pt")
        if not ckpt.is_file():
            raise RuntimeError(f"Missing mux checkpoint: {ckpt}")
        self.mux = load_mux_for_inference(ckpt, device="cuda")
        self.extractor = TribeLatentExtractor(cache_folder="/mnt/mux/hf_cache", device="cuda")
        m0_path = Path(M0_REPORT_PATH)
        self.m0_passed = m0_passed_for_training(m0_path)

    @modal.method()
    def infer_mux(
        self,
        video_bytes: bytes,
        cluster_ids: list[int],
        *,
        k_chunk: int = 8,
        session_id: str = "unknown",
    ) -> dict[str, Any]:
        import tempfile

        from salience.mux.tribe_mux import PredsFallbackMux

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name

        latent = self.extractor.encode_from_video_path(video_path)
        Path(video_path).unlink(missing_ok=True)

        if latent.mode == LatentMode.PREDS_FALLBACK:
            fallback = PredsFallbackMux(self.mux.config).cuda()
            fallback.mux.load_state_dict(self.mux.state_dict())
            preds = latent.preds.cuda()
            parcel_ts = infer_mux_arrays(
                fallback,
                preds=preds,
                cluster_ids=cluster_ids,
                k_chunk=k_chunk,
            )
        else:
            parcel_ts = infer_mux_arrays(
                self.mux.cuda(),
                x_univ=latent.x_univ.cuda(),
                cluster_ids=cluster_ids,
                k_chunk=k_chunk,
            )

        out_path = Path(f"/mnt/mux/sessions/{session_id}/mux_parcel_ts.npz")
        provenance = {
            "mux_checkpoint": "/mnt/mux/checkpoints/latest_mux.pt",
            "m0_passed": self.m0_passed,
            "eval_gate_passed": False,
            "disclaimer": DISCLAIMER,
            "latent_mode": latent.mode.value,
        }
        save_mux_parcel_npz(out_path, parcel_ts, cluster_ids, provenance)
        volume.commit()

        payload = build_stream_payload(
            parcel_ts,
            cluster_ids,
            provenance=provenance,
            k_chunk=k_chunk,
        )
        payload["parcel_ts_uri"] = str(out_path)
        return payload
