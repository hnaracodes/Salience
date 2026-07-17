from __future__ import annotations

import json
from pathlib import Path

import modal

from data_prep.viability_partition import m0_passed_for_training
from salience.modal_app.image_defs import M0_REPORT_PATH, app, mux_image, volume


@app.cls(
    gpu="A100",
    image=mux_image,
    volumes={"/mnt/mux": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60 * 8,
)
class DemographicMuxTrainer:
    @modal.enter()
    def setup(self) -> None:
        import torch

        from salience.mux.latent_extractor import TribeLatentExtractor
        from salience.mux.tribe_mux import TribeDemographicMux, TribeMuxConfig

        m0_path = Path(M0_REPORT_PATH)
        if not m0_passed_for_training(m0_path):
            raise RuntimeError(
                f"M0 gate not passed — refusing to train. Report: {m0_path}. "
                "Run scripts/run_demographic_viability.py on real data first."
            )

        self.device = torch.device("cuda")
        self.latent_extractor = TribeLatentExtractor(cache_folder="/mnt/mux/hf_cache")
        config = TribeMuxConfig(
            hidden_dim=1152,
            num_clusters=8,
            cluster_emb_dim=64,
            readout_rank=16,
            n_parcels=400,
            target_space="parcel",
        )
        self.mux = TribeDemographicMux(config).to(self.device)
        self.scaler = torch.amp.GradScaler("cuda")

    @modal.method()
    def train(self, shard_urls: list[str], *, epochs: int = 5, batch_size: int = 1, lr: float = 3e-4) -> str:
        import time

        import torch
        import webdataset as wds

        from salience.mux.checkpoints import save_mux_checkpoint, write_metrics_json
        from salience.mux.losses import correlation_loss, delta_regularization
        from salience.mux.tribe_mux import PredsFallbackMux

        loader = (
            wds.WebDataset(shard_urls, shardshuffle=True)
            .shuffle(256)
            .to_tuple("mp4", "y_target.npy", "cluster_ids.npy", "json")
            .batched(batch_size, partial=False)
        )

        optimizer = torch.optim.AdamW(
            self.mux.trainable_parameter_groups(),
            lr=lr,
            betas=(0.9, 0.95),
        )

        global_step = 0
        self.mux.train()
        for epoch in range(epochs):
            for video_bytes_batch, y_target_np, cluster_ids_np, _meta in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    latent = self.latent_extractor.encode_video_batch(list(video_bytes_batch))

                y_target = torch.as_tensor(y_target_np, device=self.device, dtype=torch.float32)
                cluster_ids = torch.as_tensor(cluster_ids_np, device=self.device, dtype=torch.long)
                if y_target.ndim == 3:
                    y_target = y_target.unsqueeze(0)
                if cluster_ids.ndim == 1:
                    cluster_ids = cluster_ids.unsqueeze(0)

                if latent.mode.value == "preds_fallback":
                    fallback = PredsFallbackMux(self.mux.config).to(self.device)
                    fallback.mux.load_state_dict(self.mux.state_dict())
                    x_in = latent.preds.to(self.device)
                    if x_in.ndim == 2:
                        x_in = x_in.unsqueeze(0)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        y_pred = fallback(x_in[0], cluster_ids[0])
                        y_pred = y_pred.unsqueeze(0)
                else:
                    x_univ = latent.x_univ.to(self.device, dtype=torch.float32)
                    if x_univ.ndim == 2:
                        x_univ = x_univ.unsqueeze(0)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        y_pred = self.mux(x_univ[0], cluster_ids[0]).unsqueeze(0)

                loss = correlation_loss(y_pred, y_target, dim=-1)
                loss = loss + 0.01 * delta_regularization(self.mux.delta_u, self.mux.delta_v)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                global_step += 1

        ckpt = Path("/mnt/mux/checkpoints") / f"mux_step_{global_step:08d}.pt"
        save_mux_checkpoint(ckpt, self.mux, optimizer=optimizer, step=global_step, epoch=epochs - 1)
        metrics_path = Path("/mnt/mux/checkpoints") / f"metrics_{int(time.time())}.json"
        write_metrics_json(metrics_path, {"global_step": global_step, "epochs": epochs})
        volume.commit()
        return str(metrics_path)
