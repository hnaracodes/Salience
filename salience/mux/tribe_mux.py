from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

TargetSpace = Literal["parcel", "network", "vertex"]


@dataclass(frozen=True)
class TribeMuxConfig:
    hidden_dim: int = 1152
    num_clusters: int = 8
    cluster_emb_dim: int = 64
    readout_rank: int = 16
    n_parcels: int = 400
    n_networks: int = 7
    n_vertices: int = 20484
    use_film: bool = False
    target_space: TargetSpace = "parcel"

    @property
    def n_out(self) -> int:
        return {"parcel": self.n_parcels, "network": self.n_networks, "vertex": self.n_vertices}[
            self.target_space
        ]


class TribeDemographicMux(nn.Module):
    """
    Population readout + low-rank cluster-conditional delta.

    TRIBE encoders run outside this module exactly once per video.
    """

    def __init__(self, config: TribeMuxConfig) -> None:
        super().__init__()
        self.config = config
        n_out = config.n_out

        self.cluster_embedding = nn.Embedding(config.num_clusters, config.cluster_emb_dim)

        self.pop_weight = nn.Linear(config.hidden_dim, n_out, bias=True)
        self.pop_weight.weight.requires_grad_(False)
        self.pop_weight.bias.requires_grad_(False)

        self.delta_u = nn.Parameter(
            torch.zeros(config.num_clusters, config.hidden_dim, config.readout_rank)
        )
        self.delta_v = nn.Parameter(
            torch.zeros(config.num_clusters, config.readout_rank, n_out)
        )

        # FiLM modulation from cluster embedding (uses prototype table in forward)
        self.film_gamma = nn.Linear(config.cluster_emb_dim, n_out, bias=True)
        self.film_beta = nn.Linear(config.cluster_emb_dim, n_out, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cluster_embedding.weight, std=0.02)
        nn.init.normal_(self.delta_u, std=0.01)
        nn.init.normal_(self.delta_v, std=0.01)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x_univ: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        if x_univ.ndim == 2:
            return self._forward_single(x_univ, cluster_ids)
        if x_univ.ndim == 3:
            return self._forward_batched(x_univ, cluster_ids)
        raise ValueError(f"x_univ must be [T,D] or [B,T,D], got {tuple(x_univ.shape)}")

    def _forward_single(self, x_univ: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        if cluster_ids.ndim != 1:
            raise ValueError(
                f"cluster_ids must be [K] for single input, got {tuple(cluster_ids.shape)}"
            )

        T, D = x_univ.shape
        if D != self.config.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.config.hidden_dim}, got {D}")

        cluster_ids = cluster_ids.to(device=x_univ.device, dtype=torch.long)
        K = cluster_ids.shape[0]

        y_base = self.pop_weight(x_univ)  # [T, N_out]
        u = self.delta_u[cluster_ids]  # [K, D, r]
        v = self.delta_v[cluster_ids]  # [K, r, N_out]
        emb = self.cluster_embedding(cluster_ids)  # [K, D_emb]
        x_k = x_univ.unsqueeze(0).expand(K, T, D)
        y_delta = torch.einsum("ktd,kdr,kro->kto", x_k, u, v)
        gamma = self.film_gamma(emb)
        beta = self.film_beta(emb)
        y_mod = y_base.unsqueeze(0) * gamma.unsqueeze(1) + beta.unsqueeze(1)
        return y_mod + y_delta

    def _forward_batched(self, x_univ: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        B, _, D = x_univ.shape
        if D != self.config.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.config.hidden_dim}, got {D}")

        cluster_ids = cluster_ids.to(device=x_univ.device, dtype=torch.long)
        if cluster_ids.ndim == 1:
            cluster_ids = cluster_ids.unsqueeze(0).expand(B, -1)
        if cluster_ids.ndim != 2 or cluster_ids.shape[0] != B:
            raise ValueError(
                f"For batched input, cluster_ids must be [K] or [B,K], "
                f"got {tuple(cluster_ids.shape)} for B={B}"
            )

        outs = [self._forward_single(x_univ[b], cluster_ids[b]) for b in range(B)]
        return torch.stack(outs, dim=0)

    def load_population_readout(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        """Load frozen TRIBE unseen-subject readout weights (R1)."""
        with torch.no_grad():
            self.pop_weight.weight.copy_(weight)
            if bias is not None and self.pop_weight.bias is not None:
                self.pop_weight.bias.copy_(bias)

    def trainable_parameter_groups(
        self,
        *,
        emb_wd: float = 0.1,
        delta_wd: float = 0.01,
    ) -> list[dict]:
        return [
            {
                "params": [self.cluster_embedding.weight],
                "weight_decay": emb_wd,
                "name": "cluster_embedding",
            },
            {
                "params": list(self.film_gamma.parameters()) + list(self.film_beta.parameters()),
                "weight_decay": delta_wd,
                "name": "film_readout",
            },
            {
                "params": [self.delta_u, self.delta_v],
                "weight_decay": delta_wd,
                "name": "delta_readout",
            },
        ]


class PredsFallbackMux(nn.Module):
    """
    No-hidden-state fallback: learn cluster delta on TRIBE preds[T, V].

    Uses preds as input features with optional linear projection to n_out.
    """

    def __init__(self, config: TribeMuxConfig) -> None:
        super().__init__()
        self.config = config
        n_out = config.n_out
        self.input_proj = nn.Linear(config.n_vertices, config.hidden_dim, bias=False)
        self.mux = TribeDemographicMux(
            TribeMuxConfig(
                hidden_dim=config.hidden_dim,
                num_clusters=config.num_clusters,
                cluster_emb_dim=config.cluster_emb_dim,
                readout_rank=config.readout_rank,
                n_parcels=config.n_parcels,
                n_networks=config.n_networks,
                n_vertices=n_out if config.target_space == "vertex" else config.n_vertices,
                target_space=config.target_space,
            )
        )

    def forward(self, preds: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        if preds.ndim == 2:
            x = self.input_proj(preds)
            return self.mux(x, cluster_ids)
        if preds.ndim == 3:
            B, T, V = preds.shape
            x = self.input_proj(preds.reshape(B * T, V)).reshape(B, T, self.config.hidden_dim)
            return self.mux(x, cluster_ids)
        raise ValueError(f"preds must be [T,V] or [B,T,V], got {tuple(preds.shape)}")
