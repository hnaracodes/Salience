from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class NormBundleMeta(BaseModel):
    norm_id: str
    tribe_checkpoint: str | None = None
    clip_corpus_id: str | None = None
    parcellation_atlas_id: str | None = None
    reducer: str = "mean"
    n_clips: int = 0
    notes: str | None = None


class ThresholdHit(BaseModel):
    session_id: str
    rule_id: str
    t_start: int
    t_end: int
    networks_json: str
    evidence_json: str
    confidence: float
    insight_key: str
    insight_text: str


class ThresholdContext(BaseModel):
    session_id: str
    fps: float = 1.0
    network_names: list[str]
    network_ts: list[list[float]]  # T x N (raw aggregate)
    z_network: list[list[float]]  # T x N


class ROIAggregateRow(BaseModel):
    parcel_id: int
    mean: float
    std: float
    n_samples: int = 0
    q05: float | None = None
    q50: float | None = None
    q95: float | None = None


def analysis_bundle_path(session_dir: Path) -> Path:
    return session_dir / "analysis_bundle.json"


class AnalysisBundle(BaseModel):
    schema_version: int = 1
    session_id: str
    norm_id: str
    fps: float
    parcel_ids: list[int]
    network_ids: list[int]
    network_names: list[str]
    threshold_hits: list[dict[str, Any]] = Field(default_factory=list)
    insight_catalog_version: str = "1"
