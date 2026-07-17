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


# ---------------------------------------------------------------------------
# Feature Isolation — Spatial Credit Assignment (schema v2)
# ---------------------------------------------------------------------------

class GroundingResult(BaseModel):
    """Winning DOM element selected by attention density argmax."""

    dom_id: str = ""
    tag: str = ""
    bbox: list[int] = Field(default_factory=list)
    attention_density: float = 0.0


class HeatmapProvenance(BaseModel):
    """Tracks whether a heatmap came from a real Modal inference or a placeholder."""

    heatmap_path: str | None = None
    source: str | None = None          # "modal" | "uniform_placeholder" | None
    placeholder: bool = False
    frame_path: str | None = None
    sha256: str | None = None
    alignment_status: str | None = None  # "exact" | "tolerated" | "clamped" | "invalid"


class GroundingEvent(BaseModel):
    """One neural spike + its spatially grounded UI hypothesis.

    Appended to ``AnalysisBundle.events`` by the feature isolation pipeline.
    The ``grounding`` field is ``None`` when a heatmap or manifest was unavailable.
    """

    type: str = "neural_spike_grounding"
    t_spike: int
    triggers: dict[str, Any] = Field(default_factory=dict)
    grounding: GroundingResult | None = None
    grounding_skip_reason: str | None = None
    heatmap_provenance: HeatmapProvenance | None = None
    neural_context: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Shared bundle path helper
# ---------------------------------------------------------------------------

def analysis_bundle_path(session_dir: Path) -> Path:
    return session_dir / "analysis_bundle.json"


# ---------------------------------------------------------------------------
# Top-level analysis bundle
# ---------------------------------------------------------------------------

class SessionCaptureMeta(BaseModel):
    """Website session capture metadata (video + manifest alignment)."""

    video_path: str = "walkthrough.mp4"
    manifest_schema_version: int = 2
    tr_duration_sec: float = 1.0
    alignment_ok: bool | None = None
    alignment_message: str | None = None


class MarketingNarrativeSection(BaseModel):
    section_id: str
    narrative: str = ""
    actions: list[str] = Field(default_factory=list)


class ElementInsight(BaseModel):
    """Plain-language interpretation of one DOM component vs the site goal."""

    dom_id: str
    section_id: str = ""
    role: str = ""
    text: str = ""
    job: str = ""
    reaction: str = "neutral"  # engaging | boring | friction | neutral
    attention_density: float | None = None
    engagement_z: float | None = None
    emotion: str | None = None
    salient_t: int | None = None
    plain_summary: str = ""
    recommendation: str = ""


class FrameInsight(BaseModel):
    """One-sentence caption for a sampled TR."""

    t: int
    caption: str = ""


class MarketingNarrative(BaseModel):
    """LLM-generated prose from structured section_report (not from raw HTML)."""

    executive_summary: str = ""
    site_goal: str = ""
    sections: list[MarketingNarrativeSection] = Field(default_factory=list)
    element_insights: list[ElementInsight] = Field(default_factory=list)
    frame_insights: list[FrameInsight] = Field(default_factory=list)
    provider: str = "template"
    model: str | None = None


class AnalysisBundle(BaseModel):
    """Top-level session analysis artifact written to analysis_bundle.json.

    schema_version history:
        1 — parcellation + threshold_hits only (analyze_session.py).
        2 — adds ``events`` list with neural spike grounding (feature isolation).
        3 — website session: section_report, session_capture, marketing_narrative.
        4 — adds ``marketing_scores`` (0–100 display curve + session rubric).
        5 — adds supervised ``emotion_track`` decoder fields (mode, probabilities).
    """

    schema_version: int = 1
    session_id: str
    norm_id: str
    fps: float
    parcel_ids: list[int]
    network_ids: list[int]
    network_names: list[str]
    threshold_hits: list[dict[str, Any]] = Field(default_factory=list)
    insight_catalog_version: str = "1"
    # Feature Isolation — populated when --ground flag is used
    events: list[dict[str, Any]] = Field(default_factory=list)
    # Website pipeline (schema v3)
    session_capture: SessionCaptureMeta | None = None
    section_report: list[dict[str, Any]] = Field(default_factory=list)
    marketing_narrative: MarketingNarrative | None = None
