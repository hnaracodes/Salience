from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DatasetId = Literal["hcp_7t", "cneuromod", "nndb", "camcan", "forrest"]
AgeBand = Literal["young_adult", "middle_adult", "older_adult"]
Sex = Literal["female", "male", "unknown"]


class SubjectMetadataRow(BaseModel):
    subject_id: str
    source_subject_id: str
    dataset: DatasetId
    age: float | None = None
    age_band: AgeBand
    sex: Sex
    site_id: str
    handedness: str | None = None
    metadata_quality: str = "unknown"
    consent_scope: str | None = None
    dua_version: str | None = None


class ClusterRule(BaseModel):
    cluster_id: int
    label: str
    where: dict[str, str]
    min_datasets: int = 1
    notes: str | None = None
    suppressed: bool = False


class M0NetworkResult(BaseModel):
    network_name: str
    network_id: int
    r2_demographic_given_site: float
    r2_site_only: float
    r2_demographic_only: float
    permutation_p: float
    permutation_null_95: float
    n_subjects: int


class M0Report(BaseModel):
    schema_version: int = 2
    passed: bool
    is_synthetic: bool = False
    multi_site_validated: bool = False
    n_sites: int = 1
    data_source: str
    n_subjects: int
    n_permutations: int
    networks: list[M0NetworkResult]
    go_criteria: dict[str, bool]
    notes: list[str] = Field(default_factory=list)
