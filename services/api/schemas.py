from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class ScanCreate(BaseModel):
    url: HttpUrl
    site_goal: Optional[str] = Field(default=None, max_length=2000)
    idempotency_key: Optional[str] = Field(default=None, max_length=255)


class ScanResponse(BaseModel):
    id: str
    url: str
    status: str
    current_stage: Optional[str]
    viewer_url: Optional[str]
    error: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ScanListResponse(BaseModel):
    scans: list[ScanResponse]
    total: int
