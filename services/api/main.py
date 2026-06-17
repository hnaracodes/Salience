"""Salience API — thin job orchestrator.

Endpoints:
  POST /v1/scans         — submit URL, enqueue job
  GET  /v1/scans         — list user's scans (paginated)
  GET  /v1/scans/{id}    — status + viewer URL
  DELETE /v1/scans/{id}  — purge scan + object storage artifacts
"""
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import boto3
from arq import create_pool
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from services.api import ssrf
from services.api.auth import get_current_user
from services.api.models import Base, Scan, ScanArtifact, User
from services.api.schemas import ScanCreate, ScanListResponse, ScanResponse
from services.infra.config import load_object_storage_config, load_redis_settings

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./scout.db")

_CORS_ORIGINS = json.loads(os.environ.get("CORS_ORIGINS", '["http://localhost:3000"]'))

# Use synchronous SQLAlchemy for simplicity (asyncpg swap happens in prod via env).
_sync_url = DATABASE_URL.replace("+asyncpg", "")
engine = create_engine(_sync_url)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

arq_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global arq_pool
    # In dev, create tables directly; Alembic handles prod migrations.
    Base.metadata.create_all(engine)
    redis_settings = load_redis_settings()
    arq_pool = await create_pool(redis_settings)
    yield
    if arq_pool:
        await arq_pool.close()


app = FastAPI(title="Salience API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_or_create_user(clerk_user_id: str, db: Session) -> User:
    user = db.execute(select(User).where(User.clerk_user_id == clerk_user_id)).scalar_one_or_none()
    if user is None:
        user = User(clerk_user_id=clerk_user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def _scan_to_response(scan: Scan) -> ScanResponse:
    return ScanResponse(
        id=scan.id,
        url=scan.url,
        status=scan.status,
        current_stage=scan.current_stage,
        viewer_url=scan.viewer_url,
        error=scan.error,
        created_at=scan.created_at,
    )


@app.get("/health")
def health():
    """Railway health-check probe — no auth required."""
    return {"status": "ok"}


@app.post("/v1/scans", status_code=201, response_model=ScanResponse)
async def create_scan(
    body: ScanCreate,
    db: Session = Depends(get_db),
    claims: dict = Depends(get_current_user),
):
    clerk_user_id = claims.get("sub")
    user = _get_or_create_user(clerk_user_id, db)

    # Free-tier rate limit: max 1 active scan at a time.
    active_count = db.execute(
        select(func.count()).select_from(Scan).where(
            Scan.user_id == user.id,
            Scan.status.in_(["queued", "running"]),
        )
    ).scalar()
    if active_count and active_count >= 1:
        raise HTTPException(status_code=429, detail="Free tier allows 1 concurrent scan")

    # Idempotency: return existing non-failed scan if key matches.
    if body.idempotency_key:
        existing = db.execute(
            select(Scan).where(
                Scan.idempotency_key == body.idempotency_key,
                Scan.user_id == user.id,
                Scan.status != "failed",
            )
        ).scalar_one_or_none()
        if existing:
            return _scan_to_response(existing)

    url_str = str(body.url)
    try:
        ssrf.validate_url(url_str)
    except ssrf.SSRFViolation as exc:
        raise HTTPException(status_code=422, detail=f"URL rejected: {exc.reason}") from exc

    session_id = uuid4().hex
    scan = Scan(
        user_id=user.id,
        url=url_str,
        site_goal=body.site_goal,
        status="queued",
        session_id=session_id,
        idempotency_key=body.idempotency_key,
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    await arq_pool.enqueue_job("run_scan_task", scan.id, session_id, url_str, body.site_goal)

    return _scan_to_response(scan)


@app.get("/v1/scans", response_model=ScanListResponse)
def list_scans(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    claims: dict = Depends(get_current_user),
):
    user = _get_or_create_user(claims["sub"], db)
    offset = (page - 1) * page_size

    total = db.execute(
        select(func.count()).select_from(Scan).where(Scan.user_id == user.id)
    ).scalar()

    scans = db.execute(
        select(Scan)
        .where(Scan.user_id == user.id)
        .order_by(Scan.created_at.desc())
        .offset(offset)
        .limit(page_size)
    ).scalars().all()

    return ScanListResponse(scans=[_scan_to_response(s) for s in scans], total=total or 0)


@app.get("/v1/scans/{scan_id}", response_model=ScanResponse)
def get_scan(
    scan_id: str,
    db: Session = Depends(get_db),
    claims: dict = Depends(get_current_user),
):
    user = _get_or_create_user(claims["sub"], db)
    scan = db.get(Scan, scan_id)
    if scan is None or scan.user_id != user.id:
        raise HTTPException(status_code=404, detail="Scan not found")
    return _scan_to_response(scan)


@app.delete("/v1/scans/{scan_id}", status_code=204)
def delete_scan(
    scan_id: str,
    db: Session = Depends(get_db),
    claims: dict = Depends(get_current_user),
):
    user = _get_or_create_user(claims["sub"], db)
    scan = db.get(Scan, scan_id)
    if scan is None or scan.user_id != user.id:
        raise HTTPException(status_code=404, detail="Scan not found")

    _delete_object_storage_artifacts(scan_id)

    db.delete(scan)
    db.commit()


def _delete_object_storage_artifacts(scan_id: str):
    """Remove all objects under scans/{scan_id}/. Silently skips if storage is unconfigured."""
    cfg = load_object_storage_config()
    if not cfg.configured:
        return

    s3 = boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
    )

    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"scans/{scan_id}/"

    for page in paginator.paginate(Bucket=cfg.bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        if not objects:
            continue
        s3.delete_objects(
            Bucket=cfg.bucket,
            Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
        )
