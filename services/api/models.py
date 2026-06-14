from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=30)


class User(Base):
    __tablename__ = "users"

    id = Column(String(32), primary_key=True, default=_uuid)
    clerk_user_id = Column(String(255), unique=True, nullable=False)
    plan = Column(String(50), nullable=False, default="free")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    scans = relationship("Scan", back_populates="user", cascade="all, delete-orphan")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(String(32), primary_key=True, default=_uuid)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    url = Column(Text, nullable=False)
    site_goal = Column(Text, nullable=True)

    # Lifecycle
    status = Column(String(20), nullable=False, default="queued")  # queued/running/done/failed
    current_stage = Column(String(100), nullable=True)

    # Worker linkage
    session_id = Column(String(64), nullable=True)
    viewer_url = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    idempotency_key = Column(String(255), nullable=True, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    expires_at = Column(DateTime(timezone=True), nullable=False, default=_expires_at)

    user = relationship("User", back_populates="scans")
    artifacts = relationship("ScanArtifact", back_populates="scan", cascade="all, delete-orphan")


class ScanArtifact(Base):
    __tablename__ = "scan_artifacts"

    id = Column(String(32), primary_key=True, default=_uuid)
    scan_id = Column(String(32), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    artifact_type = Column(String(50), nullable=False)  # manifest/bundle/video/viewer
    r2_key = Column(String(512), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    scan = relationship("Scan", back_populates="artifacts")
