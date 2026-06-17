"""Arq background task: run the full scan pipeline for one scan."""
import asyncio
import os
import sys
from pathlib import Path

from arq.connections import RedisSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from services.api.models import Scan

DATABASE_URL = os.environ.get("DATABASE_URL", "")
DEFAULT_NORM_ID = os.environ.get("PIPELINE_NORM_ID", "synthetic_bootstrap_v1")


def _ensure_norm_bootstrap() -> None:
    """Ensure bootstrap norm parquet files exist and SQLite paths are container-portable."""
    import json
    import sqlite3
    import subprocess

    from activation_store import DB_PATH
    from scout_core.storage_migrations import (
        _portable_storage_path,
        register_norm_bundle,
        resolve_storage_path,
    )

    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            """
            SELECT path_roi_parquet, path_network_parquet, meta_json
            FROM norm_bundle WHERE norm_id = ?
            """,
            (DEFAULT_NORM_ID,),
        ).fetchone()
        if row is not None:
            roi_stored, net_stored, meta_json = row
            roi_path = resolve_storage_path(roi_stored)
            net_path = resolve_storage_path(net_stored)
            if roi_path.is_file() and net_path.is_file():
                portable_roi = _portable_storage_path(roi_path)
                portable_net = _portable_storage_path(net_path)
                if roi_stored != portable_roi or net_stored != portable_net:
                    meta = json.loads(meta_json) if meta_json else {}
                    register_norm_bundle(
                        conn,
                        norm_id=DEFAULT_NORM_ID,
                        roi_parquet=roi_path,
                        network_parquet=net_path,
                        meta=meta,
                    )
                    conn.commit()
                return
    finally:
        conn.close()

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "bootstrap_norms.py"),
            "--norm-id",
            DEFAULT_NORM_ID,
        ],
        check=True,
    )


async def run_scan_task(ctx, scan_id: str, session_id: str, url: str, site_goal: str | None):
    """Fetch the scan record, run the pipeline, and persist the result."""
    db: Session = ctx["db"]

    scan = db.get(Scan, scan_id)
    if scan is None:
        return  # Scan was deleted before the job ran

    scan.status = "running"
    db.commit()

    async def on_stage(stage: str, status: str):
        scan.current_stage = stage
        db.commit()

    try:
        from services.pipeline import runner  # noqa: PLC0415 — local import avoids heavy import on startup

        # config={} uses runner defaults (explore_production.yaml, FAKE_TRIBE env, etc.)
        result: dict = await asyncio.to_thread(
            runner.run_scan,
            session_id,
            url,
            site_goal,
            {},
            lambda stage, status: asyncio.run(on_stage(stage, status)),
            scan_id=scan_id,
        )
        scan.status = "done"
        scan.viewer_url = result["viewer_url"]
    except Exception as exc:  # noqa: BLE001
        scan.status = "failed"
        scan.error = str(exc)
    finally:
        db.commit()


class WorkerSettings:
    """Arq worker configuration."""

    functions = [run_scan_task]
    # Arq reads this class attribute at startup; None would silently fall back
    # to localhost:6379 and break in any container environment.
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379"))

    @staticmethod
    async def on_startup(ctx):
        """Inject a synchronous DB session into the worker context."""
        from sqlalchemy import create_engine  # noqa: PLC0415
        from sqlalchemy.orm import sessionmaker  # noqa: PLC0415

        engine = create_engine(DATABASE_URL.replace("+asyncpg", ""))
        Session = sessionmaker(bind=engine)
        ctx["db"] = Session()
        _ensure_norm_bootstrap()

    @staticmethod
    async def on_shutdown(ctx):
        ctx["db"].close()
