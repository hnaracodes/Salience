from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NEURO_SCHEMA = """
CREATE TABLE IF NOT EXISTS norm_bundle (
    norm_id TEXT PRIMARY KEY,
    path_roi_parquet TEXT NOT NULL,
    path_network_parquet TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_roi_timeseries (
    session_id TEXT NOT NULL,
    t_idx INTEGER NOT NULL,
    parcel_id INTEGER NOT NULL,
    value REAL NOT NULL,
    z_vs_norm REAL,
    norm_id TEXT NOT NULL,
    PRIMARY KEY (session_id, t_idx, parcel_id, norm_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (norm_id) REFERENCES norm_bundle(norm_id)
);

CREATE TABLE IF NOT EXISTS session_network_timeseries (
    session_id TEXT NOT NULL,
    t_idx INTEGER NOT NULL,
    yeo_network_id INTEGER NOT NULL,
    value REAL NOT NULL,
    z_vs_norm REAL,
    norm_id TEXT NOT NULL,
    PRIMARY KEY (session_id, t_idx, yeo_network_id, norm_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (norm_id) REFERENCES norm_bundle(norm_id)
);

CREATE TABLE IF NOT EXISTS threshold_hit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    t_start INTEGER NOT NULL,
    t_end INTEGER NOT NULL,
    networks_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    insight_key TEXT NOT NULL,
    insight_text TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_thresh_session ON threshold_hit(session_id);
CREATE INDEX IF NOT EXISTS idx_roi_ts ON session_roi_timeseries(session_id, t_idx);
CREATE INDEX IF NOT EXISTS idx_net_ts ON session_network_timeseries(session_id, t_idx);
"""


def ensure_neuro_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(NEURO_SCHEMA)


def register_norm_bundle(
    conn: sqlite3.Connection,
    *,
    norm_id: str,
    roi_parquet: Path,
    network_parquet: Path,
    meta: dict,
) -> None:
    ensure_neuro_schema(conn)
    conn.execute(
        """
        INSERT OR REPLACE INTO norm_bundle (norm_id, path_roi_parquet, path_network_parquet, meta_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            norm_id,
            str(roi_parquet.resolve()),
            str(network_parquet.resolve()),
            json.dumps(meta),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def clear_session_neuro_rows(conn: sqlite3.Connection, session_id: str, norm_id: str) -> None:
    conn.execute("DELETE FROM session_roi_timeseries WHERE session_id = ? AND norm_id = ?", (session_id, norm_id))
    conn.execute(
        "DELETE FROM session_network_timeseries WHERE session_id = ? AND norm_id = ?",
        (session_id, norm_id),
    )
    conn.execute("DELETE FROM threshold_hit WHERE session_id = ?", (session_id,))


def insert_roi_timeseries_batch(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    norm_id: str,
    parcel_ts: np.ndarray,
    z_parcel: np.ndarray,
    parcel_ids: np.ndarray,
) -> None:
    T, P = parcel_ts.shape
    rows = []
    for t in range(T):
        for j in range(P):
            rows.append(
                (
                    session_id,
                    t,
                    int(parcel_ids[j]),
                    float(parcel_ts[t, j]),
                    float(z_parcel[t, j]),
                    norm_id,
                )
            )
    conn.executemany(
        """
        INSERT OR REPLACE INTO session_roi_timeseries (
            session_id, t_idx, parcel_id, value, z_vs_norm, norm_id
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_network_timeseries_batch(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    norm_id: str,
    net_ts: np.ndarray,
    z_net: np.ndarray,
    network_ids: list[int],
) -> None:
    T, N = net_ts.shape
    rows = []
    for t in range(T):
        for j in range(N):
            rows.append(
                (
                    session_id,
                    t,
                    int(network_ids[j]),
                    float(net_ts[t, j]),
                    float(z_net[t, j]),
                    norm_id,
                )
            )
    conn.executemany(
        """
        INSERT OR REPLACE INTO session_network_timeseries (
            session_id, t_idx, yeo_network_id, value, z_vs_norm, norm_id
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_threshold_hits(conn: sqlite3.Connection, hits: list[dict]) -> None:
    for h in hits:
        conn.execute(
            """
            INSERT INTO threshold_hit (
                session_id, rule_id, t_start, t_end, networks_json, evidence_json,
                confidence, insight_key, insight_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                h["session_id"],
                h["rule_id"],
                h["t_start"],
                h["t_end"],
                h["networks_json"],
                h["evidence_json"],
                h["confidence"],
                h["insight_key"],
                h["insight_text"],
            ),
        )
