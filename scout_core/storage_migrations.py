from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _portable_storage_path(path: Path) -> str:
    """Store repo-relative POSIX paths so SQLite works across host OS and containers."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _canonical_scout_norms_path(stored: str) -> Path | None:
    """Map legacy absolute or backslash paths to scout_norms/<norm_id>/<file>."""
    normalized = stored.replace("\\", "/")
    marker = "scout_norms/"
    if marker not in normalized:
        return None
    suffix = normalized.split(marker, 1)[1]
    if not suffix or "/" not in suffix:
        return None
    return PROJECT_ROOT / "scout_norms" / suffix


def resolve_storage_path(stored: str) -> Path:
    """Resolve a path from SQLite — supports repo-relative and legacy absolute entries."""
    normalized = stored.replace("\\", "/")
    candidates: list[Path] = []
    raw = Path(normalized)
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(PROJECT_ROOT / normalized)
    canonical = _canonical_scout_norms_path(stored)
    if canonical is not None:
        candidates.append(canonical)
    if raw.name and raw.parent.name:
        candidates.append(PROJECT_ROOT / "scout_norms" / raw.parent.name / raw.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


KRAGEL_EMOTION_COLUMNS = [
    "contentment", "amusement", "surprise", "fear", "anger", "sadness", "neutral",
]
LEGACY_EMOTION_COLUMNS = [
    "anger", "disgust", "fear", "happy", "neutral", "sad", "negative_affect",
]
KNOWN_EMOTION_COLUMNS = list(dict.fromkeys(KRAGEL_EMOTION_COLUMNS + LEGACY_EMOTION_COLUMNS))

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

DUAL_TRACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_dual_track_meta (
    session_id          TEXT PRIMARY KEY,
    baseline_session_id TEXT,
    baseline_trs        INTEGER,
    baseline_flag       TEXT,
    template_source     TEXT,
    thresholds_json     TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS session_engagement_trace (
    session_id        TEXT NOT NULL,
    t_idx             INTEGER NOT NULL,
    engagement_score  REAL,
    label             TEXT,
    PRIMARY KEY (session_id, t_idx),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS session_emotion_trace (
    session_id      TEXT NOT NULL,
    t_idx           INTEGER NOT NULL,
    contentment     REAL NOT NULL,
    amusement       REAL NOT NULL,
    surprise        REAL NOT NULL,
    fear            REAL NOT NULL,
    anger           REAL NOT NULL,
    sadness         REAL NOT NULL,
    neutral         REAL NOT NULL,
    PRIMARY KEY (session_id, t_idx),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_engagement ON session_engagement_trace(session_id);
CREATE INDEX IF NOT EXISTS idx_emotion ON session_emotion_trace(session_id);
"""


def ensure_neuro_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(NEURO_SCHEMA)


def ensure_dual_track_schema(conn: sqlite3.Connection) -> None:
    """Create dual-track tables if they don't exist. Safe to call multiple times."""
    conn.executescript(DUAL_TRACK_SCHEMA)
    _ensure_emotion_trace_columns(conn)
    _ensure_emotion_decoder_trace_table(conn)


def _ensure_emotion_trace_columns(conn: sqlite3.Connection) -> None:
    """Add Kragel columns to older wide emotion tables without dropping rows."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(session_emotion_trace)").fetchall()}
    for col in KRAGEL_EMOTION_COLUMNS:
        if col not in cols:
            conn.execute(f"ALTER TABLE session_emotion_trace ADD COLUMN {col} REAL")
            cols.add(col)


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
            _portable_storage_path(roi_parquet),
            _portable_storage_path(network_parquet),
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


# ---------------------------------------------------------------------------
# Dual-track insert helpers
# ---------------------------------------------------------------------------

def upsert_dual_track_meta(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    baseline_session_id: str | None,
    baseline_trs: int | None,
    baseline_flag: str | None,
    template_source: str | None,
    thresholds_json: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO session_dual_track_meta (
            session_id, baseline_session_id, baseline_trs, baseline_flag,
            template_source, thresholds_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            baseline_session_id,
            baseline_trs,
            baseline_flag,
            template_source,
            thresholds_json,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def insert_engagement_trace(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    scores: list,
    labels: list,
) -> None:
    rows = [
        (session_id, t, float(s) if s is not None else None, lbl)
        for t, (s, lbl) in enumerate(zip(scores, labels))
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO session_engagement_trace (session_id, t_idx, engagement_score, label)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )


def insert_emotion_decoder_trace(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    emotion_result: dict[str, Any],
) -> None:
    """Persist decoder probabilities as JSON (session_emotion_decoder_trace)."""
    _ensure_emotion_decoder_trace_table(conn)
    probabilities = emotion_result.get("probabilities") or []
    class_names = emotion_result.get("class_names") or []
    rows = []
    for t, prob_row in enumerate(probabilities):
        payload = {
            "class_names": class_names,
            "probabilities": prob_row,
            "scores": (emotion_result.get("scores") or [None])[t]
            if t < len(emotion_result.get("scores") or [])
            else prob_row,
            "model_id": emotion_result.get("model_id"),
        }
        rows.append((session_id, t, json.dumps(payload)))
    conn.executemany(
        """
        INSERT OR REPLACE INTO session_emotion_decoder_trace (session_id, t_idx, trace_json)
        VALUES (?, ?, ?)
        """,
        rows,
    )


def _ensure_emotion_decoder_trace_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_emotion_decoder_trace (
            session_id TEXT NOT NULL,
            t_idx INTEGER NOT NULL,
            trace_json TEXT NOT NULL,
            PRIMARY KEY (session_id, t_idx),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_emotion_decoder ON session_emotion_decoder_trace(session_id)"
    )


def insert_emotion_trace(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    cosine_scores: list,
    template_names: list[str],
) -> None:
    name_to_idx = {n: i for i, n in enumerate(template_names)}

    table_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(session_emotion_trace)").fetchall()
    }
    emotion_cols = [c for c in KNOWN_EMOTION_COLUMNS if c in table_cols]
    if not emotion_cols:
        raise ValueError("session_emotion_trace has no recognised emotion columns")

    def _value_for(column: str, row: list) -> float:
        if column in name_to_idx:
            return float(row[name_to_idx[column]])
        if column == "sad" and "sadness" in name_to_idx:
            return float(row[name_to_idx["sadness"]])
        return 0.0

    rows = []
    for t, row in enumerate(cosine_scores):
        vals = [_value_for(col, row) for col in emotion_cols]
        rows.append((session_id, t, *vals))

    col_sql = ", ".join(emotion_cols)
    placeholders = ", ".join("?" for _ in range(2 + len(emotion_cols)))
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO session_emotion_trace (
            session_id, t_idx, {col_sql}
        ) VALUES ({placeholders})
        """,
        rows,
    )
