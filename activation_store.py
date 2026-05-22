"""
Persist Tribe v2 cortical predictions locally.

What gets stored
----------------
- **Full cortical time series**: ``scout_data/sessions/<id>/preds.npz`` → array ``preds``
  with shape **(n_timesteps, n_vertices)** — every vertex, every TRIBE output frame.
- **SQLite** ``scout_data/activations.sqlite``:
  - ``timestep_summary`` — aggregate stats per timestep (all vertices).
  - ``activation_peak`` — **top-K vertices per timestep** (dense export stays in ``.npz``).

Interpretation lookup tables ( qualitative UX layer, not clinical truth )
------------------------------------------------------------------------
- ``stimulation_band``: maps **percentiles** (within timestep / within session) to short labels.
- ``location_coarse_band``: maps **vertex_fraction** ∈ [0, 1] (vertex_index / (V−1)) to coarse text.
  This fraction follows **mesh vertex enumeration**, not true anatomical coordinates unless you
  replace seeds with atlas-derived ranges. For neuroscience-facing mapping, use a proper surface
  atlas (Schaefer→Yeo, etc.) via ``configs/vertex_regions.csv``.

Is this the right approach?
---------------------------
Yes as **engineering scaffolding**: persist arrays + summarize + attach calibrated language.
No if you need **literal emotion decoding**: that requires validated parcellation, norms, and
study-specific thresholds — implement as a later ``scout_core`` analysis layer on top of these rows.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scout_core.storage_migrations import ensure_neuro_schema

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "scout_data"
SESSIONS_DIR = DATA_DIR / "sessions"
DB_PATH = DATA_DIR / "activations.sqlite"

DEFAULT_VERTEX_REGIONS_CSV = PROJECT_ROOT / "configs" / "vertex_regions.csv"
DEFAULT_LOCATION_BANDS_CSV = PROJECT_ROOT / "configs" / "location_coarse_bands.csv"

SCHEMA = """
CREATE TABLE IF NOT EXISTS stimulation_band (
    code TEXT PRIMARY KEY,
    pct_low REAL NOT NULL,
    pct_high REAL NOT NULL,
    label TEXT NOT NULL,
    interpretation TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS location_coarse_band (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    norm_start REAL NOT NULL,
    norm_end REAL NOT NULL,
    label TEXT NOT NULL,
    interpretation TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    CHECK (norm_start >= 0 AND norm_end <= 1.0001 AND norm_start < norm_end)
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source_video TEXT NOT NULL,
    mesh_name TEXT NOT NULL DEFAULT 'fsaverage5',
    n_timesteps INTEGER NOT NULL,
    n_vertices INTEGER NOT NULL,
    preds_npz_path TEXT NOT NULL,
    meta_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS timestep_summary (
    session_id TEXT NOT NULL,
    t_idx INTEGER NOT NULL,
    mean_activation REAL NOT NULL,
    std_activation REAL NOT NULL,
    max_vertex_index INTEGER NOT NULL,
    max_activation REAL NOT NULL,
    mean_timestep_percentile REAL,
    PRIMARY KEY (session_id, t_idx),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS activation_peak (
    session_id TEXT NOT NULL,
    t_idx INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    vertex_index INTEGER NOT NULL,
    activation REAL NOT NULL,
    brain_sector TEXT,
    vertex_fraction REAL,
    timestep_percentile REAL NOT NULL,
    session_percentile REAL NOT NULL,
    stimulation_band_code TEXT NOT NULL,
    location_band_label TEXT NOT NULL,
    interpretation_summary TEXT NOT NULL,
    PRIMARY KEY (session_id, t_idx, rank),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (stimulation_band_code) REFERENCES stimulation_band(code)
);

CREATE INDEX IF NOT EXISTS idx_peaks_session ON activation_peak(session_id);
CREATE INDEX IF NOT EXISTS idx_summary_session ON timestep_summary(session_id);
"""

VIEW_PEAK_INTERPRETED = """
DROP VIEW IF EXISTS v_peak_interpreted;
CREATE VIEW v_peak_interpreted AS
SELECT
    p.session_id,
    p.t_idx,
    p.rank,
    p.vertex_index,
    p.activation,
    p.brain_sector,
    p.vertex_fraction,
    p.timestep_percentile,
    p.session_percentile,
    p.stimulation_band_code,
    s.label AS stimulation_label,
    s.interpretation AS stimulation_interpretation,
    p.location_band_label,
    l.label AS location_band_match,
    l.interpretation AS location_interpretation,
    p.interpretation_summary
FROM activation_peak AS p
JOIN stimulation_band AS s ON s.code = p.stimulation_band_code
JOIN location_coarse_band AS l
    ON p.vertex_fraction >= l.norm_start AND p.vertex_fraction < l.norm_end;
"""


def load_vertex_regions_csv(csv_path: Path) -> dict[int, str]:
    """CSV columns: vertex_index,region_name (header required)."""
    out: dict[int, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "vertex_index" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} needs columns vertex_index,region_name; got {reader.fieldnames!r}"
            )
        region_col = "region_name" if "region_name" in reader.fieldnames else reader.fieldnames[1]
        for row in reader:
            if not row.get("vertex_index"):
                continue
            vid = int(row["vertex_index"])
            out[vid] = str(row[region_col]).strip()
    return out


def _percentile_rank_timestep(row: np.ndarray, value: float) -> float:
    """Empirical CDF percentile in [0,100]: fraction of vertices <= value at this timestep."""
    return float(100.0 * np.mean(row <= value))


def _percentile_rank_session(sorted_flat: np.ndarray, value: float) -> float:
    """Percentile position vs entire session distribution (all timesteps × vertices)."""
    idx = np.searchsorted(sorted_flat, np.float32(value), side="right")
    return float(100.0 * idx / len(sorted_flat))


def _lookup_stimulation_band(conn: sqlite3.Connection, pct: float) -> tuple[str, str, str]:
    row = conn.execute(
        """
        SELECT code, label, interpretation FROM stimulation_band
        WHERE ? >= pct_low AND ? < pct_high
        ORDER BY sort_order LIMIT 1
        """,
        (pct, pct),
    ).fetchone()
    if row:
        return str(row[0]), str(row[1]), str(row[2])
    row = conn.execute(
        "SELECT code, label, interpretation FROM stimulation_band ORDER BY sort_order DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    return str(row[0]), str(row[1]), str(row[2])


def _lookup_location_band(conn: sqlite3.Connection, vertex_fraction: float) -> tuple[str, str]:
    vf = min(1.0, max(0.0, vertex_fraction))
    row = conn.execute(
        """
        SELECT label, interpretation FROM location_coarse_band
        WHERE ? >= norm_start AND ? < norm_end
        ORDER BY sort_order LIMIT 1
        """,
        (vf, vf),
    ).fetchone()
    if row:
        return str(row[0]), str(row[1])
    row = conn.execute(
        "SELECT label, interpretation FROM location_coarse_band ORDER BY sort_order DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    return str(row[0]), str(row[1])


def _seed_stimulation_bands(conn: sqlite3.Connection) -> None:
    seeds = [
        ("S_LO", 0.0, 20.0, "Low relative stimulation", "Within this timestep, activation at this vertex is among the lowest vs other cortical vertices (model output, not fMRI).", 0),
        ("S_BELO", 20.0, 40.0, "Below-average stimulation", "Below the median vertex response for this frame.", 1),
        ("S_MID", 40.0, 60.0, "Moderate stimulation", "Near the middle of the vertex distribution for this frame.", 2),
        ("S_ABOVE", 60.0, 80.0, "Elevated stimulation", "Upper tier relative to other vertices at this timestep.", 3),
        ("S_HI", 80.0, 100.001, "High relative stimulation", "Among the strongest vertex responses this frame — candidate salient cortical drive in the model.", 4),
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO stimulation_band (code, pct_low, pct_high, label, interpretation, sort_order)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        seeds,
    )


def _clear_location_bands(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM location_coarse_band")


def _seed_location_bands_default(conn: sqlite3.Connection) -> None:
    """Placeholder coarse bins on vertex_index fraction — replace via CSV for atlas-aligned bands."""
    seeds = [
        (0.0, 0.34, "Mesh index fraction 0.00–0.33", "Low third of vertex enumeration on fsaverage5 (not an anatomical axis unless you remap via atlas).", 0),
        (0.34, 0.67, "Mesh index fraction 0.34–0.66", "Middle third of vertex enumeration — use atlas CSV for real anatomy.", 1),
        (0.67, 1.0001, "Mesh index fraction 0.67–1.00", "Upper third of vertex enumeration — use atlas CSV for real anatomy.", 2),
    ]
    conn.executemany(
        """
        INSERT INTO location_coarse_band (norm_start, norm_end, label, interpretation, sort_order)
        VALUES (?, ?, ?, ?, ?)
        """,
        seeds,
    )


def load_location_coarse_bands_csv(csv_path: Path, conn: sqlite3.Connection) -> None:
    """CSV columns: norm_start,norm_end,label,interpretation[,sort_order]. Replaces default rows."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"norm_start", "norm_end", "label", "interpretation"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{csv_path} needs columns {required}; got {reader.fieldnames!r}")
        for i, row in enumerate(reader):
            rows.append(
                (
                    float(row["norm_start"]),
                    float(row["norm_end"]),
                    row["label"].strip(),
                    row["interpretation"].strip(),
                    int(row["sort_order"]) if row.get("sort_order") else i,
                )
            )
    _clear_location_bands(conn)
    conn.executemany(
        """
        INSERT INTO location_coarse_band (norm_start, norm_end, label, interpretation, sort_order)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )


def resolve_location_bands(conn: sqlite3.Connection) -> None:
    """Load custom bands from configs if present, else defaults."""
    if DEFAULT_LOCATION_BANDS_CSV.is_file():
        load_location_coarse_bands_csv(DEFAULT_LOCATION_BANDS_CSV, conn)
    elif conn.execute("SELECT COUNT(*) FROM location_coarse_band").fetchone()[0] == 0:
        _seed_location_bands_default(conn)


def _migrate_activation_peak(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(activation_peak)").fetchall()}
    if not cols:
        return
    additions = [
        ("vertex_fraction", "REAL"),
        ("timestep_percentile", "REAL"),
        ("session_percentile", "REAL"),
        ("stimulation_band_code", "TEXT"),
        ("location_band_label", "TEXT"),
        ("interpretation_summary", "TEXT"),
    ]
    for name, decl in additions:
        if name not in cols:
            conn.execute(f"ALTER TABLE activation_peak ADD COLUMN {name} {decl}")


def _migrate_timestep_summary(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(timestep_summary)").fetchall()}
    if "mean_timestep_percentile" not in cols and cols:
        conn.execute("ALTER TABLE timestep_summary ADD COLUMN mean_timestep_percentile REAL")


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.executescript(VIEW_PEAK_INTERPRETED)

    ensure_neuro_schema(conn)
    _migrate_activation_peak(conn)
    _migrate_timestep_summary(conn)
    _seed_stimulation_bands(conn)
    resolve_location_bands(conn)
    conn.commit()
    return conn


def save_cortical_timeseries(
    preds: np.ndarray,
    *,
    session_id: str | None = None,
    source_video: str,
    mesh_name: str = "fsaverage5",
    vertex_to_region: dict[int, str] | None = None,
    top_k_peaks: int = 64,
    notes: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> str:
    """
    Save preds[T, V] to scout_data/sessions/<id>/preds.npz and SQLite summaries/peaks.

    Peak rows include percentile-based stimulation bands and coarse location labels.
    """
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D (T, V), got shape {preds.shape}")

    n_t, n_v = preds.shape
    denom = max(n_v - 1, 1)
    session_sorted = np.sort(preds.ravel())

    session_id = session_id or uuid.uuid4().hex
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    npz_path = session_dir / "preds.npz"
    np.savez_compressed(npz_path, preds=preds, mesh_name=np.array(mesh_name))

    meta = {
        "dtype": str(preds.dtype),
        "shape": [int(n_t), int(n_v)],
        "description": "preds[i,j] = predicted activation at timestep i, fsaverage5 vertex j",
        "interpretation_note": "stimulation_band uses timestep-percentile; location uses vertex_fraction mesh-index proxy unless configs/location_coarse_bands.csv overrides.",
    }
    if extra_meta:
        meta.update(extra_meta)

    conn = _connect()
    try:
        existing = conn.execute(
            "SELECT id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        rel_path = str(npz_path.relative_to(PROJECT_ROOT))
        if existing:
            conn.execute(
                """
                UPDATE sessions SET
                    source_video = ?, mesh_name = ?, n_timesteps = ?, n_vertices = ?,
                    preds_npz_path = ?, meta_json = ?, notes = ?
                WHERE id = ?
                """,
                (
                    source_video,
                    mesh_name,
                    n_t,
                    n_v,
                    rel_path,
                    json.dumps(meta),
                    notes,
                    session_id,
                ),
            )
            conn.execute(
                "DELETE FROM timestep_summary WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM activation_peak WHERE session_id = ?", (session_id,)
            )
        else:
            conn.execute(
                """
                INSERT INTO sessions (
                    id, created_at, source_video, mesh_name,
                    n_timesteps, n_vertices, preds_npz_path, meta_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    datetime.now(timezone.utc).isoformat(),
                    source_video,
                    mesh_name,
                    n_t,
                    n_v,
                    rel_path,
                    json.dumps(meta),
                    notes,
                ),
            )

        k = min(top_k_peaks, n_v)
        for t in range(n_t):
            row = preds[t]
            mean_v = float(row.mean())
            std_v = float(row.std())
            max_idx = int(row.argmax())
            max_val = float(row[max_idx])
            mean_pct = _percentile_rank_timestep(row, mean_v)

            conn.execute(
                """
                INSERT INTO timestep_summary (
                    session_id, t_idx, mean_activation, std_activation,
                    max_vertex_index, max_activation, mean_timestep_percentile
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, t, mean_v, std_v, max_idx, max_val, mean_pct),
            )

            top_idx = np.argpartition(row, -k)[-k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            for rank, vidx in enumerate(top_idx):
                vidx_i = int(vidx)
                val = float(row[vidx_i])
                sector = vertex_to_region.get(vidx_i) if vertex_to_region else None
                v_frac = vidx_i / denom
                t_pct = _percentile_rank_timestep(row, val)
                s_pct = _percentile_rank_session(session_sorted, val)
                stim_code, stim_label, stim_txt = _lookup_stimulation_band(conn, t_pct)
                loc_label, loc_txt = _lookup_location_band(conn, v_frac)
                summary = (
                    f"{stim_label} (timestep pct {t_pct:.1f}); "
                    f"session pct {s_pct:.1f}; {loc_label}"
                    + (f"; atlas={sector}" if sector else "")
                )

                conn.execute(
                    """
                    INSERT INTO activation_peak (
                        session_id, t_idx, rank, vertex_index, activation, brain_sector,
                        vertex_fraction, timestep_percentile, session_percentile,
                        stimulation_band_code, location_band_label, interpretation_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        t,
                        rank,
                        vidx_i,
                        val,
                        sector,
                        v_frac,
                        t_pct,
                        s_pct,
                        stim_code,
                        loc_label,
                        summary,
                    ),
                )

        conn.commit()
    finally:
        conn.close()

    return session_id


def resolve_vertex_regions(
    vertex_regions_csv: Path | None = None,
) -> dict[int, str] | None:
    path = vertex_regions_csv or DEFAULT_VERTEX_REGIONS_CSV
    if not path.is_file():
        return None
    return load_vertex_regions_csv(path)
