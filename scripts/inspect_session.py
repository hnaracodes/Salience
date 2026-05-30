"""Print sample cortical activation data from preds.npz and activations.sqlite."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "scout_data"
DB_PATH = DATA_DIR / "activations.sqlite"
SESSIONS_DIR = DATA_DIR / "sessions"


def _latest_session_id() -> str:
    dirs = [p for p in SESSIONS_DIR.iterdir() if p.is_dir() and (p / "preds.npz").is_file()]
    if not dirs:
        raise SystemExit("No sessions with preds.npz under scout_data/sessions/")
    return max(dirs, key=lambda p: (p / "preds.npz").stat().st_mtime).name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    session_id = args.session_id or _latest_session_id()
    npz_path = SESSIONS_DIR / session_id / "preds.npz"
    preds = np.load(npz_path)["preds"]
    n_t, n_v = preds.shape
    t = min(max(args.timestep, 0), n_t - 1)

    print(f"session_id: {session_id}")
    print(f"preds.npz: {npz_path}")
    print(f"shape: T={n_t}, V={n_v} (timesteps × vertices)")
    print(f"dtype: {preds.dtype}")
    print(f"global min/max: {float(preds.min()):.6f} / {float(preds.max()):.6f}")
    print()

    row = preds[t]
    top_idx = np.argsort(row)[-args.top :][::-1]
    print(f"Top {args.top} vertices at t_idx={t}:")
    for rank, vidx in enumerate(top_idx, start=1):
        print(f"  rank {rank:2d}  vertex {int(vidx):5d}  activation {float(row[vidx]):.6f}")

    if not DB_PATH.is_file():
        print(f"\nSQLite not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    sess = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if sess:
        print("\nsessions row:")
        for k in sess.keys():
            print(f"  {k}: {sess[k]}")
    summary = conn.execute(
        "SELECT * FROM timestep_summary WHERE session_id = ? AND t_idx = ?",
        (session_id, t),
    ).fetchone()
    if summary:
        print(f"\ntimestep_summary (t_idx={t}):")
        for k in summary.keys():
            print(f"  {k}: {summary[k]}")
    peaks = conn.execute(
        """
        SELECT rank, vertex_index, activation, brain_sector, interpretation_summary
        FROM activation_peak WHERE session_id = ? AND t_idx = ?
        ORDER BY rank LIMIT ?
        """,
        (session_id, t, args.top),
    ).fetchall()
    if peaks:
        print(f"\nactivation_peak top rows (t_idx={t}):")
        for p in peaks:
            print(
                f"  rank {p['rank']}  v={p['vertex_index']}  val={p['activation']:.6f}  "
                f"sector={p['brain_sector']!r}  {p['interpretation_summary']}"
            )
    conn.close()


if __name__ == "__main__":
    main()
