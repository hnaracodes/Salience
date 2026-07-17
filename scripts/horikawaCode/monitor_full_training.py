"""Poll Horikawa full training log every N minutes for errors."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run" / "full_training.log"
MONITOR_LOG = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run" / "full_training_monitor_internal.log"
BATCH = PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal" / "batch_report.json"
STATE = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run" / "full_training_state.json"

ERROR_PAT = re.compile(
    r"(Traceback|ERROR|STOP:|status.*error|RuntimeError|KeyError|FileNotFoundError|function .* is stopped)",
    re.I,
)


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _tail(path: Path, n: int = 15) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def _batch_stats() -> str:
    if not BATCH.is_file():
        return "batch_report: missing"
    data = json.loads(BATCH.read_text(encoding="utf-8"))
    rows = data.get("results") or []
    ok = sum(1 for r in rows if r.get("status") == "ok")
    skipped = sum(1 for r in rows if r.get("status") == "skipped")
    err = sum(1 for r in rows if r.get("status") == "error")
    return f"batch_report: ok={ok} skipped={skipped} errors={err} total={len(rows)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-min", type=float, default=5.0)
    parser.add_argument("--max-hours", type=float, default=72.0)
    args = parser.parse_args()

    deadline = time.time() + args.max_hours * 3600
    interval_s = max(60.0, args.interval_min * 60.0)
    last_size = 0

    _log(f"Monitor start interval={args.interval_min}min log={LOG}")

    while time.time() < deadline:
        if STATE.is_file():
            state = json.loads(STATE.read_text(encoding="utf-8"))
            phase = state.get("phase")
            if phase in ("complete", "modal_incomplete"):
                _log(f"Training state terminal: {state}")
                break

        if LOG.is_file():
            size = LOG.stat().st_size
            if size != last_size:
                tail = _tail(LOG, 20)
                hits = [ln for ln in tail if ERROR_PAT.search(ln)]
                _log(_batch_stats())
                if hits:
                    _log("RECENT ERRORS/WARNINGS:")
                    for ln in hits[-5:]:
                        _log(f"  {ln}")
                else:
                    _log(f"heartbeat ok (log {size} bytes)")
                last_size = size

        time.sleep(interval_s)

    _log("Monitor exit")


if __name__ == "__main__":
    main()
