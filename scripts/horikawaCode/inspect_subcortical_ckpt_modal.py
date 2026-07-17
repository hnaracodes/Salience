"""Inspect tribev2-subcortical checkpoint on Modal (needs .venv311 + torch on worker only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run" / "subcortical_ckpt_inspect.json"


def main() -> None:
    import modal

    from tribe import TribeInference, app

    with modal.enable_output(), app.run():
        inference = TribeInference()
        report = inference.inspect_subcortical_checkpoint.remote()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
