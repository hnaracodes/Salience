"""One-off helper to rewrite paths after neuroEmo archive move. Not part of runtime."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REPLACEMENTS = [
    ("scout_data/neuroEmoCode", "scout_data/neuroEmoCode"),
    ("scout_data\\neuroemo", "scout_data\\neuroEmoCode"),
    ("scripts/neuroEmoCode/prepare_neuroemo", "scripts/neuroEmoCode/prepare_neuroemo"),
    ("scripts/neuroEmoCode/train_neuroemo", "scripts/neuroEmoCode/train_neuroemo"),
    ("scripts/neuroEmoCode/build_schaefer", "scripts/neuroEmoCode/build_schaefer"),
    ("scripts/neuroEmoCode/verify_tribe_vertex", "scripts/neuroEmoCode/verify_tribe_vertex"),
    ("scripts/neuroEmoCode/verify_schaefer", "scripts/neuroEmoCode/verify_schaefer"),
    ("scripts/neuroEmoCode/run_neuroemo", "scripts/neuroEmoCode/run_neuroemo"),
    ("scripts/neuroEmoCode/run_surface_phase2", "scripts/neuroEmoCode/run_surface_phase2"),
    ("from scout_core.neuroEmoCode.roi_features", "from scout_core.neuroEmoCode.roi_features"),
    ("from scout_core.neuroEmoCode.schaefer_surface_labels", "from scout_core.neuroEmoCode.schaefer_surface_labels"),
    (".cursor/plans/neuroEmoCode/neuroemo", ".cursor/plans/neuroEmoCode/neuroemo"),
]

PROJECT_ROOT_FIX = (
    'PROJECT_ROOT = Path(__file__).resolve().parents[1]',
    'PROJECT_ROOT = Path(__file__).resolve().parents[2]',
)


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if path.parent.name == "neuroEmoCode" and path.suffix == ".py" and path.name != "_archive_path_fix.py":
        text = text.replace(*PROJECT_ROOT_FIX)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    roots = [
        ROOT / "scripts" / "neuroEmoCode",
        ROOT / "scout_core" / "neuroEmoCode",
        ROOT / "tests" / "neuroEmoCode",
    ]
    changed = 0
    for base in roots:
        for path in base.rglob("*"):
            if path.suffix in {".py", ".ps1", ".md"} and path.is_file():
                if patch_file(path):
                    changed += 1
                    print("patched", path.relative_to(ROOT))
    print(f"done ({changed} files)")


if __name__ == "__main__":
    main()
