"""One-shot migration: move flat data/raw/frames/{image_name}{frame_id}.jpg
into per-session subdirs data/raw/frames/{session_stem}/{frame_id}.jpg.

Run from the repository root:
    python scripts/migrate_frames.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = ROOT / "data/raw/sessions"
FRAMES_DIR = ROOT / "data/raw/frames"


def migrate() -> None:
    session_files = sorted(SESSIONS_DIR.glob("*.json"))
    assert session_files, f"No session files found in {SESSIONS_DIR}"

    moved_total = 0
    missing_total = 0

    for session_path in session_files:
        data = json.loads(session_path.read_text())
        image_name: str = data["image_name"]
        total_frames: int = data["total_frames"]
        session_stem: str = session_path.stem

        dest_dir = FRAMES_DIR / session_stem
        dest_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        missing = 0
        for frame_id in range(total_frames):
            src = FRAMES_DIR / f"{image_name}{frame_id}.jpg"
            dst = dest_dir / f"{frame_id}.jpg"
            if src.exists():
                src.rename(dst)
                moved += 1
            else:
                missing += 1

        print(f"{session_stem}: moved={moved} missing={missing}")
        moved_total += moved
        missing_total += missing

    print(f"\nDone. moved={moved_total} missing={missing_total}")

    leftover = [p for p in FRAMES_DIR.iterdir() if p.is_file() and p.suffix == ".jpg"]
    if leftover:
        print(f"WARNING: {len(leftover)} loose .jpg files remain in {FRAMES_DIR}")
        for p in leftover[:10]:
            print(f"  {p.name}")


if __name__ == "__main__":
    sys.exit(migrate())
