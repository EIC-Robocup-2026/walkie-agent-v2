"""Wipe Walkie's vector DBs for a clean slate.

Two stores accumulate over a robot's life and sometimes you just want to start
fresh — the legacy explore-stage object DB and the CLIP scene memory. This is
the one place to clear either (or both), with their archived frame folders.

    uv run python -m tools.reset_db --object   # legacy object DB (chroma_db) + object_frames
    uv run python -m tools.reset_db --scene    # CLIP scene memory (chroma_db_scene) + frames
    uv run python -m tools.reset_db --all      # both
    uv run python -m tools.reset_db --object -y  # skip the confirmation prompt

Paths come from config (CHROMA_DIR / OBJECT_FRAMES_DIR / SCENE_CHROMA_DIR /
SCENE_FRAMES_DIR), so this tracks whatever main.py uses. It only deletes those
directories — run it while the robot / viewer are stopped so nothing has the
SQLite files open.

(``tools/scene_explore --reset-only`` also wipes just the scene store; this
tool adds the object store and an --all sweep.)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from walkie_config import load_config


def _wipe(label: str, dirs: list[str], *, assume_yes: bool) -> None:
    """Delete ``dirs`` (a logical store) after one confirmation."""
    existing = [Path(d) for d in dirs if Path(d).exists()]
    if not existing:
        print(f"[reset:{label}] nothing to delete ({', '.join(dirs)} absent)")
        return
    print(f"[reset:{label}] about to DELETE:")
    for p in existing:
        print(f"            - {p.resolve()}")
    if not assume_yes:
        ans = input(f"[reset:{label}] continue? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print(f"[reset:{label}] aborted.")
            return
    for p in existing:
        shutil.rmtree(p)
    print(f"[reset:{label}] done.")


def main() -> None:
    load_dotenv()
    load_config()  # resolve the same dir names main.py uses
    ap = argparse.ArgumentParser(
        description="Wipe Walkie's vector DBs (object and/or CLIP scene memory)."
    )
    ap.add_argument(
        "--object", action="store_true",
        help="wipe the legacy object DB (CHROMA_DIR) + OBJECT_FRAMES_DIR",
    )
    ap.add_argument(
        "--scene", action="store_true",
        help="wipe the CLIP scene memory (SCENE_CHROMA_DIR) + SCENE_FRAMES_DIR",
    )
    ap.add_argument(
        "--all", action="store_true", help="wipe both stores (object + scene)"
    )
    ap.add_argument(
        "-y", "--yes", action="store_true", help="skip the delete confirmation(s)"
    )
    args = ap.parse_args()

    do_object = args.object or args.all
    do_scene = args.scene or args.all
    if not (do_object or do_scene):
        ap.error("pick what to wipe: --object, --scene, or --all")

    if do_object:
        _wipe(
            "object",
            [
                os.getenv("CHROMA_DIR", "chroma_db"),
                os.getenv("OBJECT_FRAMES_DIR", "object_frames"),
            ],
            assume_yes=args.yes,
        )
    if do_scene:
        _wipe(
            "scene",
            [
                os.getenv("SCENE_CHROMA_DIR", "chroma_db_scene"),
                os.getenv("SCENE_FRAMES_DIR", "frames"),
            ],
            assume_yes=args.yes,
        )
    print("[reset] complete. The stores are recreated empty on the next run.")


if __name__ == "__main__":
    main()
