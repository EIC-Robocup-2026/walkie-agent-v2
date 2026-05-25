"""Reset the CLIP scene store, then run a standalone collection ("explore") loop.

This is the data-collection counterpart to ``main.py`` for the CLIP scene
memory (``chroma_db_scene``). It wipes the store + archived frames so every
run starts clean, then drives the same background perception loop the robot
uses in its ready stage — detect → lift to 3D → CLIP-embed → upsert — while
*you* drive the robot around. No agent, no mic, no perception.json: pure
catalogue building.

    uv run python -m tools.scene_explore          # prompt before wiping, then collect
    uv run python -m tools.scene_explore -y       # skip the wipe confirmation
    uv run python -m tools.scene_explore --keep    # don't wipe; append to the store
    uv run python -m tools.scene_explore --reset-only   # just wipe and exit

Pruning is intentionally OFF here: an explore run accumulates everything it
sees. Eviction belongs to the ready stage (see SCENE_PRUNE_* in .env), not to
catalogue building.

Inspect the result with the viewer (reads the same dir, read-only):

    uv run python -m tools.chroma_viewer
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# Reuse main.py's robot + scene-store wiring so this script tracks any changes
# to construction (image-embed probe, frame-refresh flag, position source, …).
from main import build_scene_store, get_robot
from agents.core.robot_context import RobotContext
from client import WalkieAIClient
from interfaces.walkie_interface import WalkieInterface
from perception import RobotPoseLifter
from services import ScenePerceptionService
from walkie_config import load_config


def _reset(scene_dir: str, frames_dir: str, *, assume_yes: bool) -> None:
    """Delete the scene Chroma dir and archived frames for a clean slate."""
    targets = [Path(scene_dir), Path(frames_dir)]
    existing = [p for p in targets if p.exists()]
    if not existing:
        print(f"[reset] nothing to delete ({scene_dir}, {frames_dir} absent)")
        return
    print("[reset] about to DELETE:")
    for p in existing:
        print(f"        - {p.resolve()}")
    if not assume_yes:
        ans = input("[reset] continue? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("[reset] aborted.")
            sys.exit(1)
    for p in existing:
        shutil.rmtree(p)
    print("[reset] done.")


def main() -> None:
    load_dotenv()
    load_config()  # config.toml tuning defaults (scene interval, dedup, …)
    ap = argparse.ArgumentParser(
        description="Reset the CLIP scene store, then run an explore/collection loop."
    )
    ap.add_argument(
        "-y", "--yes", action="store_true", help="skip the delete confirmation"
    )
    ap.add_argument(
        "--keep",
        action="store_true",
        help="don't wipe the store; append to whatever's already there",
    )
    ap.add_argument(
        "--reset-only",
        action="store_true",
        help="wipe the store and exit without collecting",
    )
    args = ap.parse_args()

    scene_dir = os.getenv("SCENE_CHROMA_DIR", "chroma_db_scene")
    frames_dir = os.getenv("SCENE_FRAMES_DIR", "frames")

    if not args.keep:
        _reset(scene_dir, frames_dir, assume_yes=args.yes)
    if args.reset_only:
        return

    # Match main.py's logging: quiet third-party libs, surface perception.* INFO
    # (tick lines + scene.dedup INSERT/UPDATE decisions).
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("perception").setLevel(
        os.getenv("WALKIE_LOG_LEVEL", "INFO").upper()
    )

    walkieAI = WalkieAIClient(
        base_url=os.getenv("WALKIE_AI_BASE_URL", "http://localhost:5000"),
    )
    walkie = WalkieInterface(get_robot())
    # Some shared reads expect the context to exist even though we run no agents.
    RobotContext.init(perception_path=os.getenv("PERCEPTION_PATH", "perception.json"))

    store, embedder = build_scene_store(walkieAI)
    if store is None or embedder is None:
        print(
            "[scene] scene perception unavailable (see message above) — nothing to "
            "collect into. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    pos_source = os.getenv("SCENE_POSITION_SOURCE", "lift").lower()
    lifter = RobotPoseLifter(walkie.status) if pos_source == "robot" else None
    print(f"[scene] position source: {pos_source}")

    # No prune args → eviction stays off for the whole collection run.
    service = ScenePerceptionService(
        walkieAI,
        walkie,
        store,
        embedder,
        lifter=lifter,
        interval=float(os.getenv("SCENE_PERCEPTION_INTERVAL_SEC", "2.0")),
        min_confidence=float(os.getenv("SCENE_MIN_CONF", "0.0")),
        caption_per_object=os.getenv("SCENE_CAPTION_PER_OBJECT", "0").lower()
        in ("1", "true", "yes"),
        # Keep moving classes (people) out of the catalogue — same as main.py's
        # ready stage. Read from env so config.toml's SCENE_EXCLUDE_CLASSES wins.
        exclude_classes=[
            c.strip()
            for c in os.getenv("SCENE_EXCLUDE_CLASSES", "person").split(",")
            if c.strip()
        ],
    )
    service.start()
    try:
        print("[explore] Drive the robot around. Press Enter when done.")
        input()
    except KeyboardInterrupt:
        print("\n[explore] interrupt — stopping.")
    finally:
        service.stop_and_join(timeout=5)
    print(f"[explore] scene store now holds {store.count} record(s) in {scene_dir}/.")
    print("[explore] inspect it with:  uv run python -m tools.chroma_viewer")


if __name__ == "__main__":
    main()
