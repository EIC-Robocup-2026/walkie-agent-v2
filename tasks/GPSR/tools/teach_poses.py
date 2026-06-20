"""Teach the arena poses into world.toml by driving the robot to each place.

The ~2-hour pre-competition window: instead of hand-reading ~15 map coordinates
and typing them (slow, and a wrong heading sends the robot the wrong way), drive
the robot to each room/placement and press Enter — the tool reads
``walkie.status.get_position()`` and writes the pose back into the world file,
preserving every comment/alias/field on the line. Progress is saved after each
capture, so an interruption never loses what you already taught.

    uv run python -m tasks.GPSR.tools.teach_poses          # only un-surveyed ([0,0,0]) places
    uv run python -m tasks.GPSR.tools.teach_poses --all    # re-survey everything
    uv run python -m tasks.GPSR.tools.teach_poses --file my_arena.toml

The pose WRITER (`set_pose`) is pure and offline-tested; reading the robot pose +
driving are robot-side (this entrypoint imports the hardware stack lazily, so the
writer stays importable on a GPU-less dev box).
"""

from __future__ import annotations

import re

_ZERO = (0.0, 0.0, 0.0)


def _pose_literal(pose: tuple[float, float, float]) -> str:
    x, y, h = pose
    return f"[{x:.3f}, {y:.3f}, {h:.4f}]"


def set_pose(text: str, section: str, key: str, pose: tuple[float, float, float]) -> tuple[str, bool]:
    """Replace the ``pose = [...]`` of *key* inside ``[section]`` in TOML *text*.

    Rewrites only that one inline-table's pose literal, preserving the rest of the
    line (``room``/``placement``/``category``/``aliases``) and the whole file
    (comments, ordering, other tables). Section-scoped, so a location named the
    same as a room is not touched. Returns ``(new_text, replaced)``; *replaced* is
    False (and the text unchanged) when no matching ``key`` line with a ``pose``
    exists in that section.
    """
    literal = _pose_literal(pose)
    out: list[str] = []
    current: str | None = None
    replaced = False
    for line in text.splitlines(keepends=True):
        header = re.match(r"^\[([^\[\]]+)\]$", line.strip())
        if header:
            current = header.group(1).strip()
        elif current == section and not replaced and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            line, n = re.subn(r"pose\s*=\s*\[[^\]]*\]", f"pose = {literal}", line)
            replaced = replaced or bool(n)
        out.append(line)
    return "".join(out), replaced


# --- robot-side entrypoint (lazy heavy imports keep set_pose importable) ------

def _read_pose(walkie) -> tuple[float, float, float]:
    """The robot's current map pose (x, y, heading_rad); zeros on failure."""
    try:
        p = walkie.status.get_position()
        if p:
            return (float(p["x"]), float(p["y"]), float(p["heading"]))
    except Exception as exc:
        print(f"  pose read failed ({exc})")
    return _ZERO


def _capture(path, text: str, section: str, key: str, walkie) -> str:
    pose = _read_pose(walkie)
    if pose == _ZERO:
        # The origin IS the "not surveyed" sentinel (load_world / --all treat it so),
        # and an odometry no-fix reads as (0,0,0). Refuse to claim a capture that
        # would write the un-surveyed marker — the operator must see this and retry.
        print(f"  !! pose read returned the origin (odometry fix?) — leaving "
              f"{section}.{key} un-surveyed; drive there and retry")
        return text
    new_text, ok = set_pose(text, section, key, pose)
    if ok:
        path.write_text(new_text)  # persist after every capture
        print(f"  captured {section}.{key} -> {_pose_literal(pose)}")
        return new_text
    print(f"  !! no 'pose = [...]' line for {section}.{key}; skipped")
    return text


def main() -> None:
    import argparse
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    from ...common import initialize_robot, load_task_config  # type: ignore  # heavy: torch
    from ..world import load_world

    load_task_config(Path(__file__).resolve().parents[1])

    ap = argparse.ArgumentParser(description="Teach arena poses into the GPSR world file by driving the robot.")
    ap.add_argument("--file", help="world TOML to edit (default: $GPSR_WORLD_FILE or tasks/GPSR/world.toml)")
    ap.add_argument("--all", action="store_true", help="re-survey poses already set (default: only [0,0,0])")
    args = ap.parse_args()

    from .. import world as world_mod
    path = Path(args.file or os.getenv("GPSR_WORLD_FILE") or Path(world_mod.__file__).with_name("world.toml"))
    if not path.exists():
        raise SystemExit(f"world file not found: {path}")
    world = load_world(path)
    text = path.read_text()

    walkie = initialize_robot()
    print(f"Teaching poses into {path}. At each prompt, drive the robot to the place, then Enter.")
    try:
        for section, items in (("rooms", world.rooms), ("locations", world.locations)):
            for key, rec in items.items():
                if not args.all and rec.pose != _ZERO:
                    print(f"[skip] {section}.{key} already at {_pose_literal(rec.pose)}")
                    continue
                ans = input(f"\nDrive to '{key}' ({section}) — Enter=capture, s=skip, q=quit: ").strip().lower()
                if ans == "q":
                    print("Stopping. Captured poses are saved.")
                    return
                if ans == "s":
                    continue
                text = _capture(path, text, section, key, walkie)

        ans = input("\nCapture the INSTRUCTION POINT at the robot's spot now? Enter=yes, s=skip: ").strip().lower()
        if ans != "s":
            x, y, h = _read_pose(walkie)
            print(
                "\nSet this in tasks/GPSR/config.toml (it lives there, not in world.toml):\n"
                f'  GPSR_INSTRUCTION_POINT_POSE = "{x:.3f},{y:.3f},{h:.4f}"'
            )
        print(f"\nDone. {path} updated.")
    finally:
        try:
            walkie.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
