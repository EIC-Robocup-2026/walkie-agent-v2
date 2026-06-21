"""On-robot arena-entry test: ask for a closed door, then drive to the
instruction point.

Exercises exactly what the GPSR run does on entry (subtasks.GoToInstructionPoint):
when the arena door may be shut, ask a human to open it (request_open_door —
depth self-watches open/closed and walks in when it reads clear, falls back to a
spoken confirmation), then navigate to GPSR_INSTRUCTION_POINT_POSE via
go_to_through_door — so a *partly-open* door (reads "open" but is too narrow to
fit, which the depth check can't catch at distance) still triggers a "please open
it wider" + retry on the nav block. Use it to rehearse the arena entry in
isolation, without running a whole GPSR command.

Needs the robot reachable (set WALKIE_ROBOT_IP from a dev PC; 127.0.0.1 on-robot)
and walkie-ai-server up for STT/TTS. No LLM/API key needed — entry uses neither.

    uv run python -m tasks.GPSR.tools.arena_entry                 # request door, then go to instruction point
    DISABLE_LISTENING=1 uv run python -m tasks.GPSR.tools.arena_entry   # type "open" instead of speaking
    uv run python -m tasks.GPSR.tools.arena_entry --through-door  # try to drive first, ask only if BLOCKED
    uv run python -m tasks.GPSR.tools.arena_entry --no-door       # skip the door step (plain nav baseline)
    # Enter the arena -> stop at the instruction point -> (Enter) -> drive to the living
    # room, where a partition blocks nav and gets a "please open it" + retry:
    uv run python -m tasks.GPSR.tools.arena_entry --goto-room living_room

Reads GPSR_INSTRUCTION_POINT_POSE, the world.toml room poses, and the WALKIE_DOOR_*
thresholds from config, so it behaves identically to the real run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from client import WalkieAIClient
from tasks.base import TaskContext
from tasks.common import initialize_robot, load_task_config
from tasks.skills import go_to_through_door, request_open_door

GPSR_DIR = Path(__file__).resolve().parents[1]  # tasks/GPSR — holds config.toml


def _pose(env_key: str, default: str = "0.0,0.0,0.0") -> tuple[float, float, float]:
    """Parse 'x,y,heading_rad' from the named env var (same as subtasks._pose)."""
    parts = [p.strip() for p in os.getenv(env_key, default).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{env_key}: expected 'x,y,heading_rad', got {parts!r}")
    x, y, h = (float(p) for p in parts)
    return x, y, h


def _wait_enter(prompt: str) -> None:
    """Block until the operator presses Enter; continue immediately if there's no TTY."""
    try:
        input(prompt)
    except EOFError:
        print("(no interactive TTY — continuing without a pause)")


def main() -> None:
    load_dotenv()
    load_task_config(GPSR_DIR)  # GPSR config.toml then root — fills the door knobs

    ap = argparse.ArgumentParser(description="Test closed-door arena entry to the instruction point or a room.")
    ap.add_argument("--through-door", action="store_true",
                    help="drive first and ask for the door only on a nav FAILURE "
                         "(go_to_through_door) instead of asking up front")
    ap.add_argument("--no-door", action="store_true",
                    help="skip the door request entirely — plain nav, as a baseline")
    ap.add_argument("--goto-room", metavar="NAME", default=None,
                    help="after stopping at the instruction point, continue to this world.toml "
                         "room/location (e.g. living_room) — a partition/screen blocking the route "
                         "reads 'open' on depth but blocks nav, so it gets a 'please open it' + retry")
    args = ap.parse_args()

    disable_listening = os.getenv("DISABLE_LISTENING", "0").lower() in ("1", "true", "yes")

    # Leg 1 always targets the instruction point (the arena-entry rehearsal).
    ix, iy, ih = _pose("GPSR_INSTRUCTION_POINT_POSE")
    # Leg 2 (optional): continue to a named room after stopping at the instruction point.
    room_pose: tuple[float, float, float] | None = None
    if args.goto_room:
        from tasks.GPSR.world import load_world
        world = load_world()
        room_pose = world.location_pose(args.goto_room)
        if room_pose is None:
            raise SystemExit(f"--goto-room: no pose for {args.goto_room!r} in the world file "
                             f"(known rooms: {', '.join(world.rooms)})")

    print("=== GPSR arena-entry test ===")
    print(f"instruction point : ({ix:.3f}, {iy:.3f}, {ih:.4f} rad)")
    if room_pose is not None:
        rx, ry, rh = room_pose
        print(f"then room {args.goto_room:<7}: ({rx:.3f}, {ry:.3f}, {rh:.4f} rad)")
    print(f"GPSR_REQUEST_DOOR : {os.getenv('GPSR_REQUEST_DOOR', '0')}  "
          f"(this test forces the door step on unless --no-door)")
    print(f"door thresholds   : CLEAR_M={os.getenv('WALKIE_DOOR_CLEAR_M', '1.2')}  "
          f"CENTER_FRAC={os.getenv('WALKIE_DOOR_CENTER_FRAC', '0.4')}  "
          f"MIN_VALID_FRAC={os.getenv('WALKIE_DOOR_MIN_VALID_FRAC', '0.5')}")
    print(f"listening         : {'OFF (type replies)' if disable_listening else 'ON (mic+STT)'}\n")

    walkie = initialize_robot()
    # The entry flow (door ask + nav) uses neither the LLM nor people memory, so
    # model is None here — keeps the test runnable without an API key.
    ctx = TaskContext(walkie=walkie, walkieAI=WalkieAIClient(),
                      model=None, disable_listening=disable_listening)

    door_ok: bool | None = None
    reached_room: bool | None = None
    try:
        # --- Leg 1: enter the arena and drive to the instruction point ---
        if args.through_door:
            # Drive first; ask for the door only if the route is actually blocked.
            # ask_even_if_open mirrors the real entry: a partly-open door reads "open"
            # yet blocks nav, so a block still asks for it to be opened wider.
            print("[step] go_to_through_door: drive, and ask for the door if blocked...")
            reached_ip = go_to_through_door(ctx, ix, iy, ih, ask_even_if_open=True)
        elif args.no_door:
            print("[step] --no-door: skipping the door request; driving straight in...")
            reached_ip = ctx.goto(ix, iy, ih)
        else:
            # Mirror GoToInstructionPoint exactly: ask for the (possibly closed) door
            # first, then drive with go_to_through_door so a *partly-open* door (reads
            # "open" but blocks nav — e.g. the depth check can't see it's too narrow at
            # this standoff) still gets a "please open it wider" + retry.
            print("[step] request_open_door: checking the door / asking for help...")
            door_ok = request_open_door(ctx)
            print(f"[step] door {'confirmed OPEN' if door_ok else 'assumed open after waiting'}; "
                  f"driving in (re-asking to open wider if a partly-open door blocks nav)...")
            reached_ip = go_to_through_door(ctx, ix, iy, ih, ask_even_if_open=True, door_attempts=3)

        # --- Stop at the instruction point, then Leg 2: drive to the room ---
        if room_pose is not None and reached_ip:
            ctx.say("I have reached the instruction point.")
            print(f"\n[stop] reached the instruction point — stopped.")
            _wait_enter(f"[step] press Enter to head to {args.goto_room!r} "
                        f"(set up the partition first if needed)... ")
            rx, ry, rh = room_pose
            print(f"[step] go_to_through_door -> {args.goto_room!r}: driving; if a partition "
                  f"blocks nav it will ask to open it, then retry...")
            reached_room = go_to_through_door(ctx, rx, ry, rh, ask_even_if_open=True, door_attempts=3)
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
        return
    finally:
        walkie.close()

    print("\n=== result ===")
    if door_ok is not None:
        print(f"door              : {'OPEN (confirmed)' if door_ok else 'proceeded after timeout'}")
    print(f"instruction point : {'REACHED ✅' if reached_ip else 'NOT reached ❌'}")
    if reached_room is not None:
        print(f"room {args.goto_room:<12}: {'REACHED ✅' if reached_room else 'NOT reached ❌'}")
    elif room_pose is not None and not reached_ip:
        print(f"room {args.goto_room:<12}: SKIPPED (never reached the instruction point)")
    if not reached_ip or reached_room is False:
        print("  Nav did not reach a goal. If a door/partition is still in the way, check that "
              "the depth door-check (WALKIE_DOOR_*) is calibrated, or try --through-door so it "
              "re-asks on the block. Otherwise verify the destination pose / map.")


if __name__ == "__main__":
    main()
