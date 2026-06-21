"""Calibrate WALKIE_DOOR_CLEAR_M on the robot.

The closed-door check (tasks.skills.door.door_open_from_depth) calls the doorway
CLOSED when the median depth of the central WALKIE_DOOR_CENTER_FRAC box is below
WALKIE_DOOR_CLEAR_M (and enough of that box has valid depth). This tool measures
that exact median — with the door CLOSED, then OPEN — and recommends a threshold
sitting between the two readings.

Needs the robot reachable (set WALKIE_ROBOT_IP from a dev PC; 127.0.0.1 on-robot).

    uv run python -m manual_tests.calibrate_door            # guided closed/open capture
    uv run python -m manual_tests.calibrate_door --watch    # live continuous readout

Reads the same WALKIE_DOOR_* knobs the skill does, so the OPEN/CLOSED verdict it
prints is exactly what request_open_door / go_to_through_door would decide.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

from dotenv import load_dotenv
from walkie_sdk import WalkieRobot

from walkie_config import load_config

from tasks.skills.door import door_open_from_depth

ZENOH_PORT = 7447


def get_robot() -> WalkieRobot:
    # ROS services go over rosbridge, camera over zenoh (the SDK's default split).
    # 127.0.0.1 is correct on the robot itself; set WALKIE_ROBOT_IP from a dev PC.
    ros_protocol = os.getenv("WALKIE_ROS_PROTOCOL", "rosbridge")
    ros_port = int(os.getenv("WALKIE_ROS_PORT", str(ZENOH_PORT if ros_protocol == "zenoh" else 9090)))
    robot_ip = os.getenv("WALKIE_ROBOT_IP", "127.0.0.1")
    return WalkieRobot(
        ip=robot_ip,
        ros_protocol=ros_protocol,
        ros_port=ros_port,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )


def _knobs() -> tuple[float, float, float]:
    """The live WALKIE_DOOR_* thresholds the skill reads (same defaults)."""
    return (
        float(os.getenv("WALKIE_DOOR_CLEAR_M", "1.2")),
        float(os.getenv("WALKIE_DOOR_CENTER_FRAC", "0.4")),
        float(os.getenv("WALKIE_DOOR_MIN_VALID_FRAC", "0.5")),
    )


def _center_stats(depth, center_frac: float) -> tuple[float | None, float]:
    """Median depth (m) and valid fraction of the central center_frac box — the
    exact region door_open_from_depth inspects. (None, 0.0) when no valid pixels."""
    import numpy as np

    arr = np.asarray(depth, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None, 0.0
    h, w = arr.shape
    ch, cw = max(1, int(h * center_frac)), max(1, int(w * center_frac))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    region = arr[y0:y0 + ch, x0:x0 + cw]
    valid = region[np.isfinite(region) & (region > 0.0)]
    if valid.size == 0:
        return None, 0.0
    return float(np.median(valid)), valid.size / region.size


def _sample(robot: WalkieRobot, center_frac: float, frames: int, settle: float = 0.15):
    """Sample `frames` depth frames; return (median-of-medians|None, mean valid frac).

    Per-frame median first (robust to speckle), then the median across frames
    (robust to one bad frame) — matching how the skill is meant to read a static
    doorway."""
    meds: list[float] = []
    fracs: list[float] = []
    for i in range(max(1, frames)):
        if i:
            time.sleep(settle)
        try:
            depth = robot.camera.get_depth()
        except Exception as exc:  # noqa: BLE001 - robot-side read failure
            print(f"  depth read failed ({exc})")
            continue
        if depth is None:
            print("  depth unavailable this frame")
            continue
        med, frac = _center_stats(depth, center_frac)
        if med is not None:
            meds.append(med)
            fracs.append(frac)
    if not meds:
        return None, 0.0
    return statistics.median(meds), statistics.mean(fracs)


def _recommend(closed_med: float, closed_frac: float, open_med: float, min_valid_frac: float) -> None:
    print("\n=== Recommendation ===")
    if closed_frac < min_valid_frac:
        print(
            f"WARNING: with the door closed only {closed_frac * 100:.0f}% of the centre had valid "
            f"depth — below WALKIE_DOOR_MIN_VALID_FRAC={min_valid_frac}. The skill would read this "
            f"as OPEN no matter the distance. Lower MIN_VALID_FRAC below {closed_frac:.2f}, or fix "
            f"the depth return on the door surface (glossy/dark doors absorb IR), then redo."
        )
    if open_med != float("inf") and open_med <= closed_med:
        print(
            f"WARNING: open ({open_med:.2f} m) is not farther than closed ({closed_med:.2f} m) — the "
            f"two readings don't separate. Re-aim so the open doorway shows the far room (not a wall "
            f"just beyond the door), then redo."
        )
        return
    if open_med == float("inf"):
        rec = round(closed_med + 0.4, 2)
        print(f"Open path read as clear (far / see-through). Set the threshold a margin above the "
              f"closed surface ({closed_med:.2f} m):")
    else:
        rec = round((closed_med + open_med) / 2, 2)
        print(f"Midpoint between closed ({closed_med:.2f} m) and open ({open_med:.2f} m):")
    print(f"\n    WALKIE_DOOR_CLEAR_M = \"{rec}\"\n")
    print("Put that in root config.toml (the [walkie] door section), then re-run with --watch to "
          "confirm: the closed door should print CLOSED and the open door OPEN.")


def guided(robot: WalkieRobot, frames: int) -> None:
    clear_m, center_frac, min_valid_frac = _knobs()
    print(f"Connected: {robot.is_connected}")
    print(f"Current knobs: WALKIE_DOOR_CLEAR_M={clear_m}  CENTER_FRAC={center_frac}  "
          f"MIN_VALID_FRAC={min_valid_frac}")
    print("Aim the camera straight at the doorway from where the robot would stop on entry "
          "(~1 m back), so the central box covers the door, not the floor/frame.\n")

    input("[1/2] Make sure the door is CLOSED, then press Enter to sample...")
    closed_med, closed_frac = _sample(robot, center_frac, frames)
    if closed_med is None:
        print("No valid depth in the centre with the door closed — cannot calibrate. Check the "
              "depth stream and the camera aim.")
        return
    print(f"  CLOSED: median={closed_med:.2f} m  valid={closed_frac * 100:.0f}%")

    input("\n[2/2] Now OPEN the door (clear the doorway), then press Enter to sample...")
    open_med, open_frac = _sample(robot, center_frac, frames)
    if open_med is None:
        print("  OPEN: centre reads all-invalid / see-through (valid~0%) — that is a clear path.")
        open_med = float("inf")
    else:
        print(f"  OPEN: median={open_med:.2f} m  valid={open_frac * 100:.0f}%")

    _recommend(closed_med, closed_frac, open_med, min_valid_frac)


def watch(robot: WalkieRobot) -> None:
    clear_m, center_frac, min_valid_frac = _knobs()
    print(f"Connected: {robot.is_connected}")
    print(f"Live readout (Ctrl-C to stop). Thresholds: CLEAR_M={clear_m}  CENTER_FRAC={center_frac}  "
          f"MIN_VALID_FRAC={min_valid_frac}\n")
    try:
        while True:
            try:
                depth = robot.camera.get_depth()
            except Exception as exc:  # noqa: BLE001
                print(f"depth read failed ({exc})")
                time.sleep(0.5)
                continue
            if depth is None:
                print("depth unavailable")
                time.sleep(0.5)
                continue
            med, frac = _center_stats(depth, center_frac)
            verdict = door_open_from_depth(
                depth, clear_m=clear_m, center_frac=center_frac, min_valid_frac=min_valid_frac
            )
            med_s = f"{med:.2f} m" if med is not None else "  n/a "
            print(f"center median={med_s}  valid={frac * 100:3.0f}%  ->  {'OPEN' if verdict else 'CLOSED'}")
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nstopped.")


def main() -> None:
    load_dotenv()
    load_config()
    ap = argparse.ArgumentParser(description="Calibrate WALKIE_DOOR_CLEAR_M on the robot.")
    ap.add_argument("--watch", action="store_true",
                    help="live continuous OPEN/CLOSED readout instead of guided capture")
    ap.add_argument("--frames", type=int, default=int(os.getenv("WALKIE_DOOR_CAL_FRAMES", "8")),
                    help="frames per sample in guided mode (default 8)")
    args = ap.parse_args()

    robot = get_robot()
    if args.watch:
        watch(robot)
    else:
        guided(robot, args.frames)


if __name__ == "__main__":
    main()
