"""Manual hardware check for the Nav2-free base creep (cmd_vel direct drive).

The whole `creep_base_relative` fix rests on ONE assumption: a raw Twist published
to the SDK's `cmd_vel` topic actually moves the base (i.e. that topic is the base
*input*, not Nav2's muxed *output*). This script verifies that in isolation, then
exercises the helper itself.

Needs the robot reachable and on open floor with ~0.5 m clear all round (no table —
this bypasses the costmap, so nothing will stop the base but the code's own guards).
Keep the e-stop handy.

Run: uv run python -m manual_tests.test_base_creep            # raw linear + rotate + creep
     uv run python -m manual_tests.test_base_creep --raw-only # both raw probes, no creep
     uv run python -m manual_tests.test_base_creep --rotate   # ONLY the angular spin probe
     uv run python -m manual_tests.test_base_creep --forward 0.15 --strafe -0.10

The --rotate probe answers "will the Restaurant scan's RESTAURANT_SCAN_ROTATE_MODE=cmdvel
(continuous spin) actually turn this base?" — it publishes angular.z and measures heading.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from walkie_sdk import WalkieRobot

from tasks.skills.navigation import creep_base_relative
from walkie_config import load_config

ZENOH_PORT = 7447


def get_robot() -> WalkieRobot:
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


def _pose(robot) -> dict | None:
    return robot.status.get_position()


def _displacement(p0, p1) -> float:
    return math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])


def _heading_delta(p0, p1) -> float:
    """Absolute heading change (rad), wrap-safe."""
    d = p1["heading"] - p0["heading"]
    return abs(math.atan2(math.sin(d), math.cos(d)))


def raw_cmd_vel_probe(robot, *, vx=0.05, secs=1.0, hz=15.0) -> bool:
    """THE load-bearing check: does publishing raw cmd_vel move the base?

    Publishes linear.x=vx at hz for secs via ``nav.set_velocity`` (the SDK helper that
    builds the correct geometry_msgs/msg/TwistStamped), then zero, and reports the odom
    displacement. If the base creeps ~vx*secs forward, the channel reaches the base and the
    whole fix is sound. If it doesn't move, the topic/mux is the real blocker.
    """
    p0 = _pose(robot)
    if not p0:
        print("[probe] no odom fix; cannot measure — is the robot localised?")
        return False
    print(f"[probe] publishing linear.x={vx} m/s for {secs}s via nav.set_velocity ...")
    dt = 1.0 / hz
    try:
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            robot.nav.set_velocity(float(vx), 0.0, 0.0)
            time.sleep(dt)
    finally:
        for _ in range(3):
            robot.nav.set_velocity(0.0, 0.0, 0.0)
    time.sleep(0.3)
    p1 = _pose(robot)
    moved = _displacement(p0, p1)
    expected = abs(vx) * secs
    print(f"[probe] odom moved {moved:.3f} m (expected ~{expected:.3f} m)")
    ok = moved > expected * 0.3  # generous: any clear motion proves the channel
    print("[probe] PASS — cmd_vel reaches the base, creep is sound" if ok
          else "[probe] FAIL — base did not move; check the cmd_vel topic / twist mux")
    return ok


def raw_cmd_vel_rotate_probe(robot, *, wz=0.3, secs=2.0, hz=15.0) -> bool:
    """Does publishing raw ANGULAR cmd_vel ROTATE the base?

    This is what the Restaurant live scan's ``RESTAURANT_SCAN_ROTATE_MODE=cmdvel`` needs —
    that sweep publishes ``angular.z`` (a continuous spin), NOT the linear creep the probe
    above tests, and a base can accept one but not the other. Publishes ``angular.z=wz`` at
    hz for secs, then zero, and reports the odom HEADING change. PASS ⇒ the cmdvel scan
    mode will actually turn the base; FAIL ⇒ it would stand still — use the default
    ``gotostep`` mode (which rotates via ``nav.go_to``).
    """
    p0 = _pose(robot)
    if not p0:
        print("[rotate] no odom fix; cannot measure — is the robot localised?")
        return False
    print(f"[rotate] publishing angular.z={wz} rad/s ({math.degrees(wz):.0f} deg/s) "
          f"for {secs}s via nav.set_velocity ...")
    dt = 1.0 / hz
    try:
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            robot.nav.set_velocity(0.0, 0.0, float(wz))
            time.sleep(dt)
    finally:
        for _ in range(3):
            robot.nav.set_velocity(0.0, 0.0, 0.0)
    time.sleep(0.3)
    p1 = _pose(robot)
    turned = _heading_delta(p0, p1)
    expected = abs(wz) * secs
    print(f"[rotate] odom heading changed {math.degrees(turned):.1f} deg "
          f"(expected ~{math.degrees(expected):.0f} deg)")
    ok = turned > expected * 0.3  # generous: any clear rotation proves the channel
    print("[rotate] PASS — angular cmd_vel rotates the base; the scan's cmdvel mode will work" if ok
          else "[rotate] FAIL — base did not rotate; the cmdvel scan mode would stand still "
               "(keep RESTAURANT_SCAN_ROTATE_MODE=gotostep)")
    return ok


def creep_check(robot, forward_m: float, left_m: float) -> None:
    """Drive creep_base_relative and report requested vs. measured displacement."""
    ctx = SimpleNamespace(walkie=SimpleNamespace(nav=robot.nav, status=robot.status))
    p0 = _pose(robot)
    requested = math.hypot(forward_m, left_m)
    print(f"[creep] requesting forward={forward_m:+.2f} left={left_m:+.2f} (|{requested:.2f}| m) ...")
    ok = creep_base_relative(ctx, forward_m, left_m)
    time.sleep(0.3)
    p1 = _pose(robot)
    moved = _displacement(p0, p1) if (p0 and p1) else float("nan")
    print(f"[creep] returned {ok}; odom moved {moved:.3f} m (requested {requested:.3f} m)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-only", action="store_true", help="both raw cmd_vel probes, no creep")
    ap.add_argument("--rotate", action="store_true",
                    help="ONLY the angular spin probe (what the scan's cmdvel mode uses)")
    ap.add_argument("--wz", type=float, default=0.3, help="angular rate for the rotate probe (rad/s)")
    ap.add_argument("--forward", type=float, default=0.15, help="forward creep (m)")
    ap.add_argument("--strafe", type=float, default=-0.10, help="lateral creep, +left (m)")
    args = ap.parse_args()

    load_dotenv()
    load_config()
    robot = get_robot()
    print("Robot connected:", robot.is_connected)
    input("Clear ~0.5 m around the base, then press Enter to start... ")

    try:
        # Rotate-only: just answer "does angular cmd_vel spin the base" (the scan's cmdvel mode).
        if args.rotate:
            raw_cmd_vel_rotate_probe(robot, wz=args.wz)
            return
        if not raw_cmd_vel_probe(robot):
            print("Raw linear probe failed — stopping before exercising the helper.")
            return
        # Also test ANGULAR cmd_vel — the scan's cmdvel mode spins, it doesn't translate.
        raw_cmd_vel_rotate_probe(robot, wz=args.wz)
        if args.raw_only:
            return
        if args.forward:
            creep_check(robot, args.forward, 0.0)
            time.sleep(0.5)
        if args.strafe:
            creep_check(robot, 0.0, args.strafe)
    finally:
        robot.nav.stop()  # belt-and-braces e-stop on the way out


if __name__ == "__main__":
    main()
