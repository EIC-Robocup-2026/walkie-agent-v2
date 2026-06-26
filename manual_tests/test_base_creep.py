"""Manual hardware check for the Nav2-free base creep (cmd_vel direct drive).

The whole `creep_base_relative` fix rests on ONE assumption: a raw Twist published
to the SDK's `cmd_vel` topic actually moves the base (i.e. that topic is the base
*input*, not Nav2's muxed *output*). This script verifies that in isolation, then
exercises the helper itself.

Needs the robot reachable and on open floor with ~0.5 m clear all round (no table —
this bypasses the costmap, so nothing will stop the base but the code's own guards).
Keep the e-stop handy.

Run: uv run python -m manual_tests.test_base_creep            # raw check + fwd + strafe
     uv run python -m manual_tests.test_base_creep --raw-only # just the cmd_vel probe
     uv run python -m manual_tests.test_base_creep --forward 0.15 --strafe -0.10
"""

from __future__ import annotations

import argparse
import math
import os
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from walkie_sdk import WalkieRobot
from walkie_sdk.config.ros_topics import NAV_TOPICS

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


def raw_cmd_vel_probe(robot, *, vx=0.05, secs=1.0, hz=15.0) -> bool:
    """THE load-bearing check: does publishing raw cmd_vel move the base?

    Publishes linear.x=vx at hz for secs, then zero, and reports the odom
    displacement. If the base creeps ~vx*secs forward, the topic reaches the base
    and the whole fix is sound. If it doesn't move, the topic/mux is the real
    blocker and creep_base_relative won't work until that's resolved.
    """
    topic = robot.nav.cmd_vel_topic
    transport = robot.nav._transport
    msg_type = NAV_TOPICS["cmd_vel_type"]
    p0 = _pose(robot)
    if not p0:
        print("[probe] no odom fix; cannot measure — is the robot localised?")
        return False
    print(f"[probe] publishing linear.x={vx} m/s for {secs}s on {topic!r} ...")
    dt = 1.0 / hz
    try:
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            transport.publish(topic, msg_type, {
                "linear": {"x": float(vx), "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            })
            time.sleep(dt)
    finally:
        for _ in range(3):
            transport.publish(topic, msg_type, {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            })
    time.sleep(0.3)
    p1 = _pose(robot)
    moved = _displacement(p0, p1)
    expected = abs(vx) * secs
    print(f"[probe] odom moved {moved:.3f} m (expected ~{expected:.3f} m)")
    ok = moved > expected * 0.3  # generous: any clear motion proves the channel
    print("[probe] PASS — cmd_vel reaches the base, creep is sound" if ok
          else "[probe] FAIL — base did not move; check the cmd_vel topic / twist mux")
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
    ap.add_argument("--raw-only", action="store_true", help="just the cmd_vel probe")
    ap.add_argument("--forward", type=float, default=0.15, help="forward creep (m)")
    ap.add_argument("--strafe", type=float, default=-0.10, help="lateral creep, +left (m)")
    args = ap.parse_args()

    load_dotenv()
    load_config()
    robot = get_robot()
    print("Robot connected:", robot.is_connected)
    input("Clear ~0.5 m around the base, then press Enter to start... ")

    try:
        if not raw_cmd_vel_probe(robot):
            print("Raw probe failed — stopping before exercising the helper.")
            return
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
