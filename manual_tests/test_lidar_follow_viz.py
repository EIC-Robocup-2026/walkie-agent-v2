"""Live top-down visualization of the hybrid lidar+CV person tracker. No driving.

Runs the exact scan pipeline the hybrid ``follow_person`` uses — `cluster_scan`
→ map-frame lift → gate association → :class:`AlphaBetaTrack` — against the
real robot's lidar and odometry, and plots it in the map frame through the SAME
:class:`~tasks.skills.lidar_follow_viz.LidarFollowViz` the live follow loop uses
(``HRI_FOLLOW_LIDAR_VIZ=1``), so what you see here is what the robot draws while
following: grey scan points, hollow-orange candidate cluster centroids, the RED
member points of the SELECTED cluster (the person's lidar cloud), the gate
circle at the track's prediction, the track position + velocity arrow, CV fixes
as stars, and the robot pose as a triangle. The base is never commanded, so
this is safe to run anywhere.

Seeding the track (identity source):
  * default   — CLICK a point on the plot to seed/reseed the track there
                (stands in for a CV fix; great for tuning the lidar knobs
                with no AI server running).
  * --cv      — also run the real :class:`_CvFixWorker` with
                ``select_largest_person`` (needs walkie-ai-server + the robot
                camera); its fixes seed/confirm/reseed exactly as in the loop.

Knobs come from the same ``HRI_FOLLOW_LIDAR_*`` env vars / TOML entries as the
real loop, so what you tune here is what the follow runs.

    uv run python -m manual_tests.test_lidar_follow_viz
    uv run python -m manual_tests.test_lidar_follow_viz --cv
    WALKIE_ROBOT_IP=10.0.0.10 uv run python -m manual_tests.test_lidar_follow_viz
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()
from pathlib import Path

# The lidar knobs live in the HRI task config, so layer it in the same way
# tasks/HRI/run.py does (task config, then root) — env / .env still win.
from tasks.common import load_task_config

load_task_config(Path(__file__).resolve().parent.parent / "tasks" / "HRI")

from walkie_sdk import WalkieRobot

from tasks.skills.lidar_follow_viz import LidarFollowViz, follow_title
from tasks.skills.lidar_track import (
    AlphaBetaTrack,
    LidarFollowParams,
    associate,
    cluster_scan,
    scan_points_map,
    sensor_to_map,
)

ZENOH_PORT = 7447


def get_robot() -> WalkieRobot:
    ros_protocol = os.getenv("WALKIE_ROS_PROTOCOL", "rosbridge")
    ros_port = int(os.getenv("WALKIE_ROS_PORT", str(ZENOH_PORT if ros_protocol == "zenoh" else 9090)))
    return WalkieRobot(
        ip=os.getenv("WALKIE_ROBOT_IP", "127.0.0.1"),
        ros_protocol=ros_protocol,
        ros_port=ros_port,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )


def _make_cv_worker(robot):
    """Full ctx (camera + AI server) driving the real CV fix worker."""
    from client import WalkieAIClient
    from interfaces.walkie_interface import WalkieInterface
    from tasks.base import TaskContext
    from tasks.skills.navigation import _CvFixWorker
    from tasks.skills.people import select_largest_person

    ctx = TaskContext(
        walkie=WalkieInterface(robot),
        walkieAI=WalkieAIClient(base_url=os.getenv("WALKIE_AI_BASE_URL", "http://localhost:5000")),
        model=None,  # the selector never touches the LLM
    )
    p = LidarFollowParams.from_env()
    return _CvFixWorker(ctx, select_largest_person, min_period=p.cv_min_period).start()


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize the hybrid lidar person tracker (no driving).")
    parser.add_argument("--cv", action="store_true",
                        help="also run the real CV fix worker (needs walkie-ai-server + camera)")
    parser.add_argument("--lim", type=float, default=6.0, help="plot half-extent (m) around the robot")
    parser.add_argument("--save", default=None, help="also save each rendered frame to this path")
    args = parser.parse_args()

    p = LidarFollowParams.from_env()
    print(f"Params: gate={p.gate}m(+{p.gate_grow}/s, cap {p.gate_max}) "
          f"jump={p.jump_base}+{p.jump_slope}r merge={p.merge_dist}m "
          f"width=[{p.cluster_min_width},{p.cluster_max_width}]m miss={p.miss_sec}s")
    robot = get_robot()
    worker = None
    track: AlphaBetaTrack | None = None
    consumed_seq = 0
    try:
        print(f"Scan topic: {robot.lidar.scan_topic}; waiting for first scan ...")
        if robot.lidar.get_once(timeout=10.0) is None:
            print("[FAIL] No scan received — is the lidar publishing?")
            return 1
        if args.cv:
            print("Starting the CV fix worker (select_largest_person) ...")
            worker = _make_cv_worker(robot)

        try:
            viz = LidarFollowViz(lim=args.lim, gate=p.gate)
        except ImportError:
            print("[FAIL] matplotlib not installed. Run: uv pip install matplotlib")
            return 1

        def on_click(event):
            nonlocal track
            if event.inaxes is not viz.ax or event.xdata is None:
                return
            now = time.monotonic()
            if track is None:
                track = AlphaBetaTrack(event.xdata, event.ydata, now,
                                       alpha=p.alpha, beta=p.beta, max_speed=p.max_speed)
            else:
                track.reseed(event.xdata, event.ydata, now)
            print(f"[seed] track @ ({event.xdata:.2f}, {event.ydata:.2f}) — click again to reseed")

        viz.fig.canvas.mpl_connect("button_press_event", on_click)
        print("Animating. CLICK the plot to seed the track on a cluster; close the window to stop.")

        last_fix_xy = None
        while viz.alive():
            now = time.monotonic()
            scan = robot.lidar.get_scan()
            pose = robot.status.get_position()
            if scan is None or not pose:
                time.sleep(0.1)
                continue

            # Same per-scan pipeline as _follow_person_lidar.
            clusters = cluster_scan(scan, p)
            pts = [sensor_to_map(c.cx, c.cy, pose, p) for c in clusters]
            gate = None
            selected_i = None
            if track is not None:
                pred = track.predict(now)
                gate = min(p.gate_max, p.gate + p.gate_grow * (now - track.t_accept))
                i = associate(pts, pred, gate)
                if i is not None:
                    selected_i = i
                    track.update(now, *pts[i])
                elif now - track.t_accept > p.miss_sec:
                    print("[track] gate dry past miss window — track dropped")
                    track = None
            if worker is not None:
                fix = worker.latest()
                if fix is not None and fix.seq != consumed_seq:
                    consumed_seq = fix.seq
                    last_fix_xy = fix.xy
                    if now - fix.t <= p.cv_max_age:
                        if track is None:
                            track = AlphaBetaTrack(fix.xy[0], fix.xy[1], fix.t,
                                                   alpha=p.alpha, beta=p.beta, max_speed=p.max_speed)
                            print(f"[cv] seeded track @ ({fix.xy[0]:.2f}, {fix.xy[1]:.2f})")
                        else:
                            px, py = track.predict(fix.t)
                            if math.hypot(fix.xy[0] - px, fix.xy[1] - py) > p.cv_reseed_dist:
                                track.reseed(fix.xy[0], fix.xy[1], fix.t)
                                print(f"[cv] RESEED @ ({fix.xy[0]:.2f}, {fix.xy[1]:.2f})")

            # The RED "selected cloud": member points of the cluster the gate
            # locked onto this scan (the lidar blob being followed).
            selected = ()
            if selected_i is not None:
                selected = [sensor_to_map(mx, my, pose, p) for (mx, my) in clusters[selected_i].points]

            viz.update(
                robot_xy=(pose["x"], pose["y"]),
                scan_xy=scan_points_map(scan, pose, p),
                cand_xy=pts,
                selected_xy=selected,
                track_xy=(track.predict(now) if track is not None else None),
                track_vel=((track.vx, track.vy) if track is not None else (0.0, 0.0)),
                gate=gate,
                fix_xy=last_fix_xy,
                title=follow_title("TRACK" if track is not None else "SEARCH", len(pts), track),
            )
            if args.save:
                viz.save(args.save)
            time.sleep(0.1)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    finally:
        if worker is not None:
            worker.stop()
        robot.disconnect()


if __name__ == "__main__":
    sys.exit(main())
