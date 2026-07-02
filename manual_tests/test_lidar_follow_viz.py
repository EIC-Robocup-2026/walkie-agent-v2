"""Live top-down visualization of the hybrid lidar+CV person tracker. No driving.

Runs the exact scan pipeline the hybrid ``follow_person`` uses — `cluster_scan`
→ map-frame lift → gate association → :class:`AlphaBetaTrack` — against the
real robot's lidar and odometry, and plots it in the map frame (matplotlib,
mirroring walkie-sdk/tests/test_lidar_interactive.py): grey scan points,
colored candidate cluster centroids, the gate circle at the track's prediction,
the track position + velocity arrow, CV fixes as stars, and the robot pose as a
triangle. The base is never commanded, so this is safe to run anywhere.

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

from tasks.skills.lidar_track import (
    AlphaBetaTrack,
    LidarFollowParams,
    associate,
    cluster_scan,
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


def _scan_points_map(scan: dict, pose: dict, p: LidarFollowParams) -> tuple[list, list]:
    """Every valid beam as a map-frame (xs, ys) — display only."""
    xs, ys = [], []
    angle = scan.get("angle_min", 0.0)
    inc = scan.get("angle_increment", 0.0)
    rmin = scan.get("range_min", 0.0) or 0.0
    rmax = min(float(scan.get("range_max") or p.max_range), p.max_range)
    for r in scan.get("ranges") or []:
        theta = angle
        angle += inc
        try:
            rf = float(r)
        except (TypeError, ValueError):
            continue
        if math.isnan(rf) or math.isinf(rf) or rf < rmin or rf > rmax:
            continue
        x, y = sensor_to_map(rf * math.cos(theta), rf * math.sin(theta), pose, p)
        xs.append(x)
        ys.append(y)
    return xs, ys


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

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[FAIL] matplotlib not installed. Run: uv pip install matplotlib")
        return 1

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

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_aspect("equal")
        ax.set_xlabel("map x (m)")
        ax.set_ylabel("map y (m)")
        ax.grid(True, linestyle=":", alpha=0.5)
        (scan_dots,) = ax.plot([], [], ".", color="0.6", markersize=2, label="scan")
        (cand_dots,) = ax.plot([], [], "o", color="tab:orange", markersize=7,
                               fillstyle="none", label="candidates")
        (track_dot,) = ax.plot([], [], "o", color="tab:green", markersize=10, label="track")
        (fix_star,) = ax.plot([], [], "*", color="tab:purple", markersize=14, label="CV fix")
        (robot_tri,) = ax.plot([], [], "b^", markersize=12, label="robot")
        gate_circle = plt.Circle((0, 0), p.gate, fill=False, color="tab:green",
                                 linestyle="--", alpha=0.7)
        ax.add_patch(gate_circle)
        gate_circle.set_visible(False)
        (vel_line,) = ax.plot([], [], "-", color="tab:green", lw=2)  # 1-second velocity lead
        ax.legend(loc="upper right")

        def on_click(event):
            nonlocal track
            if event.inaxes is not ax or event.xdata is None:
                return
            now = time.monotonic()
            if track is None:
                track = AlphaBetaTrack(event.xdata, event.ydata, now,
                                       alpha=p.alpha, beta=p.beta, max_speed=p.max_speed)
            else:
                track.reseed(event.xdata, event.ydata, now)
            print(f"[seed] track @ ({event.xdata:.2f}, {event.ydata:.2f}) — click again to reseed")

        fig.canvas.mpl_connect("button_press_event", on_click)
        print("Animating. CLICK the plot to seed the track on a cluster; close the window to stop.")
        plt.ion()
        plt.show()

        last_fix_xy = None
        while plt.fignum_exists(fig.number):
            now = time.monotonic()
            scan = robot.lidar.get_scan()
            pose = robot.status.get_position()
            if scan is None or not pose:
                plt.pause(0.1)
                continue

            # Same per-scan pipeline as _follow_person_lidar.
            clusters = cluster_scan(scan, p)
            pts = [sensor_to_map(c.cx, c.cy, pose, p) for c in clusters]
            gate = None
            if track is not None:
                pred = track.predict(now)
                gate = min(p.gate_max, p.gate + p.gate_grow * (now - track.t_accept))
                i = associate(pts, pred, gate)
                if i is not None:
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

            # ---- draw ----
            xs, ys = _scan_points_map(scan, pose, p)
            scan_dots.set_data(xs, ys)
            cand_dots.set_data([x for x, _ in pts], [y for _, y in pts])
            robot_tri.set_data([pose["x"]], [pose["y"]])
            if track is not None:
                tx, ty = track.predict(now)
                track_dot.set_data([tx], [ty])
                gate_circle.center = (tx, ty)
                gate_circle.set_radius(gate if gate is not None else p.gate)
                gate_circle.set_visible(True)
                vel_line.set_data([tx, tx + track.vx], [ty, ty + track.vy])
                speed = math.hypot(track.vx, track.vy)
            else:
                track_dot.set_data([], [])
                gate_circle.set_visible(False)
                vel_line.set_data([], [])
                speed = 0.0
            if last_fix_xy is not None:
                fix_star.set_data([last_fix_xy[0]], [last_fix_xy[1]])
            ax.set_xlim(pose["x"] - args.lim, pose["x"] + args.lim)
            ax.set_ylim(pose["y"] - args.lim, pose["y"] + args.lim)
            ax.set_title(
                f"{len(pts)} candidates / {len(xs)} pts — "
                + (f"TRACK v={speed:.2f} m/s" if track is not None else "no track (click to seed)")
            )
            if args.save:
                fig.savefig(args.save, dpi=100, bbox_inches="tight")
            fig.canvas.draw_idle()
            plt.pause(0.1)
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
