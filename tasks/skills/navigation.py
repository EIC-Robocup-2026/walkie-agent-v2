"""Base motion, look-at, approach, and the person-follow loop.

Moved out of tasks/HRI/skills.py into the shared tasks.skills package.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from typing import NamedTuple

from tasks.base import TaskContext

from .geometry import BBox
from .lidar_track import (
    AlphaBetaTrack,
    LidarFollowParams,
    associate,
    cluster_scan,
    sensor_to_map,
)
from .lift import lift_bbox_world_xy


def sweep_snapshots(
    ctx: TaskContext, offsets_deg: Sequence[float] = (10.0, 0.0, -10.0)
) -> list:
    """Capture a snapshot at each base-heading offset, then face forward again.

    The head servo only tilts (no pan), so looking left/right is a small in-place
    base rotation around the heading at entry. *offsets_deg* are degrees relative
    to that heading (positive = left/CCW, e.g. the default sweeps left, center,
    right). Returns the successful CameraSnapshots in the given order; each
    carries its own capture-time depth/pose geometry, so a bbox found in any of
    them lifts to the correct map-frame point regardless of where the robot was
    pointing when it was taken. With no odometry fix the base heading is unknown,
    so a rotation would aim arbitrarily — it falls back to a single forward
    snapshot. Best-effort: a failed capture is dropped, never raised.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] sweep_snapshots: odometry unavailable ({exc}); single frame")
        pose = None
    if not pose:
        snap = ctx.snapshot()
        return [snap] if snap is not None else []
    center = pose["heading"]
    settle = float(os.getenv("HRI_SWEEP_SETTLE_SEC", "1.0"))
    snaps = []
    for off in offsets_deg:
        ctx.rotate_to(center + math.radians(off))
        if settle > 0:
            time.sleep(settle)  # let the base + depth settle before capturing
        snap = ctx.snapshot()
        if snap is not None:
            snaps.append(snap)
    ctx.rotate_to(center)  # leave the robot facing forward again
    return snaps


def heading_to_point(ctx: TaskContext, x: float, y: float) -> float | None:
    """Map-frame heading from the robot's odometry position toward (x, y).

    None when odometry has no fix yet — ctx.current_pose()'s zeros fallback
    would make the atan2 aim somewhere arbitrary, so it is not used here.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] heading_to_point: odometry unavailable ({exc})")
        return None
    if not pose:
        return None
    return math.atan2(y - pose["y"], x - pose["x"])


def face_point(ctx: TaskContext, x: float, y: float) -> bool:
    """Rotate the base to face a map-frame point. Best-effort one-shot."""
    heading = heading_to_point(ctx, x, y)
    return ctx.rotate_to(heading) if heading is not None else False


def approach_point(
    ctx: TaskContext,
    x: float,
    y: float,
    *,
    stop_distance: float,
    blocking: bool = True,
    settle: float = 0.0,
) -> bool:
    """Drive toward a map-frame point, halting *stop_distance* m short of it.

    The goal is computed on the line from the robot's current position to
    (x, y), pulled back by *stop_distance*, and the base is left facing the
    target — so it approaches a person without driving onto them. If already
    within *stop_distance*, it only rotates to face the point (no forward
    creep). False when odometry has no fix (current_pose's zeros fallback would
    send the base to an arbitrary goal, so it's not used here) or nav fails.

    With *blocking* False the base is commanded but not waited on
    (``nav.go_to(blocking=False)``) and control returns after *settle* seconds
    — so a follow loop can keep re-targeting a moving person every tick instead
    of committing a full blocking drive to an already-stale position.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] approach_point: odometry unavailable ({exc})")
        return False
    if not pose:
        return False
    rx, ry = pose["x"], pose["y"]
    dx, dy = x - rx, y - ry
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    if dist <= stop_distance:
        gx, gy = rx, ry  # already close enough: rotate in place to face them
    else:
        ux, uy = dx / dist, dy / dist
        gx, gy = x - ux * stop_distance, y - uy * stop_distance
    if blocking:
        return ctx.goto(gx, gy, heading)
    try:
        # ctx.walkie.nav.go_to(gx, gy, heading, blocking=False)
        ctx.walkie.nav.go_to(gx, gy, heading, blocking=False, align_method="face_target")
    except Exception as exc:
        print(f"[skills] approach_point: non-blocking go_to failed ({exc})")
        return False
    if settle > 0:
        time.sleep(settle)  # let the base make progress before re-targeting
    return True


def approach_point_cmdvel(
    ctx: TaskContext,
    x: float,
    y: float,
    *,
    stop_distance: float,
    max_speed: float = 0.4,
    min_speed: float = 0.08,
    kp_lin: float = 0.8,
    kp_yaw: float = 1.2,
    max_yaw: float = 0.8,
    pose: dict | None = None,
) -> bool:
    """ONE body-frame velocity pulse toward a map point (Nav2-free); True if issued.

    A single ``nav.set_velocity`` command meant to run INSIDE a fast re-targeting
    loop (``follow_person``'s close-range regime) — NOT a self-contained move like
    :func:`creep_base_relative` (which loops + integrates odom to a fixed
    displacement). Here the caller's loop IS the closed loop: it re-evaluates the
    target and calls this again every tick, so each call just sets an instantaneous
    velocity setpoint.

    Why prefer this over a Nav2 goal at close range: a ``go_to`` target
    ~``stop_distance`` behind a person makes Nav2 treat the PERSON as a costmap
    obstacle and plan around / back up / refuse (the rotate-translate-rotate balk
    the creep skill also fights); a direct Twist ignores the costmap and drives
    straight toward them. That's a correctness win at close range, not just speed.

    Motion: rotate to FACE the target (``wz = kp_yaw * yaw_err``, clamped to
    ``max_yaw``) and drive FORWARD toward the point ``stop_distance`` short of it
    (``vx = kp_lin * remaining``, clamped to ``[min_speed, max_speed]``). No lateral
    strafe — facing the target keeps them centred for the next tick's detection.
    Forward speed is gated by heading alignment (``max(0, cos(yaw_err))``) so an
    off-axis target is turned toward before it drives, not lunged at sideways.
    Within ``stop_distance`` it only rotates to face (zero translation).

    IMPORTANT — the base HOLDS the last commanded velocity until the next command
    or the cmd_vel watchdog zeroes it. Keep *max_speed* low enough that one watchdog
    window of coast is a safe fraction of *stop_distance* (e.g. 0.4 m/s x ~0.5 s =
    0.2 m against a 1.2 m follow gap), and guarantee a zero-velocity stop when the
    loop ends. Best-effort: returns False and commands NOTHING without an odom fix
    or a set_velocity channel, so the caller can fall back to ``go_to``. *pose* may
    be passed in to reuse a read the caller already made (else it reads its own).
    """
    nav = getattr(ctx.walkie, "nav", None)
    if nav is None or not hasattr(nav, "set_velocity"):
        return False
    if pose is None:
        try:
            pose = ctx.walkie.status.get_position()
        except Exception as exc:
            print(f"[skills] approach_point_cmdvel: odometry unavailable ({exc})")
            return False
    if not pose:
        return False
    rx, ry, h = pose["x"], pose["y"], pose["heading"]
    dx, dy = x - rx, y - ry
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    yaw_err = math.atan2(
        math.sin(bearing - h), math.cos(bearing - h)
    )  # wrap to (-pi, pi]
    wz = max(-max_yaw, min(max_yaw, kp_yaw * yaw_err))
    remaining = dist - stop_distance
    if remaining <= 0.0:
        vx = 0.0  # already within the follow gap: only rotate to face
    else:
        align = max(
            0.0, math.cos(yaw_err)
        )  # ramp forward in as the base faces the target
        vx = min(max_speed, max(min_speed, kp_lin * remaining)) * align
    try:
        nav.set_velocity(float(vx), 0.0, float(wz))
    except Exception as exc:
        print(f"[skills] approach_point_cmdvel: set_velocity failed ({exc})")
        return False
    return True


def rotate_by(ctx: TaskContext, delta_rad: float, *, blocking: bool = True) -> bool:
    """Rotate the base by *delta_rad* relative to its current heading.

    Reads the real odometry heading (not ``current_pose``'s zeros fallback,
    which would make a relative turn meaningless) and no-ops without a fix.
    Positive = CCW / left, matching the frame-offset convention used elsewhere
    (a target left of center needs a positive turn to re-center it).

    With *blocking* False the turn is commanded but not waited on
    (``nav.go_to(blocking=False)``, the goal preempting any in-flight one), so a
    fast control loop can re-issue a fresh correction every tick — closed-loop,
    since the new target heading is recomputed from the latest odometry each
    time. Default True keeps the one-shot "look toward" behaviour.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] rotate_by: odometry unavailable ({exc})")
        return False
    if not pose:
        return False
    target = pose["heading"] + delta_rad
    if blocking:
        return ctx.rotate_to(target)
    try:
        ctx.walkie.nav.go_to(pose["x"], pose["y"], target, blocking=False)
    except Exception as exc:
        print(f"[skills] rotate_by: non-blocking go_to failed ({exc})")
        return False
    return True


def move_base_relative(
    ctx: TaskContext,
    dx: float,
    dy: float = 0.0,
    dheading_rad: float = 0.0,
    *,
    blocking: bool = True,
) -> bool:
    """Move the base relative to its CURRENT pose (robot-local frame).

    +dx = forward, +dy = left (matching the actuator agent's convention); a
    backward step is a negative dx. *dheading_rad* adds to the current heading
    (positive = CCW / left). Reads the real odometry heading (not
    ``current_pose``'s zeros fallback, which would turn a "relative" move into an
    absolute one toward the map origin) and no-ops without a fix. Best-effort:
    False on no fix or nav failure, never raises.

    Example: ``move_base_relative(ctx, -0.30)`` steps the base 30 cm straight back.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] move_base_relative: odometry unavailable ({exc})")
        return False
    if not pose:
        return False
    x_cur, y_cur, h = pose["x"], pose["y"], pose["heading"]
    x_global = x_cur + dx * math.cos(h) - dy * math.sin(h)
    y_global = y_cur + dx * math.sin(h) + dy * math.cos(h)
    target_heading = h + dheading_rad
    if blocking:
        return ctx.goto(x_global, y_global, target_heading)
    try:
        ctx.walkie.nav.go_to(x_global, y_global, target_heading, blocking=False)
        return True
    except Exception as exc:
        print(f"[skills] move_base_relative: non-blocking go_to failed ({exc})")
        return False


def creep_base_relative(
    ctx: TaskContext,
    forward_m: float,
    left_m: float = 0.0,
    *,
    speed_mps: float | None = None,
    tol_m: float | None = None,
    hz: float | None = None,
    hold_heading: bool = True,
) -> bool:
    """Nav2-free closed-loop relative translation for final precision docking.

    Drives the omnidirectional base by publishing a body-frame velocity ``Twist``
    straight to ``cmd_vel`` and integrating ODOMETRY displacement, stopping the
    instant the travelled planar distance reaches ``hypot(forward_m, left_m)``.

    This deliberately bypasses Nav2's planner / costmap / recovery behaviours,
    which are exactly what make a short goal **right next to a table** back the
    base up: the global planner routes around the inflated table obstacle, and/or
    the local controller can't strafe and so maneuvers (rotate-translate-rotate) —
    both read on the omni base as "it nudges itself backwards". Commanding the base
    directly in its own frame removes all of that. Integrating odom (never map)
    also sidesteps any map↔odom divergence — unlike ``move_base_relative``, which
    adds the offset to the odom pose and hands the sum to ``nav.go_to`` as a *map*
    goal — so the move ends exactly where asked relative to the start.

    +forward_m = toward the base heading, +left_m = to its left (REP-103 /
    actuator-agent convention; a backward step is a negative ``forward_m``).
    Heading is held with a light P term on yaw when *hold_heading*.

    Because the costmap that used to stop the base short of the table is gone, the
    drive is hard-guarded: a slow default speed eased down near the goal, a stop
    margin for publish/odom latency, a wall-clock timeout, and a guaranteed
    zero-velocity stop in a ``finally`` (so any exception or early return halts the
    base). Best-effort: True once the target displacement is reached, False on no
    odom fix or an unavailable cmd_vel channel; never raises.
    """
    ctx.walkie.robot.head.set_auto_tilt(False)
    target = math.hypot(forward_m, left_m)
    if target < 1e-3:
        return True
    speed = (
        float(os.getenv("WALKIE_CREEP_SPEED_MPS", "0.08"))
        if speed_mps is None
        else speed_mps
    )
    tol = float(os.getenv("WALKIE_CREEP_TOL_M", "0.01")) if tol_m is None else tol_m
    rate = float(os.getenv("WALKIE_CREEP_HZ", "15.0")) if hz is None else hz
    min_speed = float(os.getenv("WALKIE_CREEP_MIN_SPEED_MPS", "0.03"))
    kp_lin = float(os.getenv("WALKIE_CREEP_KP_LIN", "1.5"))
    kp_yaw = float(os.getenv("WALKIE_CREEP_KP_YAW", "1.2"))
    max_yaw = float(os.getenv("WALKIE_CREEP_MAX_YAW_RPS", "0.4"))
    stall_eps = float(os.getenv("WALKIE_CREEP_STALL_EPS_M", "0.002"))
    stall_sec = float(os.getenv("WALKIE_CREEP_STALL_SEC", "0.6"))

    nav = ctx.walkie.nav
    if not hasattr(nav, "set_velocity"):
        print("[skills] creep_base_relative: nav.set_velocity unavailable")
        ctx.walkie.robot.head.set_auto_tilt(True)
        return False

    start = ctx.walkie.status.get_position()
    if not start:
        print("[skills] creep_base_relative: no odom fix; refusing to creep")
        ctx.walkie.robot.head.set_auto_tilt(True)
        return False
    x0, y0, h0 = start["x"], start["y"], start["heading"]
    ux, uy = forward_m / target, left_m / target  # body-frame unit direction
    dt = 1.0 / max(1.0, rate)
    # one control cycle of coasting + a tolerance floor cover publish/odom latency
    stop_margin = max(tol, speed * dt)
    # Backstop sized to CRUISE speed (not min_speed): when odom drops out mid-loop
    # `travelled` freezes at 0 and the only thing left to stop the base is this
    # deadline, so it must reflect how fast we're actually commanding, not the slow
    # ease-in floor — otherwise it's ~3x too loose and rides full speed into the table.
    deadline = time.monotonic() + target / max(speed, 1e-3) + 1.0
    max_stall = max(
        1, int(stall_sec * rate)
    )  # cycles with no odom progress before bailing

    def _publish(vx: float, vy: float, wz: float) -> None:
        # SDK helper: builds the correct geometry_msgs/msg/TwistStamped and publishes it to
        # cmd_vel. A raw plain-Twist publish is malformed for this base (cmd_vel_type is
        # TwistStamped) and moves it 0 m — the "base stuck" symptom we saw.
        nav.set_velocity(float(vx), float(vy), float(wz))

    print(
        f"[skills] creep_base_relative: fwd={forward_m:+.2f} left={left_m:+.2f} "
        f"(|{target:.2f}|m @ {speed:.2f}m/s, Nav2-free)"
    )
    reached = False
    best_travelled = 0.0
    stalled = 0
    try:
        while time.monotonic() < deadline:
            pose = ctx.walkie.status.get_position()
            travelled = math.hypot(pose["x"] - x0, pose["y"] - y0) if pose else 0.0
            remaining = target - travelled
            if remaining <= stop_margin:
                reached = True
                break
            # No-progress guard: while we ARE commanding motion, dead/frozen odom or a
            # physically blocked base shows up as travelled not advancing. Bail rather
            # than ride the deadline at full speed (the distance check can't trip on
            # frozen odom — it reads the same stale value).
            if travelled > best_travelled + stall_eps:
                best_travelled = travelled
                stalled = 0
            else:
                stalled += 1
                if stalled >= max_stall:
                    print(
                        "[skills] creep_base_relative: no odom progress "
                        f"({travelled:.3f}m of {target:.2f}m); base stuck or odom "
                        "stale — stopping"
                    )
                    break
            v = min(speed, max(min_speed, kp_lin * remaining))  # ease in to the goal
            wz = 0.0
            if hold_heading and pose:
                yaw_err = math.atan2(
                    math.sin(h0 - pose["heading"]), math.cos(h0 - pose["heading"])
                )
                wz = max(-max_yaw, min(max_yaw, kp_yaw * yaw_err))
            _publish(v * ux, v * uy, wz)
            time.sleep(dt)
        else:
            print("[skills] creep_base_relative: timed out before reaching target")
    except Exception as exc:  # noqa: BLE001 — never let a creep raise mid-grasp
        print(f"[skills] creep_base_relative: drive error ({exc}); stopping")
    finally:
        for _ in range(3):  # publish zero a few times so the stop isn't lost
            try:
                _publish(0.0, 0.0, 0.0)
            except Exception:  # noqa: BLE001
                break
        ctx.walkie.robot.head.set_auto_tilt(True)
    return reached


def strafe_servo(
    ctx: TaskContext,
    lateral_error_fn: Callable[[], float | None],
    *,
    tol_m: float = 0.015,
    timeout_sec: float = 12.0,
    stall_sec: float | None = None,
    speed_mps: float | None = None,
    hz: float | None = None,
    hold_heading: bool = True,
) -> bool:
    """Closed-loop lateral (strafe) velocity servo via direct cmd_vel.

    Unlike :func:`creep_base_relative`, which drives a displacement computed
    ONCE, this servos on a LIVE error: every tick *lateral_error_fn* returns how
    far the base must still move to its LEFT (body +y, metres; negative =
    right; ``None`` = couldn't compute this tick), and the loop commands a
    P-eased strafe velocity toward zeroing it. Because the error is recomputed
    from the current pose each tick, the drive self-corrects — over/undershoot
    just shrinks on the next tick instead of being baked in.

    Obstacles are deliberately NOT a stop condition: the nav stack scales
    cmd_vel down near them, so the servo watches ACTUAL motion instead — odom
    position differencing per tick (odom twist is unusable here:
    ``status.get_velocity()`` drops the strafe axis linear.y). If the base is
    being commanded at/above the ease-in floor yet its planar displacement
    stays under ``WALKIE_CREEP_STALL_EPS_M`` for *stall_sec* (default
    ``WALKIE_GRASP_SERVO_STALL_SEC``, longer than creep's guard because the nav
    scaling ramps down gradually), it is as far over as the world allows —
    stop and let the caller continue from there (best-effort).

    Heading is held with the same yaw P-term as the creep, and all low-level
    gains reuse the ``WALKIE_CREEP_*`` knobs — one tuning surface for both
    cmd_vel drives. Hard-guarded like the creep: wall-clock *timeout_sec*
    backstop and a guaranteed zero-velocity stop in ``finally``. Returns True
    once ``|error| <= tol_m``; False on blocked/timeout/no odom/no cmd_vel
    channel (base stopped safely). Never raises.
    """
    speed = (
        float(os.getenv("WALKIE_CREEP_SPEED_MPS", "0.08"))
        if speed_mps is None
        else speed_mps
    )
    rate = float(os.getenv("WALKIE_CREEP_HZ", "15.0")) if hz is None else hz
    min_speed = float(os.getenv("WALKIE_CREEP_MIN_SPEED_MPS", "0.03"))
    kp_lin = float(os.getenv("WALKIE_CREEP_KP_LIN", "1.5"))
    kp_yaw = float(os.getenv("WALKIE_CREEP_KP_YAW", "1.2"))
    max_yaw = float(os.getenv("WALKIE_CREEP_MAX_YAW_RPS", "0.4"))
    stall_eps = float(os.getenv("WALKIE_CREEP_STALL_EPS_M", "0.002"))
    if stall_sec is None:
        stall_sec = float(os.getenv("WALKIE_GRASP_SERVO_STALL_SEC", "1.5"))

    nav = ctx.walkie.nav
    if not hasattr(nav, "set_velocity"):
        print("[skills] strafe_servo: nav.set_velocity unavailable")
        return False

    ctx.walkie.robot.head.set_auto_tilt(False)
    start = ctx.walkie.status.get_position()
    if not start:
        print("[skills] strafe_servo: no odom fix; refusing to drive")
        ctx.walkie.robot.head.set_auto_tilt(True)
        return False
    h0 = start["heading"]
    dt = 1.0 / max(1.0, rate)
    deadline = time.monotonic() + timeout_sec
    max_stall = max(
        1, int(stall_sec * rate)
    )  # commanded-but-not-moving ticks before bailing

    reached = False
    stalled = 0
    prev_xy = (start["x"], start["y"])
    commanding = False  # was the last published |vy| at/above the ease-in floor?
    try:
        while time.monotonic() < deadline:
            pose = ctx.walkie.status.get_position()
            # Blocked guard: commanded motion with no odom displacement means the
            # nav layer scaled us to ~zero (obstacle / footprint limit) or odom is
            # frozen — either way, more commanding won't move the base.
            if pose:
                moved = math.hypot(pose["x"] - prev_xy[0], pose["y"] - prev_xy[1])
                prev_xy = (pose["x"], pose["y"])
            else:
                moved = 0.0
            if commanding and moved < stall_eps:
                stalled += 1
                if stalled >= max_stall:
                    print(
                        f"[skills] strafe_servo: commanded but not moving for "
                        f"{stall_sec:.1f}s; blocked — stopping here"
                    )
                    break
            else:
                stalled = 0

            err = lateral_error_fn()
            if err is None:  # can't compute this tick — hold still, count as stalled
                nav.set_velocity(0.0, 0.0, 0.0)
                commanding = False
                stalled += 1
                if stalled >= max_stall:
                    print("[skills] strafe_servo: error unavailable; stopping")
                    break
                time.sleep(dt)
                continue
            if abs(err) <= tol_m:
                reached = True
                break

            v = min(speed, max(min_speed, kp_lin * abs(err)))  # ease in to the goal
            vy = math.copysign(v, err)
            wz = 0.0
            if hold_heading and pose:
                yaw_err = math.atan2(
                    math.sin(h0 - pose["heading"]), math.cos(h0 - pose["heading"])
                )
                wz = max(-max_yaw, min(max_yaw, kp_yaw * yaw_err))
            nav.set_velocity(0.0, vy, wz)
            commanding = v >= min_speed
            time.sleep(dt)
        else:
            print("[skills] strafe_servo: timed out before aligning")
    except Exception as exc:  # noqa: BLE001 — never let the servo raise mid-grasp
        print(f"[skills] strafe_servo: drive error ({exc}); stopping")
    finally:
        for _ in range(3):  # publish zero a few times so the stop isn't lost
            try:
                nav.set_velocity(0.0, 0.0, 0.0)
            except Exception:  # noqa: BLE001
                break
        ctx.walkie.robot.head.set_auto_tilt(True)
    return reached


def tilt_head(ctx: TaskContext, angle_rad: float, *, settle: float = 0.0) -> None:
    """Tilt the head servo best-effort; optional settle sleep before a capture.

    POSITIVE angle = camera tilts DOWN, negative = up (walkie-sdk convention).
    The servo publish is asynchronous (it returns before the head arrives), so
    pass *settle* > 0 when the next action depends on the new tilt. Never raises
    — an off-robot stub may lack ``robot.head`` entirely.
    """
    try:
        ctx.walkie.robot.head.tilt(angle_rad)
        time.sleep(0.5)  # let the servo start moving before the settle sleep
    except Exception as exc:
        print(f"[skills] head tilt failed ({exc})")
    if settle > 0:
        time.sleep(settle)


class MotionPredictor:
    """Estimate a tracked person's map-frame velocity and extrapolate where they
    went when they briefly leave the camera frame.

    Sightings are ``(t, x, y)`` map-frame points (``lift_bbox_world_xy`` output).
    Because they live in the fixed map frame, differencing them gives the
    person's true ground velocity — the robot's own motion doesn't pollute it.
    :meth:`velocity` is a least-squares linear fit over a sliding time window
    (robust to the per-frame depth-lift jitter a two-point difference would
    amplify); :meth:`predict` extrapolates from the last sighting, clamped to a
    human-plausible speed and a max horizon + distance so a noisy fit can't
    fling the goal across the room.

    :meth:`predict` returns ``None`` — telling the follow loop to fall back to a
    rotate-search — when there is too little history, the person was essentially
    stationary (no direction to extrapolate), or the last sighting is too stale
    to trust. All tunables come from ``HRI_FOLLOW_PREDICT_*`` env vars.
    """

    def __init__(self) -> None:
        self.window_sec = float(os.getenv("HRI_FOLLOW_PREDICT_WINDOW_SEC", "4.0"))
        self.min_samples = int(os.getenv("HRI_FOLLOW_PREDICT_MIN_SAMPLES", "3"))
        self.horizon_sec = float(os.getenv("HRI_FOLLOW_PREDICT_HORIZON_SEC", "3.0"))
        self.max_speed = float(os.getenv("HRI_FOLLOW_PREDICT_MAX_SPEED", "1.5"))
        self.min_speed = float(os.getenv("HRI_FOLLOW_PREDICT_MIN_SPEED", "0.15"))
        self.max_dist = float(os.getenv("HRI_FOLLOW_PREDICT_MAX_DIST", "3.0"))
        self.gap_reset_sec = float(os.getenv("HRI_FOLLOW_PREDICT_GAP_RESET_SEC", "3.0"))
        self._hist: list[tuple[float, float, float]] = []

    def update(self, t: float, xy: tuple[float, float]) -> None:
        """Record a fresh map-frame sighting at monotonic time *t*."""
        if self._hist and t - self._hist[-1][0] > self.gap_reset_sec:
            self._hist.clear()  # long blackout: the old trajectory is stale
        self._hist.append((t, xy[0], xy[1]))
        cutoff = t - self.window_sec
        self._hist = [s for s in self._hist if s[0] >= cutoff]

    def velocity(self) -> tuple[float, float] | None:
        """Least-squares ``(vx, vy)`` over the window, or None if degenerate."""
        if len(self._hist) < self.min_samples:
            return None
        ts = [s[0] for s in self._hist]
        t0 = ts[0]
        tu = [t - t0 for t in ts]
        n = len(tu)
        mean_t = sum(tu) / n
        denom = sum((t - mean_t) ** 2 for t in tu)
        if denom <= 1e-6:  # all samples ~same timestamp: no usable slope
            return None

        def slope(vals: list[float]) -> float:
            mean_v = sum(vals) / n
            return sum((tu[i] - mean_t) * (vals[i] - mean_v) for i in range(n)) / denom

        return slope([s[1] for s in self._hist]), slope([s[2] for s in self._hist])

    def predict(self, t: float) -> tuple[float, float] | None:
        """Extrapolated map-frame point at time *t*, or None to fall back to search."""
        if not self._hist:
            return None
        v = self.velocity()
        if v is None:
            return None
        vx, vy = v
        speed = math.hypot(vx, vy)
        if speed < self.min_speed:
            return None  # essentially stationary — no heading to extrapolate
        if speed > self.max_speed:  # clamp to a human-plausible walking speed
            vx, vy = vx / speed * self.max_speed, vy / speed * self.max_speed
        lt, lx, ly = self._hist[-1]
        dt = t - lt
        if dt > self.horizon_sec:
            return None  # last sighting too old to trust the extrapolation
        dt = max(0.0, dt)
        px, py = lx + vx * dt, ly + vy * dt
        ddx, ddy = px - lx, py - ly
        dist = math.hypot(ddx, ddy)
        if dist > self.max_dist:  # cap displacement from the last real sighting
            px, py = lx + ddx / dist * self.max_dist, ly + ddy / dist * self.max_dist
        return px, py

    def reset(self) -> None:
        self._hist.clear()


def _draw_follow_viz(img, box, *, color, label, save_path) -> None:
    """Best-effort: draw the tracked person's box + a status banner on *img*,
    then write it to *save_path* (overwritten each call).

    Used by :func:`follow_person` when ``HRI_FOLLOW_VIZ=1`` so you can watch which
    body the robot is locked onto — *box* is the tracked person on the live frame
    (which already shows everyone else), *color* encodes the loop state, and
    *label* is the one-line status. Never raises: a viz glitch must not disturb
    the follow loop.
    """
    try:
        from PIL import ImageDraw

        vis = img.convert("RGB").copy()
        draw = ImageDraw.Draw(vis)
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.rectangle((0, 0, vis.width, 22), fill=(0, 0, 0))
        draw.text((6, 6), label, fill=color)
        vis.save(save_path, "JPEG", quality=70)
    except Exception as exc:
        print(f"[skills] follow viz failed ({exc})")


@contextmanager
def _follow_base_stop(ctx: TaskContext, used_cmdvel):
    """Guarantee the base halts when a cmd_vel follow run exits by ANY path.

    *used_cmdvel* is a 0-arg callable returning whether :func:`follow_person` ever
    issued a raw ``cmd_vel`` Twist this run. On exit — normal, ``break``, stopper
    trigger, or exception — if it did, publish a few zero-velocity commands and
    cancel any lingering Nav2 goal left by a ``go_to`` tick, so the base is still
    for whatever the caller does next (e.g. the HRI bag place). A no-op when only
    ``go_to`` was used, leaving that path's teardown exactly as before.
    """
    try:
        yield
    finally:
        if used_cmdvel():
            nav = getattr(ctx.walkie, "nav", None)
            for _ in range(3):  # publish zero a few times so the stop isn't lost
                try:
                    nav.set_velocity(0.0, 0.0, 0.0)
                except Exception:  # noqa: BLE001
                    break
            try:
                nav.cancel()  # also halt any async go_to goal still executing
            except Exception:  # noqa: BLE001
                pass


def _follow_person_cv(
    ctx: TaskContext,
    select,
    *,
    stopper=None,
    on_warmup=None,
    on_lost=None,
    on_stopped=None,
    follow_distance: float | None = None,
    timeout: float | None = None,
    nav_period: float | None = None,
    search_step_deg: float | None = None,
    max_lost: int | None = None,
    lead_gain: float | None = None,
    predict_on: bool | None = None,
) -> str:
    """Camera-paced follow loop (legacy path; ``HRI_FOLLOW_LIDAR=0``).

    Follow a person by re-targeting nav at their (predicted) position each tick.

    A dead-simple synchronous loop — no background thread. Every tick it
    snapshots the camera, asks *select* for the person's box, lifts it to a
    map-frame point, and drives ``nav.go_to`` there (led forward by the target's
    velocity × *lead_gain* seconds), stopping *follow_distance* m short. Because
    nav orients the base along its travel direction, driving to the target also
    faces it — there is no separate look-at/centering step, so the faster the
    loop runs the tighter it tracks. A :class:`MotionPredictor` keeps a running
    velocity estimate; when a tick yields no usable point (no detection, or the
    depth lift failed) the loop COASTS on the predictor's extrapolation, and only
    once that is exhausted does it rotate-search toward the last-seen side.
    Returns the exit reason: ``'stopped'`` | ``'lost'`` | ``'timeout'``.

    *select* is a callable ``(ctx, snap) -> bbox_xyxy | None`` — e.g.
    :func:`select_largest_person` (follow whoever's closest, for testing) or
    ``identity.select_person_to_follow`` (match an enrolled person by face first,
    attire fallback). *stopper*, when given, is a context manager carrying a
    ``.triggered`` :class:`threading.Event` (a :class:`tasks.skills.CommandListener`);
    it is entered AFTER *on_warmup* — so a background mic listener never transcribes
    the warmup speech — and its event ends the loop the instant it is set. *on_warmup* runs
    once before the loop (e.g. a spoken ack); *on_lost* runs once the first time
    the target is lost; *on_stopped* runs after a *stopper*-triggered exit, before
    teardown, so a closing speech overlaps the join cost. Every tunable defaults
    from the ``HRI_FOLLOW_*`` env vars.
    """
    if follow_distance is None:
        follow_distance = float(os.getenv("HRI_FOLLOW_DISTANCE_M", "1.0"))
    if timeout is None:
        timeout = float(os.getenv("HRI_FOLLOW_TIMEOUT_SEC", "90"))
    if nav_period is None:
        nav_period = float(os.getenv("HRI_FOLLOW_NAV_PERIOD_SEC", "0"))
    if search_step_deg is None:
        search_step_deg = float(os.getenv("HRI_FOLLOW_SEARCH_STEP_DEG", "10"))
    search_step = math.radians(search_step_deg)
    if max_lost is None:
        max_lost = int(os.getenv("HRI_FOLLOW_MAX_LOST", "8"))
    if lead_gain is None:
        lead_gain = float(os.getenv("HRI_FOLLOW_LEAD_GAIN", "0.5"))
    if predict_on is None:
        predict_on = os.getenv("HRI_FOLLOW_PREDICT", "1").lower() in (
            "1",
            "true",
            "yes",
        )
    # Keep driving to the last seen point for this long (s) when a tick yields no
    # detection — a dropped pose frame or a briefly-stationary person must NOT
    # fall straight into the rotate-search (which would otherwise stutter / stall).
    lost_grace = float(os.getenv("HRI_FOLLOW_LOST_GRACE_SEC", "1.5"))
    debug = os.getenv("HRI_FOLLOW_TRACK_DEBUG", "0").lower() in ("1", "true", "yes")
    # Visualization: annotate each frame with the tracked person's box + status
    # and write it to HRI_FOLLOW_VIZ_PATH (overwritten every tick) so you can
    # watch who the robot is locked onto. Separate flag from the timing debug.
    viz = os.getenv("HRI_FOLLOW_VIZ", "0").lower() in ("1", "true", "yes")
    viz_path = os.getenv("HRI_FOLLOW_VIZ_PATH", "follow_viz.jpg")
    # Close-range drive: within the CMDVEL_ENTER band, drive the base straight at the
    # target with a direct cmd_vel pulse (approach_point_cmdvel) instead of a Nav2
    # go_to — a goal ~follow_distance behind the person otherwise makes Nav2 treat
    # the PERSON as a costmap obstacle and plan around / back off / refuse. Hysteresis
    # (enter < exit) stops mode-thrash at the boundary. Off by default (on-robot A/B).
    cmdvel_on = os.getenv("HRI_FOLLOW_CMDVEL", "0").lower() in ("1", "true", "yes")
    cmdvel_enter = follow_distance + float(
        os.getenv("HRI_FOLLOW_CMDVEL_ENTER_MARGIN_M", "0.8")
    )
    cmdvel_exit = follow_distance + float(
        os.getenv("HRI_FOLLOW_CMDVEL_EXIT_MARGIN_M", "1.4")
    )
    cmdvel_kw = dict(
        max_speed=float(os.getenv("HRI_FOLLOW_CMDVEL_MAX_SPEED", "0.4")),
        min_speed=float(os.getenv("HRI_FOLLOW_CMDVEL_MIN_SPEED", "0.08")),
        kp_lin=float(os.getenv("HRI_FOLLOW_CMDVEL_KP_LIN", "0.8")),
        kp_yaw=float(os.getenv("HRI_FOLLOW_CMDVEL_KP_YAW", "1.2")),
        max_yaw=float(os.getenv("HRI_FOLLOW_CMDVEL_MAX_YAW", "0.8")),
    )

    predictor = MotionPredictor() if predict_on else None
    idle = threading.Event()  # never set — an interruptible sleep when no stopper
    lost = 0
    cmdvel_mode = False  # hysteresis: currently in the close-range cmd_vel regime?
    used_cmdvel = False  # ever published a raw Twist? (gates the finally stop + cancel)
    last_goto = False  # last drive command was a Nav2 goal (cancel it before a Twist)
    last_dir = 1.0  # default search direction (left) until first seen
    last_xy: tuple[float, float] | None = (
        None  # last good lifted point, for the grace hold
    )
    last_seen_t: float | None = None
    last_box: BBox | None = None  # last selected box, drawn dim while coasting (viz)
    deadline = time.monotonic() + timeout
    reason = "timeout"
    n, t_log = 0, time.monotonic()
    if on_warmup is not None:
        on_warmup()  # speak the ack BEFORE the stopper starts (so the mic skips it)
    with (
        stopper if stopper is not None else nullcontext() as listener,
        _follow_base_stop(ctx, lambda: used_cmdvel),
    ):
        triggered = getattr(listener, "triggered", None)
        while time.monotonic() < deadline and not (
            triggered is not None and triggered.is_set()
        ):
            now = time.monotonic()
            snap = ctx.snapshot()
            t_snap = time.monotonic()
            box = select(ctx, snap) if snap is not None else None
            t_sel = time.monotonic()
            # In view: lift the box and lead it by the fitted velocity. The lift
            # feeds the predictor (raw), so it can coast when a tick comes up empty.
            side = None
            target = None
            if box is not None:
                side = (box[0] + box[2]) / 2 / snap.img.width - 0.5
                last_dir = 1.0 if side < 0 else -1.0  # left of center -> turn left (+)
                xy = lift_bbox_world_xy(ctx, snap, box, use_edge_filter=False)
                if xy is not None:
                    last_xy, last_seen_t, last_box = xy, now, box
                    if predictor is not None:
                        predictor.update(now, xy)
                        vel = predictor.velocity()
                        if vel is not None and lead_gain:
                            xy = (
                                xy[0] + lead_gain * vel[0],
                                xy[1] + lead_gain * vel[1],
                            )
                    target = xy
            if target is None:
                # No fresh point this tick — COAST instead of searching: use the
                # predictor's extrapolation if it has one (target moving), else
                # hold the last seen point for the grace window (target stationary,
                # or a single dropped detection frame). Only past the grace do we
                # actually search.
                pred = predictor.predict(now) if predictor is not None else None
                if pred is not None:
                    target = pred
                elif (
                    last_xy is not None
                    and last_seen_t is not None
                    and (now - last_seen_t) <= lost_grace
                ):
                    target = last_xy
            if target is not None:
                lost = 0
                drove_cmdvel = False
                if cmdvel_on:
                    try:
                        rp = ctx.walkie.status.get_position()
                    except Exception:  # noqa: BLE001
                        rp = None
                    if rp:
                        d = math.hypot(target[0] - rp["x"], target[1] - rp["y"])
                        if not cmdvel_mode and d <= cmdvel_enter:
                            cmdvel_mode = True
                        elif cmdvel_mode and d >= cmdvel_exit:
                            cmdvel_mode = False
                        if cmdvel_mode:
                            if last_goto:
                                # Raw cmd_vel does NOT cancel an active Nav2 goal — its
                                # controller keeps publishing Twists that fight ours.
                                try:
                                    ctx.walkie.nav.cancel()
                                except Exception as exc:  # noqa: BLE001
                                    print(
                                        f"[skills] follow_person: nav.cancel before cmd_vel failed ({exc})"
                                    )
                            drove_cmdvel = approach_point_cmdvel(
                                ctx,
                                *target,
                                stop_distance=follow_distance,
                                pose=rp,
                                **cmdvel_kw,
                            )
                            used_cmdvel = used_cmdvel or drove_cmdvel
                if drove_cmdvel:
                    last_goto = False
                else:  # far target, cmd_vel disabled, or no odom: Nav2 (keeps obstacle avoidance)
                    approach_point(
                        ctx, *target, stop_distance=follow_distance, blocking=False
                    )
                    last_goto = True
            else:
                # Truly lost (no detection, no prediction, grace expired): turn
                # toward the last-seen side until the target reappears. NON-blocking
                # so a single search turn can't freeze the loop (a blocking go_to to
                # the robot's own pose can stall on a nav recovery for seconds).
                lost += 1
                if lost == 1 and on_lost is not None:
                    on_lost()  # nudge once, then keep searching
                if lost >= max_lost:
                    print(
                        "[skills] follow_person: lost the target past the search budget"
                    )
                    reason = "lost"
                    break
                rotate_by(ctx, last_dir * search_step, blocking=False)
            if viz and snap is not None:
                # Green = tracking a fresh detection; yellow = coasting on the last
                # seen box (prediction/grace hold); red = searching, nothing to draw.
                if box is not None:
                    draw_box, color, state = box, (0, 255, 0), "TRACK"
                elif target is not None:
                    draw_box, color, state = last_box, (255, 200, 0), "COAST"
                else:
                    draw_box, color, state = None, (255, 60, 60), "SEARCH"
                tx = f"({target[0]:.2f},{target[1]:.2f})" if target is not None else "-"
                sd = f"{side:+.2f}" if side is not None else "-"
                _draw_follow_viz(
                    snap.img,
                    draw_box,
                    color=color,
                    save_path=viz_path,
                    label=f"{state} side={sd} target={tx} lost={lost}",
                )
            if debug:
                n += 1
                state = "drive" if target is not None else f"search(lost={lost})"
                print(
                    f"[follow_person] {state} box={'Y' if box is not None else 'N'} "
                    f"snap={1e3 * (t_snap - now):.0f}ms select={1e3 * (t_sel - t_snap):.0f}ms "
                    f"lift+nav={1e3 * (time.monotonic() - t_sel):.0f}ms"
                )
                if time.monotonic() - t_log >= 3.0:
                    print(f"[follow_person] {n / (time.monotonic() - t_log):.1f} Hz")
                    n, t_log = 0, time.monotonic()
            # Optional extra inter-command delay; 0 = flat out (snapshot+lift paces).
            (triggered or idle).wait(nav_period)
        if triggered is not None and triggered.is_set():
            reason = "stopped"
            if on_stopped is not None:
                on_stopped()  # speak while the threads wind down
    return reason


class _CvFix(NamedTuple):
    """One identity-confirmed CV sighting published by :class:`_CvFixWorker`."""

    seq: int  # monotonically increasing publish counter (newness check)
    t: float  # time.monotonic() at snapshot time — the fix is already stale by CV latency
    xy: tuple[float, float]  # map-frame lift of the selected box
    box: BBox
    side: float  # box center offset from frame center in [-0.5, 0.5] (search direction)


class _CvFixWorker:
    """Background CV pipeline for the hybrid follow loop: snapshot → *select* →
    map-frame lift, publishing the latest fix into a lock-guarded slot.

    This thread is the ONLY caller of *select* — stateful selectors
    (:class:`tasks.HRI.identity.FollowSelector`) are documented single-threaded,
    and that invariant moves here with them. The loop is self-paced by the CV
    round-trip (a `min_period` floor keeps a stubbed-out camera from busy
    spinning); an empty tick (no snapshot / no box / failed lift) publishes
    nothing, so the main loop keys freshness off ``fix.seq``. It also owns the
    ``HRI_FOLLOW_VIZ`` frame — it holds the snapshots; the lidar loop has none —
    drawing the selected box over the main loop's ``state_label``. Every
    exception is printed and swallowed: a CV glitch degrades to lidar coasting,
    never crashes the follow.
    """

    def __init__(
        self,
        ctx: TaskContext,
        select,
        *,
        min_period: float = 0.0,
        viz: bool = False,
        viz_path: str = "follow_viz.jpg",
    ) -> None:
        self.ctx = ctx
        self.select = select
        self.min_period = min_period
        self.viz = viz
        self.viz_path = viz_path
        self.state_label = "ACQUIRE"  # written by the main loop, drawn on the viz
        self._lock = threading.Lock()
        self._fix: _CvFix | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "_CvFixWorker":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest(self) -> _CvFix | None:
        with self._lock:
            return self._fix

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:
                print(f"[skills] follow CV worker tick failed ({exc})")
            rem = self.min_period - (time.monotonic() - t0)
            if rem > 0:
                self._stop.wait(rem)

    def _tick(self) -> None:
        t = time.monotonic()
        snap = self.ctx.snapshot()
        box = self.select(self.ctx, snap) if snap is not None else None
        if box is not None:
            side = (box[0] + box[2]) / 2 / snap.img.width - 0.5
            xy = lift_bbox_world_xy(self.ctx, snap, box, use_edge_filter=False)
            if xy is not None:
                with self._lock:
                    self._seq += 1
                    self._fix = _CvFix(self._seq, t, xy, box, side)
        if self.viz and snap is not None:
            state = self.state_label
            # Green = the CV pass sees the person this frame; otherwise color by
            # the main loop's state (yellow coasting, red searching/acquiring).
            if box is not None:
                color = (0, 255, 0)
            elif state in ("TRACK", "COAST"):
                color = (255, 200, 0)
            else:
                color = (255, 60, 60)
            _draw_follow_viz(
                snap.img,
                box,
                color=color,
                save_path=self.viz_path,
                label=f"{state} cv_box={'Y' if box is not None else 'N'}",
            )


def _scan_key(scan: dict):
    """Identity of a LaserScan message, for skipping already-processed scans.

    The header stamp when present; otherwise the ``ranges`` list's object id —
    ``Lidar.get_scan`` shallow-copies the cached message, so the list object
    only changes when a new message actually arrived.
    """
    stamp = (scan.get("header") or {}).get("stamp")
    if isinstance(stamp, dict):
        return (stamp.get("sec"), stamp.get("nanosec"))
    return id(scan.get("ranges"))


def _follow_person_lidar(
    ctx: TaskContext,
    select,
    *,
    stopper=None,
    on_warmup=None,
    on_lost=None,
    on_stopped=None,
    follow_distance: float | None = None,
    timeout: float | None = None,
    nav_period: float | None = None,
    search_step_deg: float | None = None,
    max_lost: int | None = None,
    lead_gain: float | None = None,
    predict_on: bool | None = None,
) -> str:
    """Hybrid lidar+CV follow loop (``HRI_FOLLOW_LIDAR=1``).

    The main loop ticks at lidar rate (``HRI_FOLLOW_LIDAR_TICK_SEC``, ~10 Hz)
    and owns all actuation, the stopper, the timeout, and the exit reasons; a
    :class:`_CvFixWorker` runs the slow-trusted CV pipeline in the background.
    Identity comes ONLY from CV: a fresh fix seeds the :class:`AlphaBetaTrack`
    (velocity from the shared :class:`MotionPredictor`), a fix agreeing with the
    track's prediction just confirms it, and a contradicting fix (farther than
    ``HRI_FOLLOW_LIDAR_CV_RESEED_DIST_M``) hard-reseeds — CV wins, by design, so
    a bystander who stole the lidar gate is recovered within one CV round-trip.
    Between fixes each NEW scan is clustered (:func:`cluster_scan`), lifted to
    the map frame with the pose read next to the scan, and the cluster nearest
    the track's prediction inside a staleness-grown gate updates the filter.
    The gate staying empty for ``HRI_FOLLOW_LIDAR_MISS_SEC`` of fresh scans
    drops the track into the legacy coast (predictor extrapolation, then the
    grace hold) and finally the rate-limited rotate-search. ``nav.go_to``
    re-targeting is throttled (``RETARGET_MIN_DELTA_M`` / ``RETARGET_MAX_AGE_SEC``)
    so a 10 Hz tick doesn't thrash the planner with sub-cm goal changes.

    Same contract as :func:`_follow_person_cv` (callbacks, stopper ordering,
    exit reasons); *nav_period* is unused here — the lidar tick paces the loop.
    If no scan arrives at entry the whole call falls back to the CV loop, so
    lidar-less robots and off-robot stubs behave exactly as before.
    """
    if follow_distance is None:
        follow_distance = float(os.getenv("HRI_FOLLOW_DISTANCE_M", "1.0"))
    if timeout is None:
        timeout = float(os.getenv("HRI_FOLLOW_TIMEOUT_SEC", "90"))
    if search_step_deg is None:
        search_step_deg = float(os.getenv("HRI_FOLLOW_SEARCH_STEP_DEG", "10"))
    search_step = math.radians(search_step_deg)
    if max_lost is None:
        max_lost = int(os.getenv("HRI_FOLLOW_MAX_LOST", "8"))
    if lead_gain is None:
        lead_gain = float(os.getenv("HRI_FOLLOW_LEAD_GAIN", "0.5"))
    if predict_on is None:
        predict_on = os.getenv("HRI_FOLLOW_PREDICT", "1").lower() in (
            "1",
            "true",
            "yes",
        )
    lost_grace = float(os.getenv("HRI_FOLLOW_LOST_GRACE_SEC", "1.5"))
    debug = os.getenv("HRI_FOLLOW_TRACK_DEBUG", "0").lower() in ("1", "true", "yes")
    viz = os.getenv("HRI_FOLLOW_VIZ", "0").lower() in ("1", "true", "yes")
    viz_path = os.getenv("HRI_FOLLOW_VIZ_PATH", "follow_viz.jpg")

    # Availability probe: no lidar / no scan -> the legacy loop, byte-for-byte.
    try:
        lidar = ctx.walkie.robot.lidar
        probe = lidar.get_once(timeout=2.0)
    except Exception as exc:
        print(f"[skills] follow_person: lidar unavailable ({exc})")
        probe = None
    if probe is None:
        print("[skills] follow_person: no lidar scan; falling back to the CV-only loop")
        return _follow_person_cv(
            ctx,
            select,
            stopper=stopper,
            on_warmup=on_warmup,
            on_lost=on_lost,
            on_stopped=on_stopped,
            follow_distance=follow_distance,
            timeout=timeout,
            nav_period=nav_period,
            search_step_deg=search_step_deg,
            max_lost=max_lost,
            lead_gain=lead_gain,
            predict_on=predict_on,
        )

    p = LidarFollowParams.from_env()
    predictor = MotionPredictor() if predict_on else None
    track: AlphaBetaTrack | None = None
    idle = threading.Event()  # never set — an interruptible sleep when no stopper
    lost = 0
    last_dir = 1.0  # default search direction (left) until first seen
    last_xy: tuple[float, float] | None = None  # last trusted point, for the grace hold
    last_seen_t: float | None = None
    consumed_seq = 0  # last CV fix folded into the track
    last_scan_key = _scan_key(probe)  # the probe scan predates any track — skip it
    last_goal: tuple[float, float] | None = None  # retarget throttle state
    last_goal_t = 0.0
    last_search_t = (
        0.0  # rate-limits search steps so max_lost keeps its CV-paced meaning
    )
    deadline = time.monotonic() + timeout
    reason = "timeout"
    n, t_log = 0, time.monotonic()

    worker = _CvFixWorker(
        ctx, select, min_period=p.cv_min_period, viz=viz, viz_path=viz_path
    )
    worker.start()
    try:
        if on_warmup is not None:
            on_warmup()  # speak the ack BEFORE the stopper starts (so the mic skips it)
        with stopper if stopper is not None else nullcontext() as listener:
            triggered = getattr(listener, "triggered", None)
            while time.monotonic() < deadline and not (
                triggered is not None and triggered.is_set()
            ):
                now = time.monotonic()
                # 1. Fold in the newest CV fix — the only identity source.
                fix = worker.latest()
                if fix is not None and fix.seq != consumed_seq:
                    consumed_seq = fix.seq
                    last_dir = 1.0 if fix.side < 0 else -1.0
                    last_xy, last_seen_t = fix.xy, fix.t
                    if predictor is not None:
                        predictor.update(fix.t, fix.xy)
                    if now - fix.t <= p.cv_max_age:
                        if track is None:
                            vel = (
                                predictor.velocity() if predictor is not None else None
                            )
                            vx, vy = vel if vel is not None else (0.0, 0.0)
                            track = AlphaBetaTrack(
                                fix.xy[0],
                                fix.xy[1],
                                fix.t,
                                vx,
                                vy,
                                alpha=p.alpha,
                                beta=p.beta,
                                max_speed=p.max_speed,
                            )
                        else:
                            px, py = track.predict(fix.t)
                            if (
                                math.hypot(fix.xy[0] - px, fix.xy[1] - py)
                                > p.cv_reseed_dist
                            ):
                                # Identity contradicts the track: the gate locked
                                # onto someone/something else. CV wins — reseed
                                # (zero velocity; the next scan re-acquires it).
                                track.reseed(fix.xy[0], fix.xy[1], fix.t)
                # 2. Fold in a NEW scan — the fast position carrier.
                try:
                    scan = lidar.get_scan()
                except Exception as exc:
                    print(f"[skills] follow_person: lidar read failed ({exc})")
                    scan = None
                if scan is not None and _scan_key(scan) != last_scan_key:
                    last_scan_key = _scan_key(scan)
                    if track is not None:
                        try:
                            pose = ctx.walkie.status.get_position()
                        except Exception:
                            pose = None
                        if pose:
                            clusters = cluster_scan(scan, p)
                            pts = [sensor_to_map(c.cx, c.cy, pose, p) for c in clusters]
                            pred = track.predict(now)
                            gate = min(
                                p.gate_max,
                                p.gate + p.gate_grow * (now - track.t_accept),
                            )
                            i = associate(pts, pred, gate)
                            if i is not None:
                                track.update(now, *pts[i])
                                if predictor is not None:
                                    predictor.update(now, pts[i])
                                last_xy, last_seen_t = pts[i], now
                                lost = 0
                            elif now - track.t_accept > p.miss_sec:
                                # Gate dry on fresh scans: drop the track into the
                                # grace hold at its last position (fresh window,
                                # so the handoff to coast/search is seamless).
                                last_xy, last_seen_t = (track.x, track.y), now
                                track = None
                # 3. Drive (track > prediction > grace hold) or rotate-search.
                target = None
                state = "SEARCH"
                if track is not None:
                    px, py = track.predict(now)
                    target = (px + lead_gain * track.vx, py + lead_gain * track.vy)
                    state = "TRACK"
                else:
                    pred = predictor.predict(now) if predictor is not None else None
                    if pred is not None:
                        target, state = pred, "COAST"
                    elif (
                        last_xy is not None
                        and last_seen_t is not None
                        and (now - last_seen_t) <= lost_grace
                    ):
                        target, state = last_xy, "COAST"
                worker.state_label = state
                if target is not None:
                    lost = 0
                    moved = last_goal is None or (
                        math.hypot(target[0] - last_goal[0], target[1] - last_goal[1])
                        >= p.retarget_min_delta
                    )
                    if moved or (now - last_goal_t) >= p.retarget_max_age:
                        approach_point(
                            ctx, *target, stop_distance=follow_distance, blocking=False
                        )
                        last_goal, last_goal_t = target, now
                elif now - last_search_t >= p.search_period:
                    last_search_t = now
                    lost += 1
                    if lost == 1 and on_lost is not None:
                        on_lost()  # nudge once, then keep searching
                    if lost >= max_lost:
                        print(
                            "[skills] follow_person: lost the target past the search budget"
                        )
                        reason = "lost"
                        break
                    rotate_by(ctx, last_dir * search_step, blocking=False)
                if debug:
                    n += 1
                    age = f"{now - fix.t:.1f}s" if fix is not None else "-"
                    print(
                        f"[follow_person:lidar] {state} track={'Y' if track is not None else 'N'} "
                        f"cv_age={age} lost={lost}"
                    )
                    if time.monotonic() - t_log >= 3.0:
                        print(
                            f"[follow_person:lidar] {n / (time.monotonic() - t_log):.1f} Hz"
                        )
                        n, t_log = 0, time.monotonic()
                rem = p.tick_sec - (time.monotonic() - now)
                if rem > 0:
                    (triggered or idle).wait(rem)
            if triggered is not None and triggered.is_set():
                reason = "stopped"
                if on_stopped is not None:
                    on_stopped()  # speak while the threads wind down
    finally:
        worker.stop()
    return reason


def follow_person(
    ctx: TaskContext,
    select,
    *,
    stopper=None,
    on_warmup=None,
    on_lost=None,
    on_stopped=None,
    follow_distance: float | None = None,
    timeout: float | None = None,
    nav_period: float | None = None,
    search_step_deg: float | None = None,
    max_lost: int | None = None,
    lead_gain: float | None = None,
    predict_on: bool | None = None,
) -> str:
    """Follow a person, re-targeting nav at their (predicted) position each tick.

    Returns the exit reason: ``'stopped'`` | ``'lost'`` | ``'timeout'``.

    *select* is a callable ``(ctx, snap) -> bbox_xyxy | None`` — e.g.
    :func:`select_largest_person` (follow whoever's closest, for testing) or
    ``identity.select_person_to_follow`` (match an enrolled person by face first,
    attire fallback). *stopper*, when given, is a context manager carrying a
    ``.triggered`` :class:`threading.Event` (a :class:`tasks.skills.CommandListener`);
    it is entered AFTER *on_warmup* — so a background mic listener never
    transcribes the warmup speech — and its event ends the loop the instant it
    is set. *on_warmup* runs once before the loop (e.g. a spoken ack); *on_lost*
    runs once the first time the target is lost; *on_stopped* runs after a
    *stopper*-triggered exit, before teardown, so a closing speech overlaps the
    join cost. Every tunable defaults from the ``HRI_FOLLOW_*`` env vars.

    Two interchangeable engines behind this contract, picked by
    ``HRI_FOLLOW_LIDAR``:

    * ``0`` (default) — :func:`_follow_person_cv`: the camera-paced loop, each
      tick bounded by snapshot + detector round-trips (~1-3 Hz).
    * ``1`` — :func:`_follow_person_lidar`: hybrid tracking. The 2D lidar
      carries the person's position at scan rate (~10 Hz) through a CV-seeded
      cluster track, while the CV selector keeps running in a background worker
      as the slow-trusted identity source (seed / confirm / reseed). Falls back
      to the CV loop automatically when no scan is available.
    """
    if os.getenv("HRI_FOLLOW_LIDAR", "0").lower() in ("1", "true", "yes"):
        return _follow_person_lidar(
            ctx,
            select,
            stopper=stopper,
            on_warmup=on_warmup,
            on_lost=on_lost,
            on_stopped=on_stopped,
            follow_distance=follow_distance,
            timeout=timeout,
            nav_period=nav_period,
            search_step_deg=search_step_deg,
            max_lost=max_lost,
            lead_gain=lead_gain,
            predict_on=predict_on,
        )
    return _follow_person_cv(
        ctx,
        select,
        stopper=stopper,
        on_warmup=on_warmup,
        on_lost=on_lost,
        on_stopped=on_stopped,
        follow_distance=follow_distance,
        timeout=timeout,
        nav_period=nav_period,
        search_step_deg=search_step_deg,
        max_lost=max_lost,
        lead_gain=lead_gain,
        predict_on=predict_on,
    )


def side_relative_to_listener(
    ctx: TaskContext,
    listener_xy: tuple[float, float],
    subject_xy: tuple[float, float],
) -> str | None:
    """Which side ("left"/"right") *subject_xy* sits on from the listener's own
    point of view, ASSUMING the listener faces the robot (people turn toward
    whoever is speaking to them). None when odometry has no fix.

    The listener's facing is taken as robot - listener; the 2D cross product of
    that facing with the listener->subject vector is positive when the subject
    is counter-clockwise of the facing direction, i.e. on the listener's left.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] side_relative_to_listener: odometry unavailable ({exc})")
        return None
    if not pose:
        return None
    fx, fy = pose["x"] - listener_xy[0], pose["y"] - listener_xy[1]
    sx, sy = subject_xy[0] - listener_xy[0], subject_xy[1] - listener_xy[1]
    cross = fx * sy - fy * sx
    if cross == 0:
        return None
    return "left" if cross > 0 else "right"
