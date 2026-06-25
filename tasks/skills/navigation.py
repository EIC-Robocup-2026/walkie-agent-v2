"""Base motion, look-at, approach, and the person-follow loop.

Moved out of tasks/HRI/skills.py into the shared tasks.skills package.
"""

from __future__ import annotations

import math
import os
import threading
import time

from collections.abc import Sequence
from contextlib import nullcontext

from tasks.base import TaskContext

from .geometry import BBox
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
        ctx.walkie.nav.go_to(gx, gy, heading, blocking=False)
    except Exception as exc:
        print(f"[skills] approach_point: non-blocking go_to failed ({exc})")
        return False
    if settle > 0:
        time.sleep(settle)  # let the base make progress before re-targeting
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
    """Follow a person by re-targeting nav at their (predicted) position each tick.

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
        predict_on = os.getenv("HRI_FOLLOW_PREDICT", "1").lower() in ("1", "true", "yes")
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

    predictor = MotionPredictor() if predict_on else None
    idle = threading.Event()  # never set — an interruptible sleep when no stopper
    lost = 0
    last_dir = 1.0  # default search direction (left) until first seen
    last_xy: tuple[float, float] | None = None  # last good lifted point, for the grace hold
    last_seen_t: float | None = None
    last_box: BBox | None = None  # last selected box, drawn dim while coasting (viz)
    deadline = time.monotonic() + timeout
    reason = "timeout"
    n, t_log = 0, time.monotonic()
    if on_warmup is not None:
        on_warmup()  # speak the ack BEFORE the stopper starts (so the mic skips it)
    with (stopper if stopper is not None else nullcontext()) as listener:
        triggered = getattr(listener, "triggered", None)
        while time.monotonic() < deadline and not (triggered is not None and triggered.is_set()):
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
                            xy = (xy[0] + lead_gain * vel[0], xy[1] + lead_gain * vel[1])
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
                approach_point(ctx, *target, stop_distance=follow_distance, blocking=False)
            else:
                # Truly lost (no detection, no prediction, grace expired): turn
                # toward the last-seen side until the target reappears. NON-blocking
                # so a single search turn can't freeze the loop (a blocking go_to to
                # the robot's own pose can stall on a nav recovery for seconds).
                lost += 1
                if lost == 1 and on_lost is not None:
                    on_lost()  # nudge once, then keep searching
                if lost >= max_lost:
                    print("[skills] follow_person: lost the target past the search budget")
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
                    snap.img, draw_box, color=color, save_path=viz_path,
                    label=f"{state} side={sd} target={tx} lost={lost}",
                )
            if debug:
                n += 1
                state = "drive" if target is not None else f"search(lost={lost})"
                print(f"[follow_person] {state} box={'Y' if box is not None else 'N'} "
                      f"snap={1e3 * (t_snap - now):.0f}ms select={1e3 * (t_sel - t_snap):.0f}ms "
                      f"lift+nav={1e3 * (time.monotonic() - t_sel):.0f}ms")
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
