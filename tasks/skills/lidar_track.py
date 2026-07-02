"""Pure 2D-lidar person-track logic for the hybrid follow loop.

The hybrid ``follow_person`` (``HRI_FOLLOW_LIDAR=1``) carries the person's
position between camera fixes on the 2D laser scan: each new scan is segmented
into clusters (jump-distance on adjacent-beam range discontinuity, sensor
frame), the surviving candidate centroids are lifted to the map frame with the
robot pose read next to the scan, and the cluster nearest the track's
constant-velocity prediction — inside a staleness-grown gate — updates an
alpha-beta filter. Identity comes ONLY from the CV selector's map-frame fixes
(seed / reseed); there is deliberately no leg-pattern classifier here.

Everything in this module is robot-free (``math`` + dataclasses only) so the
clustering / association / filter behaviour is unit-testable with synthetic
scans — see tests/test_lidar_follow.py.
"""

from __future__ import annotations

import math
import os

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cluster:
    """One person-sized candidate blob in the SENSOR frame (+x forward, +y left)."""

    cx: float
    cy: float
    n_points: int
    width: float  # first-to-last point distance (m), the blob's physical extent


@dataclass(frozen=True)
class LidarFollowParams:
    """Every ``HRI_FOLLOW_LIDAR_*`` knob, resolved once at follow entry."""

    tick_sec: float = 0.1
    # Fixed lidar-sensor -> base_link 2D transform (m, m, rad).
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_yaw: float = 0.0
    max_range: float = 8.0
    # Jump-distance split: new cluster when |r[i]-r[i-1]| > base + slope*r[i-1].
    jump_base: float = 0.10
    jump_slope: float = 0.04
    cluster_min_pts: int = 3
    cluster_min_width: float = 0.05
    cluster_max_width: float = 0.80
    merge_dist: float = 0.40
    # Association gate: radius grows with time since the last accepted update.
    gate: float = 0.60
    gate_grow: float = 1.0
    gate_max: float = 1.2
    alpha: float = 0.4
    beta: float = 0.2
    max_speed: float = 1.5
    miss_sec: float = 0.8
    cv_reseed_dist: float = 1.0
    cv_max_age: float = 2.0
    search_period: float = 0.7
    retarget_min_delta: float = 0.10
    retarget_max_age: float = 1.0
    cv_min_period: float = 0.05

    @classmethod
    def from_env(cls) -> "LidarFollowParams":
        """Resolve every knob from its ``HRI_FOLLOW_LIDAR_*`` env var."""

        def f(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)))

        return cls(
            tick_sec=f("HRI_FOLLOW_LIDAR_TICK_SEC", cls.tick_sec),
            offset_x=f("HRI_FOLLOW_LIDAR_OFFSET_X", cls.offset_x),
            offset_y=f("HRI_FOLLOW_LIDAR_OFFSET_Y", cls.offset_y),
            offset_yaw=math.radians(f("HRI_FOLLOW_LIDAR_OFFSET_YAW_DEG", 0.0)),
            max_range=f("HRI_FOLLOW_LIDAR_MAX_RANGE_M", cls.max_range),
            jump_base=f("HRI_FOLLOW_LIDAR_JUMP_BASE_M", cls.jump_base),
            jump_slope=f("HRI_FOLLOW_LIDAR_JUMP_SLOPE", cls.jump_slope),
            cluster_min_pts=int(f("HRI_FOLLOW_LIDAR_CLUSTER_MIN_PTS", cls.cluster_min_pts)),
            cluster_min_width=f("HRI_FOLLOW_LIDAR_CLUSTER_MIN_WIDTH_M", cls.cluster_min_width),
            cluster_max_width=f("HRI_FOLLOW_LIDAR_CLUSTER_MAX_WIDTH_M", cls.cluster_max_width),
            merge_dist=f("HRI_FOLLOW_LIDAR_MERGE_DIST_M", cls.merge_dist),
            gate=f("HRI_FOLLOW_LIDAR_GATE_M", cls.gate),
            gate_grow=f("HRI_FOLLOW_LIDAR_GATE_GROW_MPS", cls.gate_grow),
            gate_max=f("HRI_FOLLOW_LIDAR_GATE_MAX_M", cls.gate_max),
            alpha=f("HRI_FOLLOW_LIDAR_ALPHA", cls.alpha),
            beta=f("HRI_FOLLOW_LIDAR_BETA", cls.beta),
            # Track speed clamp shares the predictor's human-walking-pace cap.
            max_speed=f("HRI_FOLLOW_PREDICT_MAX_SPEED", cls.max_speed),
            miss_sec=f("HRI_FOLLOW_LIDAR_MISS_SEC", cls.miss_sec),
            cv_reseed_dist=f("HRI_FOLLOW_LIDAR_CV_RESEED_DIST_M", cls.cv_reseed_dist),
            cv_max_age=f("HRI_FOLLOW_LIDAR_CV_MAX_AGE_SEC", cls.cv_max_age),
            search_period=f("HRI_FOLLOW_LIDAR_SEARCH_PERIOD_SEC", cls.search_period),
            retarget_min_delta=f("HRI_FOLLOW_LIDAR_RETARGET_MIN_DELTA_M", cls.retarget_min_delta),
            retarget_max_age=f("HRI_FOLLOW_LIDAR_RETARGET_MAX_AGE_SEC", cls.retarget_max_age),
            cv_min_period=f("HRI_FOLLOW_LIDAR_CV_MIN_PERIOD_SEC", cls.cv_min_period),
        )


def _valid_range(r, rmin: float, rmax: float) -> float | None:
    """*r* as a usable float range, or None (NaN/inf/None/out-of-bounds)."""
    if r is None:
        return None
    try:
        rf = float(r)
    except (TypeError, ValueError):
        return None
    if math.isnan(rf) or math.isinf(rf) or rf < rmin or rf > rmax:
        return None
    return rf


def cluster_scan(scan: dict, p: LidarFollowParams) -> list[Cluster]:
    """Segment a LaserScan dict into person-sized candidate clusters.

    Walks the beams in scan order (the only frame where "adjacent" means
    anything): an invalid beam (None/NaN/inf/out-of-range/beyond ``max_range``)
    or a range jump above ``jump_base + jump_slope * r_prev`` closes the current
    cluster. Legs commonly split a person into two nearby clusters, so a merge
    pass then joins CONSECUTIVE clusters whose centroids are closer than
    ``merge_dist`` — a geometry heuristic, not a leg classifier. Finally,
    candidates must have >= ``cluster_min_pts`` beams and a physical width
    (first-to-last point distance) inside [min_width, max_width]; that rejects
    single-beam speckle and wall/furniture arcs. Centroids are SENSOR-frame.
    """
    ranges = scan.get("ranges") or []
    angle = scan.get("angle_min", 0.0)
    inc = scan.get("angle_increment", 0.0)
    rmin = scan.get("range_min", 0.0) or 0.0
    rmax = min(float(scan.get("range_max") or p.max_range), p.max_range)

    # 1. Split into raw point runs on invalid beams / range jumps.
    runs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    prev_r: float | None = None
    for r in ranges:
        theta = angle
        angle += inc
        rf = _valid_range(r, rmin, rmax)
        if rf is None:
            if cur:
                runs.append(cur)
                cur = []
            prev_r = None
            continue
        if prev_r is not None and abs(rf - prev_r) > p.jump_base + p.jump_slope * prev_r:
            if cur:
                runs.append(cur)
            cur = []
        cur.append((rf * math.cos(theta), rf * math.sin(theta)))
        prev_r = rf
    if cur:
        runs.append(cur)

    def centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
        return (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))

    # 2. Merge consecutive runs whose centroids are close (two legs -> one blob).
    merged: list[list[tuple[float, float]]] = []
    for run in runs:
        if merged:
            mx, my = centroid(merged[-1])
            cx, cy = centroid(run)
            if math.hypot(cx - mx, cy - my) <= p.merge_dist:
                merged[-1] = merged[-1] + run
                continue
        merged.append(run)

    # 3. Size/width filters -> Cluster records.
    out: list[Cluster] = []
    for pts in merged:
        if len(pts) < p.cluster_min_pts:
            continue
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        width = math.hypot(x1 - x0, y1 - y0)
        if not (p.cluster_min_width <= width <= p.cluster_max_width):
            continue
        cx, cy = centroid(pts)
        out.append(Cluster(cx=cx, cy=cy, n_points=len(pts), width=width))
    return out


def sensor_to_map(x: float, y: float, pose: dict, p: LidarFollowParams) -> tuple[float, float]:
    """Lift a sensor-frame point to the map frame via the (skew-accepted) pose.

    Applies the fixed sensor->base_link 2D offset first, then rotates by the
    robot heading and translates by its map position. *pose* is a
    ``status.get_position()`` dict read adjacent to the scan — there is no
    stamped TF sync, so a rotating base smears the point slightly; the
    association gate absorbs that.
    """
    # sensor -> base_link
    c, s = math.cos(p.offset_yaw), math.sin(p.offset_yaw)
    bx = p.offset_x + c * x - s * y
    by = p.offset_y + s * x + c * y
    # base_link -> map
    ch, sh = math.cos(pose["heading"]), math.sin(pose["heading"])
    return pose["x"] + ch * bx - sh * by, pose["y"] + sh * bx + ch * by


def associate(
    points: list[tuple[float, float]], pred: tuple[float, float], gate: float
) -> int | None:
    """Index of the point nearest *pred* within *gate* m, or None (empty gate)."""
    best_i, best_d = None, gate
    for i, (x, y) in enumerate(points):
        d = math.hypot(x - pred[0], y - pred[1])
        if d <= best_d:
            best_i, best_d = i, d
    return best_i


@dataclass
class AlphaBetaTrack:
    """Constant-velocity alpha-beta track of the followed person, map frame.

    Deliberately not a Kalman filter — nothing downstream consumes a covariance,
    and this stays dependency-free like :class:`MotionPredictor`. ``t`` is the
    time of the current state estimate; ``t_accept`` the last time a real
    measurement was folded in (the association gate and the miss timeout key
    off it). Velocity is clamped to *max_speed* so one bad association can't
    fling the prediction across the room.
    """

    x: float
    y: float
    t: float
    vx: float = 0.0
    vy: float = 0.0
    alpha: float = 0.4
    beta: float = 0.2
    max_speed: float = 1.5
    t_accept: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.t_accept = self.t

    def predict(self, t: float) -> tuple[float, float]:
        """Extrapolated position at *t* (dt clamped >= 0; state unchanged)."""
        dt = max(0.0, t - self.t)
        return self.x + self.vx * dt, self.y + self.vy * dt

    def update(self, t: float, zx: float, zy: float) -> None:
        """Fold in an accepted measurement (zx, zy) taken at time *t*."""
        dt = max(1e-3, t - self.t)  # floor so a same-tick update can't blow up beta/dt
        px, py = self.x + self.vx * dt, self.y + self.vy * dt
        rx, ry = zx - px, zy - py
        self.x, self.y = px + self.alpha * rx, py + self.alpha * ry
        self.vx += self.beta * rx / dt
        self.vy += self.beta * ry / dt
        speed = math.hypot(self.vx, self.vy)
        if speed > self.max_speed:
            self.vx, self.vy = (
                self.vx / speed * self.max_speed,
                self.vy / speed * self.max_speed,
            )
        self.t = self.t_accept = t

    def reseed(self, x: float, y: float, t: float, vx: float = 0.0, vy: float = 0.0) -> None:
        """Hard-reset to a trusted (CV) position — identity contradicts the track."""
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.t = self.t_accept = t
