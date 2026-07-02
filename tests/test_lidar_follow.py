"""Unit tests for the hybrid lidar+CV follow (tasks/skills/lidar_track.py + the
_follow_person_lidar loop in tasks/skills/navigation.py).

No robot: part 1 exercises the pure scan-processing logic (clustering,
association, alpha-beta filter, frame lift) on synthetic LaserScans; part 2
wires the whole loop against a fake ctx — a scripted lidar, frozen odometry, a
nav recorder, and a CV selector stub — in the style of
tests/test_creep_base_relative.py. Live-robot behaviour (does the base actually
chase someone?) is verified separately by manual_tests/test_lidar_follow_viz.py.
"""

from __future__ import annotations

import math
import threading
import time

from types import SimpleNamespace

import pytest

from tasks.skills import navigation
from tasks.skills.lidar_track import (
    AlphaBetaTrack,
    LidarFollowParams,
    associate,
    cluster_scan,
    sensor_to_map,
)
from tasks.skills.navigation import follow_person

P = LidarFollowParams()  # code defaults; tests that need knobs build their own


# ---------------------------------------------------------------------------
# Synthetic scans
# ---------------------------------------------------------------------------

def make_scan(blobs, seq=0, *, n=181, inc=math.radians(1.0)):
    """A LaserScan dict with a blob (dist_m, bearing_rad, width_m) per entry.

    181 one-degree beams spanning [-90°, +90°]; everything not covered by a
    blob is inf (invalid). Blobs later in the list overwrite earlier ones.
    """
    angle_min = -math.pi / 2
    ranges = [float("inf")] * n
    for dist, bearing, width in blobs:
        half = max(0, int(round((width / dist) / inc / 2)))
        c = int(round((bearing - angle_min) / inc))
        for j in range(c - half, c + half + 1):
            if 0 <= j < n:
                ranges[j] = dist
    return {
        "header": {"stamp": {"sec": 0, "nanosec": seq}, "frame_id": "laser"},
        "angle_min": angle_min,
        "angle_max": angle_min + inc * (n - 1),
        "angle_increment": inc,
        "range_min": 0.05,
        "range_max": 10.0,
        "ranges": ranges,
    }


# ---------------------------------------------------------------------------
# Part 1: pure logic
# ---------------------------------------------------------------------------

def test_cluster_scan_merges_two_legs_into_one_candidate():
    # Two 0.1 m legs at 1.5 m, centroids 0.25 m apart -> split by the inf gap,
    # rejoined by the merge pass (0.25 < merge_dist 0.40).
    bearing = 0.125 / 1.5  # half the lateral separation, as an angle
    scan = make_scan([(1.5, -bearing, 0.10), (1.5, +bearing, 0.10)])
    out = cluster_scan(scan, P)
    assert len(out) == 1
    c = out[0]
    assert abs(c.cx - 1.5) < 0.05 and abs(c.cy) < 0.05  # centroid between the legs
    assert c.n_points >= 2 * P.cluster_min_pts


def test_cluster_scan_rejects_wall_arc_by_width():
    scan = make_scan([(3.0, 0.5, 3.0)])  # ~1 rad wide arc: furniture/wall, not a person
    assert cluster_scan(scan, P) == []


def test_cluster_scan_rejects_single_beam_speckle():
    scan = make_scan([(2.0, 0.0, 0.0)])  # one lone beam
    assert cluster_scan(scan, P) == []


def test_cluster_scan_tolerates_invalid_beams():
    scan = make_scan([(2.0, 0.0, 0.4)])
    # Poke NaN / None / out-of-range into the blob: the cluster splits there but
    # the merge pass rejoins the halves (same centroid area, < merge_dist).
    c = int(round((0.0 - scan["angle_min"]) / scan["angle_increment"]))
    scan["ranges"][c] = float("nan")
    scan["ranges"][c + 1] = None
    scan["ranges"][c - 1] = 0.01  # below range_min
    out = cluster_scan(scan, P)
    assert len(out) == 1
    assert abs(out[0].cx - 2.0) < 0.05


def test_cluster_scan_splits_on_range_jump():
    # Two person-width blobs at clearly different ranges on adjacent bearings:
    # the range jump must split them, and 1.1 m centroid spacing beats merge_dist.
    scan = make_scan([(1.5, -0.15, 0.3), (2.6, 0.15, 0.3)])
    out = cluster_scan(scan, P)
    assert len(out) == 2


def test_associate_nearest_in_gate_wins():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.2, 0.1)]
    assert associate(pts, (1.1, 0.0), gate=0.5) == 1
    assert associate(pts, (5.0, 5.0), gate=0.5) is None  # empty gate
    assert associate([], (0.0, 0.0), gate=0.5) is None


def test_alpha_beta_track_converges_on_constant_velocity():
    tr = AlphaBetaTrack(0.0, 0.0, 0.0, alpha=0.5, beta=0.3, max_speed=2.0)
    for k in range(1, 51):  # target walks +x at 0.5 m/s, one fix per 0.1 s
        t = k * 0.1
        tr.update(t, 0.5 * t, 0.0)
    assert abs(tr.vx - 0.5) < 0.1 and abs(tr.vy) < 1e-6
    px, py = tr.predict(6.0)  # extrapolate 1 s past the last update
    assert abs(px - 3.0) < 0.2 and abs(py) < 1e-6


def test_alpha_beta_track_clamps_speed_and_reseeds():
    tr = AlphaBetaTrack(0.0, 0.0, 0.0, alpha=0.5, beta=0.3, max_speed=1.5)
    tr.update(0.1, 100.0, 0.0)  # absurd jump: one bad association
    assert math.hypot(tr.vx, tr.vy) <= 1.5 + 1e-9
    tr.reseed(2.0, 3.0, 1.0)
    assert (tr.x, tr.y, tr.vx, tr.vy) == (2.0, 3.0, 0.0, 0.0)
    assert tr.t == tr.t_accept == 1.0
    assert tr.predict(0.5) == (2.0, 3.0)  # dt clamped >= 0


def test_sensor_to_map_applies_offset_then_pose():
    p = LidarFollowParams(offset_x=0.2, offset_y=0.0, offset_yaw=0.0)
    pose = {"x": 1.0, "y": 2.0, "heading": math.pi / 2}
    x, y = sensor_to_map(1.0, 0.0, pose, p)
    # sensor (1,0) -> base (1.2,0) -> rotated 90° -> map (1.0, 3.2)
    assert abs(x - 1.0) < 1e-9 and abs(y - 3.2) < 1e-9

    p_yaw = LidarFollowParams(offset_yaw=math.pi / 2)  # sensor mounted facing left
    pose0 = {"x": 0.0, "y": 0.0, "heading": 0.0}
    x, y = sensor_to_map(1.0, 0.0, pose0, p_yaw)
    assert abs(x) < 1e-9 and abs(y - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Part 2: loop wiring (fake ctx, HRI_FOLLOW_LIDAR=1)
# ---------------------------------------------------------------------------

class FakeNav:
    """Records every go_to(x, y, heading) the loop issues."""

    def __init__(self):
        self.goals: list[tuple[float, float, float]] = []

    def go_to(self, x, y, heading, blocking=True):
        self.goals.append((float(x), float(y), float(heading)))
        return True


class FakeLidar:
    """Serves one scripted scan per get_scan call (advancing stamps), then holds."""

    def __init__(self, scans):
        self.scans = list(scans)
        self.i = 0

    def get_once(self, timeout=10.0):
        return self.scans[0] if self.scans else None

    def get_scan(self):
        if not self.scans:
            return None
        scan = self.scans[min(self.i, len(self.scans) - 1)]
        self.i += 1
        return scan


class FakeSnap:
    """CameraSnapshot stand-in: frozen geometry lifting any bbox to *xy*."""

    has_geometry = True

    def __init__(self, xy):
        self.img = SimpleNamespace(width=640, height=480)
        self._xy = xy

    def bbox_world_xy(self, bbox_xyxy, use_edge_filter=True):
        return self._xy


def _ctx(nav, lidar, snap):
    walkie = SimpleNamespace(
        nav=nav,
        status=SimpleNamespace(get_position=lambda: {"x": 0.0, "y": 0.0, "heading": 0.0}),
        robot=SimpleNamespace(lidar=lidar),
    )
    return SimpleNamespace(walkie=walkie, snapshot=lambda: snap)


@pytest.fixture(autouse=True)
def fast_lidar_follow(monkeypatch):
    monkeypatch.setenv("HRI_FOLLOW_LIDAR", "1")
    monkeypatch.setenv("HRI_FOLLOW_LIDAR_TICK_SEC", "0.01")
    monkeypatch.setenv("HRI_FOLLOW_LIDAR_CV_MIN_PERIOD_SEC", "0.005")
    monkeypatch.setenv("HRI_FOLLOW_LIDAR_SEARCH_PERIOD_SEC", "0.01")
    monkeypatch.setenv("HRI_FOLLOW_VIZ", "0")
    monkeypatch.setenv("HRI_FOLLOW_TRACK_DEBUG", "0")


def test_cv_seed_then_lidar_carries_the_track():
    # CV confirms the person ONCE at (2.0, 0); every scan has a blob at ~(2.05, 0).
    # The lidar must carry the track and the loop must retarget nav ~1 m short.
    scans = [make_scan([(2.05, 0.0, 0.4)], seq=i) for i in range(200)]
    nav, lidar = FakeNav(), FakeLidar(scans)
    calls = {"n": 0}

    def select(ctx, snap):
        calls["n"] += 1
        return (300, 100, 340, 400) if calls["n"] == 1 else None

    ctx = _ctx(nav, lidar, FakeSnap((2.0, 0.0)))
    reason = follow_person(ctx, select, timeout=0.5, max_lost=1000)
    assert reason == "timeout"
    assert nav.goals, "the loop never drove toward the tracked person"
    gx, gy, gh = nav.goals[-1]
    # stop_distance (default 1.0 in-code) short of the ~2.05 m blob, facing it.
    assert 0.8 < gx < 1.4 and abs(gy) < 0.3
    assert abs(gh) < 0.3


def test_stopper_order_and_stopped_exit():
    order = []

    class Stopper:
        triggered = threading.Event()

        def __enter__(self):
            order.append("enter")
            self.triggered.set()  # end the loop on its first check
            return self

        def __exit__(self, *exc):
            return False

    scans = [make_scan([], seq=i) for i in range(50)]
    ctx = _ctx(FakeNav(), FakeLidar(scans), None)
    reason = follow_person(
        ctx,
        lambda c, s: None,
        stopper=Stopper(),
        on_warmup=lambda: order.append("warmup"),
        on_stopped=lambda: order.append("stopped"),
        timeout=5.0,
    )
    assert reason == "stopped"
    assert order == ["warmup", "enter", "stopped"]  # warmup strictly before the stopper


def test_lost_after_search_budget_with_on_lost_once():
    # No CV fix ever, empty scans, prediction off -> straight to the rotate-search.
    scans = [make_scan([], seq=i) for i in range(500)]
    nav = FakeNav()
    ctx = _ctx(nav, FakeLidar(scans), None)
    lost_calls = []
    reason = follow_person(
        ctx,
        lambda c, s: None,
        on_lost=lambda: lost_calls.append(1),
        timeout=10.0,
        max_lost=3,
        predict_on=False,
    )
    assert reason == "lost"
    assert len(lost_calls) == 1  # nudged exactly once
    # Search turns rotate in place: goals stay at the (frozen) robot position.
    assert nav.goals and all(abs(x) < 1e-6 and abs(y) < 1e-6 for x, y, _ in nav.goals)


def test_dispatcher_routes_by_flag(monkeypatch):
    hits = []
    monkeypatch.setattr(navigation, "_follow_person_lidar", lambda *a, **k: hits.append("lidar") or "timeout")
    monkeypatch.setattr(navigation, "_follow_person_cv", lambda *a, **k: hits.append("cv") or "timeout")
    ctx = object()
    monkeypatch.setenv("HRI_FOLLOW_LIDAR", "1")
    assert follow_person(ctx, lambda c, s: None) == "timeout"
    monkeypatch.setenv("HRI_FOLLOW_LIDAR", "0")
    assert follow_person(ctx, lambda c, s: None) == "timeout"
    assert hits == ["lidar", "cv"]


def test_falls_back_to_cv_loop_without_a_scan(monkeypatch):
    sentinel = []
    monkeypatch.setattr(
        navigation, "_follow_person_cv", lambda *a, **k: sentinel.append(1) or "stopped"
    )
    ctx = _ctx(FakeNav(), FakeLidar([]), None)  # lidar present but never publishes
    assert follow_person(ctx, lambda c, s: None, timeout=1.0) == "stopped"
    assert sentinel == [1]


def test_cv_reseed_recovers_a_stolen_gate():
    # The scans track a blob drifting AWAY at (2.05 -> 4) while CV keeps fixing
    # the person at (2.0, 0): once the track has drifted past the reseed
    # distance, the next CV fix must snap it back (goals return near x ~= 1).
    scans = [make_scan([(min(2.05 + 0.05 * i, 4.0), 0.0, 0.4)], seq=i) for i in range(200)]
    nav = FakeNav()
    ctx = _ctx(nav, FakeLidar(scans), FakeSnap((2.0, 0.0)))
    reason = follow_person(
        ctx, lambda c, s: (300, 100, 340, 400), timeout=0.6, max_lost=1000, lead_gain=0.0,
    )
    assert reason == "timeout"
    assert nav.goals
    gx, gy, _ = nav.goals[-1]
    # Without the reseed the final goal would sit ~3 m out (blob at 4 m - 1 m).
    assert gx < 2.0, f"track was never reseeded by CV (last goal x={gx:.2f})"
