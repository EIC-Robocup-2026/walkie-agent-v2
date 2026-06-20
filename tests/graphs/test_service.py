"""Service-level detection filters + maintenance cadence (stubs, no robot/server)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from services.walkie_graphs.service import WalkieGraphsService


class _StubMemory:
    """Records which periodic-maintenance methods fired and when."""

    def __init__(self):
        self.calls = []
        self.flagged: list[list[str]] = []

    def derive_relations(self):
        self.calls.append("relations")

    def prune(self):
        self.calls.append("prune")

    def denoise_nodes(self):
        self.calls.append("denoise")

    def merge_overlapping_nodes(self):
        self.calls.append("merge")

    def evict_stale_provisional(self, now_ts):
        self.calls.append("evict")

    def flush_pcds(self):
        self.calls.append("pcd_flush")

    def refine_nodes(self, limit=3):
        self.calls.append("refine_nodes")

    def flag_for_refine(self, node_ids):
        self.flagged.append(list(node_ids))

    def carve_free_space(self, depth, intr, pose, **kw):
        self.calls.append("carve")
        return {"bg_carved": 0, "nodes_carved": 0, "nodes_evicted": 0}


@pytest.fixture
def svc():
    s = WalkieGraphsService(walkieAI=None, walkie=None, memory=_StubMemory(), verbose=False)
    s.viz = None
    # Cadence tests assert exact synchronous call counts against the stub memory, so
    # run heavy maintenance inline (the async worker would race the assertions). The
    # inline path runs the SAME due-list selector, just calling synchronously — async
    # dispatch/de-collision get their own dedicated tests below.
    s._async_maintenance = False
    return s


def _det(mask_area, bbox):
    mask = np.zeros((100, 100), dtype=bool)
    mask.flat[:mask_area] = True
    return SimpleNamespace(mask=mask, bbox=bbox)


def test_size_filters_noop_by_default(svc):
    # defaults: max_bbox_area_ratio=1.0, min_mask_area_px=0 → keep everything
    assert svc._passes_size_filters(_det(5, (0, 0, 100, 100)), img_area=10000)


def test_rejects_whole_frame_box(svc):
    svc.max_bbox_area_ratio = 0.9
    big = _det(5000, (0, 0, 100, 100))  # bbox area 10000 == whole image
    small = _det(5000, (0, 0, 50, 50))  # bbox area 2500
    assert not svc._passes_size_filters(big, img_area=10000)
    assert svc._passes_size_filters(small, img_area=10000)


def test_rejects_tiny_mask(svc):
    svc.min_mask_area_px = 64
    assert not svc._passes_size_filters(_det(10, (0, 0, 50, 50)), img_area=10000)
    assert svc._passes_size_filters(_det(200, (0, 0, 50, 50)), img_area=10000)


def test_maintenance_cadence_staggered(svc):
    svc.relation_every_n = 5
    svc.denoise_every_n = 20
    svc.merge_every_n = 20
    svc.ghost_every_n = 20
    for _ in range(45):
        svc._maybe_tick(True)
    calls = svc.memory.calls
    # relations+prune at 5,10,...,45 (9 times each)
    assert calls.count("relations") == 9
    # denoise at tick 20, 40 (t % 20 == 0)
    assert calls.count("denoise") == 2
    # merge at 21, 41 (t % 20 == 1)
    assert calls.count("merge") == 2
    # evict at 22, 42 (t % 20 == 2)
    assert calls.count("evict") == 2


def test_tick_false_runs_nothing(svc):
    svc._maybe_tick(False)
    assert svc.memory.calls == []


def test_capture_and_background_cadences(svc):
    calls = svc.memory.calls
    svc.memory.capture_store = SimpleNamespace(
        flush=lambda: calls.append("cap_flush"), gc=lambda: calls.append("cap_gc")
    )
    svc.memory.background = SimpleNamespace(save=lambda: calls.append("bg_save"))
    svc.pcd_flush_every_n = 5
    svc.bg_save_every_n = 20
    for _ in range(40):
        svc._maybe_tick(True)
    # capture flush+gc piggyback every pcd flush (ticks 5,10,...,40)
    assert calls.count("cap_flush") == 8
    assert calls.count("cap_gc") == 8
    # background saves on its own offset cadence (ticks 19, 39)
    assert calls.count("bg_save") == 2


def test_detect_prompts_include_exclude_classes_for_masking(monkeypatch):
    # Excluded classes are prompted for (masking-only) when detection is scoped...
    monkeypatch.setenv("WALKIE_GRAPHS_INTERESTED_CLASSES", "box, cup")
    monkeypatch.setenv("WALKIE_GRAPHS_EXCLUDE_CLASSES", "person")
    s = WalkieGraphsService(walkieAI=None, walkie=None, memory=_StubMemory(), verbose=False)
    assert s.detect_prompts == ["box", "cup", "person"]
    # ...but never narrow an unscoped (detect-everything) configuration.
    monkeypatch.setenv("WALKIE_GRAPHS_INTERESTED_CLASSES", "")
    s = WalkieGraphsService(walkieAI=None, walkie=None, memory=_StubMemory(), verbose=False)
    assert s.detect_prompts == []


def test_fixed_rate_wait(svc):
    svc.interval = 3.0
    # cycle faster than the interval → wait the remainder
    assert svc._wait_after(1.0) == pytest.approx(2.0)
    assert svc._wait_after(0.0) == pytest.approx(3.0)
    # cycle at/over the interval → no wait (observe immediately)
    assert svc._wait_after(3.0) == 0.0
    assert svc._wait_after(5.0) == 0.0


def test_mask_subtract_and_crop_margin_defaults(svc):
    # mask subtraction defaults ON (CG always applies it); crop margin matches CG's 20px.
    assert svc.mask_subtract is True
    assert svc.crop_margin_px == 20


def test_flying_pixel_cleanup_defaults(svc):
    # depth-bleed cleanup defaults ON: erode 2px + reject depth jumps over 5cm.
    assert svc.mask_erode_px == 2
    assert svc.depth_edge_thresh_m == pytest.approx(0.05)


def test_density_cleanup_defaults_off_in_code(svc):
    # The depth-relative edge allowance, SOR, and the trusted-range gate default OFF
    # in code (config.toml turns them on), so the unit-test lift path keeps the
    # original behaviour.
    assert svc.depth_edge_rel == pytest.approx(0.0)
    assert svc.sor_k == 0
    assert svc.sor_std_ratio == pytest.approx(2.0)
    assert svc.max_depth_m == 0.0
    assert svc.bg_max_depth_m == 0.0


def test_bg_max_depth_follows_max_depth(monkeypatch):
    monkeypatch.setenv("WALKIE_GRAPHS_MAX_DEPTH_M", "4.0")
    s = WalkieGraphsService(walkieAI=None, walkie=None, memory=_StubMemory(), verbose=False)
    assert s.max_depth_m == 4.0
    assert s.bg_max_depth_m == 4.0  # inherits unless overridden
    monkeypatch.setenv("WALKIE_GRAPHS_BG_MAX_DEPTH_M", "5.0")
    s = WalkieGraphsService(walkieAI=None, walkie=None, memory=_StubMemory(), verbose=False)
    assert s.bg_max_depth_m == 5.0


def test_refine_cadence_fires_at_offset_6(svc):
    svc.refine_every_n = 20
    svc.refine_limit = 3
    for _ in range(46):
        svc._maybe_tick(True)
    # offset 6 in period 20: fires at tick 26, 46
    assert svc.memory.calls.count("refine_nodes") == 2


def test_refine_cadence_off_by_default(svc):
    for _ in range(100):
        svc._maybe_tick(True)
    assert svc.memory.calls.count("refine_nodes") == 0


def test_flag_for_refine_called_on_rejected_registration(monkeypatch):
    """When capture ICP is enabled but the registration is rejected, touched nodes are
    flagged for the per-object refine pass."""
    from types import SimpleNamespace

    from services.walkie_graphs.capture import Capture
    from services.walkie_graphs.service import WalkieGraphsService

    stub = _StubMemory()
    stub.pop_perf_stats = lambda: {}
    stub.batch_writes = __import__("contextlib").contextmanager(lambda: (yield))
    stub.upsert = lambda det: SimpleNamespace(id="node-a")
    stub.background = SimpleNamespace(add=lambda pts: None)
    stub.capture_store = SimpleNamespace(save=lambda c: None)

    import numpy as _np

    svc = WalkieGraphsService(walkieAI=None, walkie=None, memory=stub, verbose=False)
    svc.capture_icp_max_corr_m = 0.25  # enabled
    svc.refine_every_n = 0  # don't run refine in _maybe_tick
    # Permissive lift filters so the synthetic detection survives to an upsert.
    svc.mask_erode_px = 0
    svc.min_points = 1
    svc.min_confidence = 0.0
    svc.depth_edge_thresh_m = 0.0
    svc.sor_k = 0

    # The capture passed downstream keeps the lifted segments but registration was
    # rejected (icp_accepted False) — exactly the case that should flag for refine.
    def _reject(cap):
        cap.icp_accepted = False
        cap.icp_fitness = 0.1
        return cap

    monkeypatch.setattr(svc, "_register_capture", _reject)

    from services.walkie_graphs.service import FrameSnapshot

    img = SimpleNamespace(size=(40, 40), crop=lambda box: SimpleNamespace(size=(20, 20)))
    depth = _np.ones((40, 40), dtype=_np.float32)
    from interfaces.perception.geometry import CameraPose, Intrinsics

    cam = CameraPose(R=_np.eye(3), t=_np.zeros(3))
    intr = Intrinsics(fx=50.0, fy=50.0, cx=20.0, cy=20.0, width=40, height=40)
    frame = FrameSnapshot(ts=1.0, img=img, depth=depth, cam=cam, intr=intr, robot_pose=None)
    det = SimpleNamespace(
        class_name="cup", class_id=0, confidence=0.9,
        bbox=(0, 0, 20, 20), mask=_np.ones((40, 40), dtype=bool),
        caption="a cup", embedding=[0.1, 0.1, 0.1],
    )
    monkeypatch.setattr(svc, "_maybe_tick", lambda *a, **kw: None)

    svc.ingest_frame(frame, [det], tick=False)
    assert stub.flagged and stub.flagged[0] == ["node-a"]


# ---------------------------------------------------------------------------
# Free-space carving cadence + trusted-pose gating
# ---------------------------------------------------------------------------
def _carve_frame_and_capture(accepted=True, correction=None):
    from types import SimpleNamespace

    import numpy as _np

    from interfaces.perception.geometry import CameraPose, Intrinsics

    cam = CameraPose(R=_np.eye(3), t=_np.zeros(3))
    frame = SimpleNamespace(
        depth=_np.ones((8, 8), dtype=_np.float32),
        intr=Intrinsics(fx=50.0, fy=50.0, cx=4.0, cy=4.0, width=8, height=8),
        cam=cam,
        robot_pose=None,
    )
    capture = SimpleNamespace(
        cam=cam,
        correction=_np.eye(4) if correction is None else correction,
        icp_accepted=accepted,
    )
    return frame, capture


def test_carve_cadence_fires_at_offset_3_when_accepted(svc):
    svc.carve_every_n = 5
    svc.capture_icp_max_corr_m = 0.25  # registration on; capture is accepted
    frame, capture = _carve_frame_and_capture(accepted=True)
    for _ in range(25):
        svc._maybe_tick(True, touched=[], frame=frame, capture=capture)
    # t>=5 and t%5==3 → ticks 8, 13, 18, 23
    assert svc.memory.calls.count("carve") == 4


def test_carve_skipped_on_rejected_registration(svc):
    svc.carve_every_n = 5
    svc.capture_icp_max_corr_m = 0.25  # registration on, but the solve was rejected
    frame, capture = _carve_frame_and_capture(accepted=False)
    for _ in range(25):
        svc._maybe_tick(True, touched=[], frame=frame, capture=capture)
    assert svc.memory.calls.count("carve") == 0


def test_carve_runs_when_registration_disabled(svc):
    svc.carve_every_n = 5
    svc.capture_icp_max_corr_m = 0.0  # registration off → pose is raw but trusted
    frame, capture = _carve_frame_and_capture(accepted=False)
    for _ in range(25):
        svc._maybe_tick(True, touched=[], frame=frame, capture=capture)
    assert svc.memory.calls.count("carve") == 4


def test_carve_off_by_default(svc):
    frame, capture = _carve_frame_and_capture(accepted=True)
    for _ in range(25):
        svc._maybe_tick(True, touched=[], frame=frame, capture=capture)
    assert svc.memory.calls.count("carve") == 0


def test_carve_skipped_without_geometry(svc):
    svc.carve_every_n = 5
    svc.capture_icp_max_corr_m = 0.0
    # No frame/capture (the no-geometry _maybe_tick path) → never carves.
    for _ in range(25):
        svc._maybe_tick(True, touched=[])
    assert svc.memory.calls.count("carve") == 0


# ---------------------------------------------------------------------------
# De-collision + async maintenance dispatch
# ---------------------------------------------------------------------------
def test_heavy_ops_decollide_carve_priority(svc):
    """Coprime carve/refine periods collide on tick 13 (13%5==3 and 13%7==6); the
    selector runs AT MOST ONE heavy op per tick, carve first (it needs the frustum)."""
    svc.carve_every_n = 5  # carve due at ticks 8, 13, 18, 23
    svc.refine_every_n = 7  # refine due at ticks 13, 20
    svc.capture_icp_max_corr_m = 0.0  # registration off → pose trusted, carve allowed
    frame, capture = _carve_frame_and_capture(accepted=True)
    for _ in range(25):
        svc._maybe_tick(True, touched=[], frame=frame, capture=capture)
    # Carve fires on all four of its ticks (priority 0).
    assert svc.memory.calls.count("carve") == 4
    # Refine is due at 13 and 20, but loses the tick-13 collision to carve → fires once.
    assert svc.memory.calls.count("refine_nodes") == 1


def test_async_maintenance_dispatches_to_worker_and_drains(svc):
    """With async on, a due heavy op runs on the worker (not inline); the executor
    shutdown in stop_and_join drains it so the side effect lands."""
    svc._async_maintenance = True
    svc.refine_every_n = 7  # refine due at tick 13 (offset 6)
    for _ in range(13):
        svc._maybe_tick(True, touched=[])
    # Submitted to the worker; drain it (stop_and_join calls executor.shutdown(wait=True)).
    svc.stop_and_join(timeout=0)
    assert svc.memory.calls.count("refine_nodes") == 1


# ---------------------------------------------------------------------------
# walkie-sdk integration: intrinsics from get_intrinsics, pose from optical TF
# ---------------------------------------------------------------------------
class _FakeCamera:
    def get_intrinsics(self):
        return {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}


class _FakeTransform:
    def __init__(self, tf):
        self.tf = tf
        self.calls = []

    def lookup(self, source, target, timeout=1.0):
        self.calls.append((source, target))
        return self.tf


class _FakeRobot:
    def __init__(self, tf):
        self.camera = _FakeCamera()
        self.transform = _FakeTransform(tf)


class _FakeWalkie:
    def __init__(self, tf):
        self.robot = _FakeRobot(tf)


def _svc_with_walkie(tf):
    w = _FakeWalkie(tf)
    s = WalkieGraphsService(walkieAI=None, walkie=w, memory=_StubMemory(), verbose=False)
    return s, w


def test_intrinsics_from_sdk_scaled_to_depth():
    from interfaces.devices.camera import intrinsics_for

    _, w = _svc_with_walkie(None)
    intr = intrinsics_for(w, 640, 480)
    assert (intr.fx, intr.cx, intr.cy) == pytest.approx((500.0, 320.0, 240.0))
    # different depth resolution → intrinsics rescaled
    half = intrinsics_for(w, 320, 240)
    assert (half.fx, half.cx) == pytest.approx((250.0, 160.0))


def test_camera_pose_from_optical_tf():
    from interfaces.devices.camera import camera_pose

    tf = {
        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},  # identity
    }
    _, w = _svc_with_walkie(tf)
    pose = camera_pose(w)
    assert np.allclose(pose.R, np.eye(3))
    assert np.allclose(pose.t, [1.0, 2.0, 3.0])
    # it looked up MAP_FRAME -> the optical camera frame
    assert w.robot.transform.calls == [("map", "zed_head_left_camera_frame_optical")]


def test_camera_pose_none_when_lookup_fails():
    from interfaces.devices.camera import camera_pose

    _, w = _svc_with_walkie(None)  # lookup returns None
    assert camera_pose(w) is None


# ---------------------------------------------------------------------------
# Motion gate: skip frames captured while the robot/head was moving
# ---------------------------------------------------------------------------
def _tf(x=0.0, y=0.0, z=0.0, qz=0.0, qw=1.0):
    return {
        "position": {"x": x, "y": y, "z": z},
        "quaternion": {"x": 0.0, "y": 0.0, "z": qz, "w": qw},
    }


def _frame_with_cam(walkie):
    from interfaces.devices.camera import camera_pose
    from services.walkie_graphs.service import FrameSnapshot

    cam = camera_pose(walkie)
    return FrameSnapshot(ts=0.0, img=None, depth=None, cam=cam, intr=None, robot_pose=None)


def test_motion_gate_disabled_by_default():
    s, w = _svc_with_walkie(_tf())
    frame = _frame_with_cam(w)
    w.robot.transform.tf = _tf(x=1.0)  # robot drove a metre — but gate is off
    assert s.motion_max_trans_m == 0.0 and s.motion_max_rot_deg == 0.0
    assert s._moved_during(frame) is False


def test_motion_gate_trips_on_translation_and_rotation():
    s, w = _svc_with_walkie(_tf())
    s.motion_max_trans_m, s.motion_max_rot_deg = 0.03, 2.0
    frame = _frame_with_cam(w)
    # still → clean
    assert s._moved_during(frame) is False
    # drove 10 cm during the capture→detect window → gated
    w.robot.transform.tf = _tf(x=0.10)
    assert s._moved_during(frame) is True
    # rotated ~11.5° (qz=sin(θ/2)≈0.1) → gated
    w.robot.transform.tf = _tf(qz=0.1, qw=0.995)
    assert s._moved_during(frame) is True
    # tiny jitter (5 mm) stays under the bound → clean
    w.robot.transform.tf = _tf(x=0.005)
    assert s._moved_during(frame) is False


def test_motion_gate_trips_when_pose_vanishes():
    s, w = _svc_with_walkie(_tf())
    s.motion_max_trans_m = 0.03
    frame = _frame_with_cam(w)
    w.robot.transform.tf = None  # TF lookup starts failing mid-tick
    assert s._moved_during(frame) is True


def test_default_camera_frame_is_optical():
    from interfaces.devices.camera import camera_pose

    _, w = _svc_with_walkie(None)
    camera_pose(w)  # lookup fails (tf None) but records the requested frames
    assert w.robot.transform.calls == [("map", "zed_head_left_camera_frame_optical")]


# ---------------------------------------------------------------------------
# Live perception.json snapshot (written by the loop, from the ingest result)
# ---------------------------------------------------------------------------
def test_snapshot_path_defaults_to_none(svc):
    # default: no snapshot — keeps the manual observe()/ingest paths side-effect-free
    assert svc.snapshot_path is None


def test_snapshot_path_stored_as_path(tmp_path):
    out = tmp_path / "perception.json"
    s = WalkieGraphsService(
        walkieAI=None, walkie=None, memory=_StubMemory(), snapshot_path=out, verbose=False
    )
    assert s.snapshot_path == out


def test_write_snapshot_uses_frame_time_pose_and_ts(tmp_path):
    import json

    from services.walkie_graphs.service import FrameSnapshot

    out = tmp_path / "perception.json"
    s = WalkieGraphsService(
        walkieAI=None, walkie=None, memory=_StubMemory(), snapshot_path=out, verbose=False
    )
    img = SimpleNamespace(size=(640, 480))
    dets = [SimpleNamespace(class_name="cup", bbox=(300, 220, 340, 260), confidence=0.9)]
    result = {0: {"centroid": (1.0, 2.0, 3.0), "caption": "a cup"}}
    # The heading + ts come from the frame snapshot (capture time), not a live read.
    frame = FrameSnapshot(
        ts=123.0, img=img, depth=None, cam=None, intr=None, robot_pose={"heading": 0.0}
    )
    s._write_snapshot(frame, dets, result)

    snap = json.loads(out.read_text())
    assert "people" not in snap  # pose/people are gone from the perception path
    assert snap["ts"] == 123.0  # capture-time stamp, not write time
    assert snap["objects"][0]["class"] == "cup"
    assert snap["objects"][0]["position_3d"] == [1.0, 2.0, 3.0]
    assert snap["objects"][0]["caption"] == "a cup"


def test_robot_pose_swallows_status_failure():
    # a status read that raises must not propagate out of the capture — the
    # snapshot is still built, with robot_pose=None (heading degrades downstream)
    import numpy as _np

    from interfaces.devices.camera import CameraSnapshot

    def _boom():
        raise RuntimeError("no pose")

    walkie = SimpleNamespace(
        robot=SimpleNamespace(
            camera=SimpleNamespace(
                get_depth=lambda: _np.ones((4, 4), dtype=_np.float32),
                get_intrinsics=lambda: None,
            ),
            transform=SimpleNamespace(lookup=lambda *a, **k: None),
        ),
        camera=SimpleNamespace(capture_pil=lambda: SimpleNamespace(size=(4, 4))),
        status=SimpleNamespace(get_position=_boom),
    )
    snap = CameraSnapshot.capture(walkie)
    assert snap is not None and snap.robot_pose is None


def test_write_snapshot_degrades_when_robot_pose_none(tmp_path):
    # robot_pose=None (status read failed at capture) → heading degrades to 0.0, still writes
    import json

    from services.walkie_graphs.service import FrameSnapshot

    out = tmp_path / "perception.json"
    s = WalkieGraphsService(
        walkieAI=None, walkie=None, memory=_StubMemory(), snapshot_path=out, verbose=False
    )
    img = SimpleNamespace(size=(640, 480))
    frame = FrameSnapshot(ts=1.0, img=img, depth=None, cam=None, intr=None, robot_pose=None)
    s._write_snapshot(frame, [], {})  # must not raise
    snap = json.loads(out.read_text())
    assert snap["objects"] == []
