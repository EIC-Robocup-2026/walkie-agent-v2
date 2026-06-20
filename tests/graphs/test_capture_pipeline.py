"""End-to-end capture pipeline through service.ingest_frame (synthetic frames).

Builds real CameraSnapshots over synthetic depth, runs the full deproject →
capture → register → upsert flow against a real GraphMemory wired to a
CaptureStore + BackgroundStore, with stubbed AI-client calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from interfaces.devices.camera import CameraSnapshot

from services.walkie_graphs.background import BackgroundStore
from services.walkie_graphs.capture import CaptureStore
from interfaces.perception.geometry import CameraPose, Intrinsics
from services.walkie_graphs.memory import GraphMemory
from services.walkie_graphs.service import WalkieGraphsService

W, H = 640, 480
OBJ = (260, 180, 380, 300)  # object bbox in pixels (box at z=1 against wall z=2)
PERSON = (440, 120, 560, 420)


def _frame(ts=1.0, cam_offset=(0.0, 0.0, 0.0), structured=False):
    depth = np.full((H, W), 2.0, dtype=np.float32)  # wall at z=2
    if structured:
        # Two smoothly-tilted wall halves whose normals have x / y components —
        # a frontal flat wall constrains only z, and capture ICP needs the scene
        # itself to pin all three translations (like real walls and floors do).
        u = np.arange(W, dtype=np.float32) / W
        v = (np.arange(H, dtype=np.float32) / H)[:, None]
        depth[:, : W // 2] = (2.0 + 1.0 * u)[None, : W // 2]
        depth[:, W // 2 :] = np.broadcast_to(2.0 + 1.0 * v, (H, W))[:, W // 2 :]
    x1, y1, x2, y2 = OBJ
    depth[y1:y2, x1:x2] = 1.0  # object face at z=1
    return CameraSnapshot(
        ts=ts,
        img=Image.new("RGB", (W, H)),
        depth=depth,
        cam=CameraPose(R=np.eye(3), t=np.asarray(cam_offset, dtype=float)),
        intr=Intrinsics(fx=500.0, fy=500.0, cx=W / 2, cy=H / 2, width=W, height=H),
        robot_pose={"x": 0.0, "y": 0.0, "heading": 0.0},
    )


def _mask(bbox):
    m = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = bbox
    m[y1:y2, x1:x2] = True
    return m


def _det(class_name, bbox, conf=0.9):
    # Detections now arrive with the fused per-object caption + CLIP embed already
    # attached (the server-side per_detection pass), mirroring client.DetectedObject.
    return SimpleNamespace(
        class_name=class_name, class_id=0, confidence=conf, bbox=bbox, mask=_mask(bbox),
        caption="a box", embedding=[1.0, 0.0, 0.0],
    )


def _ai():
    # ingest_frame no longer calls the AI client (caption/embed are fused onto the
    # detections upstream); a benign stub keeps the service constructor happy.
    return SimpleNamespace(image=SimpleNamespace())


@pytest.fixture
def svc(tmp_path):
    mem = GraphMemory(
        chroma_dir=None,
        pcds_dir=str(tmp_path / "pcds"),
        thumbs_dir=str(tmp_path / "thumbs"),
        edges_path=str(tmp_path / "edges.json"),
        capture_store=CaptureStore(str(tmp_path / "captures")),
        background=BackgroundStore(str(tmp_path / "bg.npz"), voxel_m=0.05),
    )
    s = WalkieGraphsService(
        walkieAI=_ai(), walkie=None, memory=mem,
        snapshot_path=tmp_path / "perception.json", verbose=False,
    )
    s.viz = None
    return s


def test_ingest_builds_node_segment_and_background(svc):
    result = svc.ingest_frame(_frame(), [_det("box", OBJ)], tick=False)
    assert result[0]["centroid"] is not None
    assert result[0]["centroid"][2] == pytest.approx(1.0, abs=0.05)
    assert svc.memory.count() == 1
    node = svc.memory.all_objects()[0]
    assert len(node.segments) == 1  # segment ref attached, capture persisted
    assert svc.memory.capture_store.load_segment(node.segments[0]) is not None
    # The wall remainder reached the background store; the object region didn't.
    bg = svc.memory.background.points()
    assert len(bg) > 500
    assert bg[:, 2].min() > 1.5  # all background points sit on the z=2 wall


def test_capture_icp_corrects_injected_pose_error(svc):
    pytest.importorskip("open3d")
    svc.capture_icp_max_corr_m = 0.15
    svc.capture_icp_min_fitness = 0.5

    svc.ingest_frame(_frame(ts=1.0, structured=True), [_det("box", OBJ)], tick=False)
    node = svc.memory.all_objects()[0]
    ext_before = node.extent

    # Frame 2: identical scene, but the REPORTED camera pose drifted 5/3/2 cm —
    # every lifted point lands offset. The capture-level registration must solve
    # the correction from the background overlap and keep the object tight.
    svc.ingest_frame(
        _frame(ts=2.0, cam_offset=(0.05, 0.03, 0.02), structured=True),
        [_det("box", OBJ)],
        tick=False,
    )
    assert svc.memory.count() == 1
    node = svc.memory.all_objects()[0]
    assert node.n_obs == 2
    assert node.extent[0] < ext_before[0] + 0.015  # no 5 cm double-exposure
    assert node.extent[2] < ext_before[2] + 0.015
    assert node.centroid[2] == pytest.approx(1.0, abs=0.03)  # back on the true plane


def test_excluded_class_masks_background_but_makes_nothing(svc, tmp_path):
    frame = _frame()
    dets = [_det("box", OBJ), _det("person", PERSON, conf=0.95)]
    result = svc.ingest_frame(frame, dets, tick=False)
    svc._write_snapshot(frame, dets, result)

    # No node, no centroid for the person...
    assert svc.memory.count() == 1
    assert svc.memory.all_objects()[0].class_name == "box"
    assert result[1]["centroid"] is None
    # ...no perception.json record...
    snap = json.loads((tmp_path / "perception.json").read_text())
    assert [o["class"] for o in snap["objects"]] == ["box"]
    # ...and the person's pixels are carved out of the background: with identity
    # pose at depth 2, the person's interior maps to x = (u - cx) * 2 / fx.
    bg = svc.memory.background.points()
    x_lo = (PERSON[0] + 15 - W / 2) * 2.0 / 500.0
    x_hi = (PERSON[2] - 15 - W / 2) * 2.0 / 500.0
    y_lo = (PERSON[1] + 15 - H / 2) * 2.0 / 500.0
    y_hi = (PERSON[3] - 15 - H / 2) * 2.0 / 500.0
    inside = (
        (bg[:, 0] > x_lo) & (bg[:, 0] < x_hi) & (bg[:, 1] > y_lo) & (bg[:, 1] < y_hi)
    )
    assert not inside.any()
