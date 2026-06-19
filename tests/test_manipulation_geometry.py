"""Pure-geometry unit tests for the shared manipulation grasp planner.

No robot, no AI server — just the map->base transform and the grasp-pose stub,
exercised against a fake TaskContext (only ``current_pose`` is needed).
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from tasks.manipulation import (
    DetectedObject,
    plan_grasp,
    world_to_base,
)
from tasks.manipulation.cloud import numpy_to_pointcloud2
from tasks.manipulation.db import node_table_box
from walkie_sdk.utils.converters import parse_point_cloud_xyz


class FakeCtx:
    """Minimal stand-in: plan_grasp/world_to_base only touch current_pose()."""

    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self._pose = {"x": x, "y": y, "heading": heading}

    def current_pose(self):
        return self._pose


def _obj(world_xyz):
    return DetectedObject(
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        class_name="cup",
        confidence=0.9,
        world_xyz=world_xyz,
    )


# --- world_to_base ----------------------------------------------------------
def test_world_to_base_identity():
    # Robot at the origin facing +x: base frame == map frame.
    ctx = FakeCtx(0.0, 0.0, 0.0)
    assert world_to_base(ctx, (1.0, 2.0, 0.5)) == pytest.approx((1.0, 2.0, 0.5))


def test_world_to_base_translation_only():
    ctx = FakeCtx(1.0, 1.0, 0.0)
    assert world_to_base(ctx, (3.0, 1.0, 0.4)) == pytest.approx((2.0, 0.0, 0.4))


def test_world_to_base_rotation_90deg():
    # Robot facing +y (heading 90deg). A map point 2 m straight ahead of it
    # (0, 2) lands at base x=+2 (forward), y=0. z passes through unchanged.
    ctx = FakeCtx(0.0, 0.0, math.pi / 2)
    bx, by, bz = world_to_base(ctx, (0.0, 2.0, 0.8))
    assert (bx, by, bz) == pytest.approx((2.0, 0.0, 0.8), abs=1e-9)


# --- plan_grasp -------------------------------------------------------------
def test_plan_grasp_none_without_3d():
    assert plan_grasp(FakeCtx(), _obj(None)) is None


def test_plan_grasp_topdown_base_frame(monkeypatch):
    monkeypatch.setenv("WALKIE_ARM_FRAME", "base_footprint")
    monkeypatch.setenv("WALKIE_GRASP_APPROACH", "top_down")
    monkeypatch.setenv("WALKIE_GRASP_RPY_TOPDOWN", "0.1,0.2,0.3")
    monkeypatch.setenv("WALKIE_GRASP_Z_OFFSET_M", "0.05")
    # Robot at origin facing +x -> base == map; object 0.6 m ahead, 0.8 m high.
    plan = plan_grasp(FakeCtx(), _obj((0.6, 0.0, 0.8)))
    assert plan is not None
    assert plan.frame_id == "base_footprint"
    assert plan.approach == "top_down"
    assert plan.position == pytest.approx((0.6, 0.0, 0.85))  # z + offset
    assert plan.rotation == pytest.approx((0.1, 0.2, 0.3))


def test_plan_grasp_map_frame_passthrough(monkeypatch):
    monkeypatch.setenv("WALKIE_ARM_FRAME", "map")
    monkeypatch.setenv("WALKIE_GRASP_APPROACH", "front")
    monkeypatch.setenv("WALKIE_GRASP_RPY_FRONT", "0.0,0.0,0.0")
    monkeypatch.setenv("WALKIE_GRASP_Z_OFFSET_M", "0.0")
    # In map frame the robot pose is ignored — centroid passes straight through.
    plan = plan_grasp(FakeCtx(5.0, 5.0, 1.0), _obj((1.0, 2.0, 3.0)))
    assert plan is not None
    assert plan.frame_id == "map"
    assert plan.approach == "front"
    assert plan.position == pytest.approx((1.0, 2.0, 3.0))


# --- numpy_to_pointcloud2 (round-trips through the SDK parser) ---------------
def test_pointcloud2_roundtrip(monkeypatch):
    monkeypatch.setenv("WALKIE_PC2_DATA_ENCODING", "bytes")
    pts = np.random.default_rng(0).uniform(-2.0, 2.0, size=(64, 3)).astype(np.float32)
    cloud = numpy_to_pointcloud2(pts, frame_id="map")
    assert cloud["header"]["frame_id"] == "map"
    assert cloud["width"] == 64 and cloud["height"] == 1 and cloud["point_step"] == 12
    out = parse_point_cloud_xyz(cloud)
    assert out is not None
    assert out.shape == (64, 3)
    assert out == pytest.approx(pts, abs=1e-5)


def test_pointcloud2_base64_roundtrip(monkeypatch):
    monkeypatch.setenv("WALKIE_PC2_DATA_ENCODING", "base64")
    pts = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32)
    out = parse_point_cloud_xyz(numpy_to_pointcloud2(pts))
    assert out == pytest.approx(pts, abs=1e-5)


# --- node_table_box (aabb -> set_table pose/size) ---------------------------
@dataclass
class _FakeNode:
    centroid: tuple
    aabb_min: tuple
    aabb_max: tuple


def test_node_table_box_from_aabb():
    node = _FakeNode(
        centroid=(1.0, 0.5, 0.4),
        aabb_min=(0.5, 0.0, 0.0),
        aabb_max=(1.5, 1.0, 0.75),
    )
    pose, size = node_table_box(node)
    # pose = [center_x, center_y, top_z, yaw]
    assert pose == pytest.approx([1.0, 0.5, 0.75, 0.0])
    # size = [depth_x, width_y]
    assert size == pytest.approx([1.0, 1.0])


def test_node_table_box_none_for_none():
    assert node_table_box(None) is None
