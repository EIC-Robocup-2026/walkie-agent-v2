"""Unit tests for the held-object blackboard (tasks/skills/held.py).

No robot — a FakeCtx with a plain ``data`` dict exercises record/recall/clear and
per-arm isolation, the state the place skill reads back from the pick skill.
"""

import numpy as np

from tasks.skills.held import (
    HeldObject,
    clear_held_object,
    held_arms,
    recall_held_object,
    record_held_object,
)


class FakeCtx:
    """Minimal stand-in: held.py only touches ctx.data."""

    def __init__(self):
        self.data = {}


def test_record_and_recall_roundtrip():
    ctx = FakeCtx()
    rot = np.eye(3)
    held = record_held_object(
        ctx, label="red can", arm="left", grasp_xyz=(1.0, 2.0, 0.8),
        rotation=rot, width=0.06, footprint_m=0.08,
        support_surface_z=0.7, grasp_to_surface_offset=0.1,
    )
    assert isinstance(held, HeldObject)
    got = recall_held_object(ctx, "left")
    assert got is held
    assert got.label == "red can"
    assert got.grasp_xyz == (1.0, 2.0, 0.8)
    assert got.grasp_to_surface_offset == 0.1
    assert got.footprint_m == 0.08
    assert got.ts > 0


def test_recall_empty_arm_is_none():
    ctx = FakeCtx()
    assert recall_held_object(ctx, "left") is None
    assert held_arms(ctx) == []


def test_per_arm_isolation():
    ctx = FakeCtx()
    record_held_object(ctx, label="can", arm="left", grasp_xyz=(0, 0, 0),
                       rotation=np.eye(3), width=0.05)
    record_held_object(ctx, label="box", arm="right", grasp_xyz=(0, 0, 0),
                       rotation=np.eye(3), width=0.05)
    assert recall_held_object(ctx, "left").label == "can"
    assert recall_held_object(ctx, "right").label == "box"
    assert set(held_arms(ctx)) == {"left", "right"}


def test_clear_releases_only_that_arm():
    ctx = FakeCtx()
    record_held_object(ctx, label="can", arm="left", grasp_xyz=(0, 0, 0),
                       rotation=np.eye(3), width=0.05)
    record_held_object(ctx, label="box", arm="right", grasp_xyz=(0, 0, 0),
                       rotation=np.eye(3), width=0.05)
    cleared = clear_held_object(ctx, "left")
    assert cleared.label == "can"
    assert recall_held_object(ctx, "left") is None
    assert recall_held_object(ctx, "right").label == "box"
    assert held_arms(ctx) == ["right"]


def test_optional_fields_default_none():
    ctx = FakeCtx()
    record_held_object(ctx, label="mug", arm="left", grasp_xyz=(0, 0, 0),
                       rotation=np.eye(3), width=0.05)
    held = recall_held_object(ctx, "left")
    assert held.footprint_m is None
    assert held.support_surface_z is None
    assert held.grasp_to_surface_offset is None
