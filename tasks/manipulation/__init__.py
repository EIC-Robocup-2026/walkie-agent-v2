"""Shared manipulation layer for the on-robot tasks (Pick and Place, Restaurant, ...).

A package of plain functions over a :class:`~tasks.base.TaskContext` — no state.
Grasp planning is flag-gated (``WALKIE_GRASP_PLANNER``): the default ``graspnet``
path computes EE poses with GraspNet from each object's stored walkie_graphs point
cloud and plans collision-aware against a MoveIt table box; the ``stub`` path keeps
the original heuristic (centroid + a fixed configured orientation) for offline /
no-GPU arm testing. Either way the arm + gripper motion is real.

Robot-wide knobs live in the root ``config.toml`` ``[manipulation]`` block; task
-specific bits (which classes to detect, where to place) are passed in by the caller.

The public surface (re-exported here) is unchanged from the old single-file module,
plus the new submodules (``db``, ``scene``, ``approach``, ``cloud``, ``grasp``) for
callers that want finer control.
"""

from __future__ import annotations

from . import approach, cloud, db, scene
from .approach import aim_camera_at_object, drive_to_object, refine_approach, viz_nav_target
from .execute import (
    PICK_NO_PLAN,
    PICKING,
    perceive_surface,
    pick_object,
    place_at_pose,
    release_in_front,
)
from .grasp import plan_grasp
from .types import (
    BBox,
    DetectedObject,
    GraspPlan,
    Quat,
    Vec3,
    world_to_base,
)

__all__ = [
    # data types
    "DetectedObject",
    "GraspPlan",
    "BBox",
    "Vec3",
    "Quat",
    # geometry
    "world_to_base",
    # planning
    "plan_grasp",
    # perception + motion
    "perceive_surface",
    "pick_object",
    "place_at_pose",
    "release_in_front",
    "PICKING",
    "PICK_NO_PLAN",
    # positioning
    "drive_to_object",
    "aim_camera_at_object",
    "refine_approach",
    "viz_nav_target",
    # submodules
    "db",
    "scene",
    "approach",
    "cloud",
]
