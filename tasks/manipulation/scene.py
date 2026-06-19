"""MoveIt planning-scene helpers for collision-aware pick/place.

Thin wrappers over ``walkie.robot.arm``'s scene services so the executor reads as
the user's sequence: add the table box, attach the grasped object to the gripper,
release it on place, allow the gripper through the octomap during the final
descent. All best-effort — a scene call failing logs and returns False rather
than aborting the grasp (the arm can still plan against the octomap alone).
"""

from __future__ import annotations

import os

from tasks.base import TaskContext

from . import db


def _arm(ctx: TaskContext):
    return ctx.walkie.robot.arm


def _fallback_size() -> list[float]:
    raw = os.getenv("WALKIE_TABLE_FALLBACK_SIZE", "0.8,1.2")
    parts = [p.strip() for p in raw.split(",")]
    try:
        return [float(parts[0]), float(parts[1])]
    except Exception:  # noqa: BLE001
        return [0.8, 1.2]


def add_surface_collision(
    ctx: TaskContext,
    *,
    node=None,
    pose: list[float] | None = None,
    size: list[float] | None = None,
    frame: str = "map",
) -> bool:
    """Add the table/shelf collision box to the planning scene.

    Prefers an explicit ``pose``/``size`` (``pose=[x,y,top_z,yaw]``,
    ``size=[depth_x,width_y]``); otherwise derives them from a walkie_graphs
    surface *node*'s aabb (:func:`db.node_table_box`). When neither yields a
    footprint, falls back to ``WALKIE_TABLE_FALLBACK_SIZE`` at the node's top_z
    (or skips if there's nothing to anchor to). Returns whether a box was set.
    """
    if pose is None and node is not None:
        box = db.node_table_box(node)
        if box is not None:
            pose, size = box
    if pose is None:
        print("[manipulation.scene] add_surface_collision — no pose/node; skipping table box")
        return False
    if size is None:
        size = _fallback_size()
    try:
        return bool(_arm(ctx).set_table(enable=True, pose=pose, size=size, frame=frame))
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.scene] set_table failed ({exc})")
        return False


def remove_surface_collision(ctx: TaskContext) -> bool:
    """Disable the explicit table box (leave other scene objects untouched)."""
    try:
        return bool(_arm(ctx).set_table(enable=False))
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.scene] disable table failed ({exc})")
        return False


def attach_grasped_object(ctx: TaskContext) -> bool:
    """Arm the scene so the NEXT gripper close attaches the grasp box to the hand.

    Sets ``grasp_scene_action=grasp``; the commander reads it on the next gripper
    command, so call this right before ``arm.grasp()`` / closing.
    """
    try:
        return bool(_arm(ctx).set_grasp_scene_action("grasp"))
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.scene] attach (grasp_scene_action=grasp) failed ({exc})")
        return False


def release_object_scene(ctx: TaskContext) -> bool:
    """Arm the scene so the NEXT gripper open detaches + removes the grasp box.

    Sets ``grasp_scene_action=place``; call right before opening to release.
    """
    try:
        return bool(_arm(ctx).set_grasp_scene_action("place"))
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.scene] release (grasp_scene_action=place) failed ({exc})")
        return False


def neutral_scene(ctx: TaskContext) -> bool:
    """Leave the planning scene untouched on the next gripper command."""
    try:
        return bool(_arm(ctx).set_grasp_scene_action("none"))
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.scene] neutral (grasp_scene_action=none) failed ({exc})")
        return False


def allow_gripper_vs_octomap(ctx: TaskContext, allow: bool) -> bool:
    """Let the gripper links ignore the octomap during the final grasp descent.

    Grasping inside sensed voxels (the object's own points) otherwise blocks
    planning. Re-enforce (``allow=False``) once clear of the surface.
    """
    try:
        return bool(_arm(ctx).set_allow_gripper_vs_octomap(allow))
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.scene] allow_gripper_vs_octomap({allow}) failed ({exc})")
        return False


def clear_scene(ctx: TaskContext) -> bool:
    """Detach anything held and remove all world collision objects (keep octomap)."""
    try:
        return bool(_arm(ctx).clear_collision_objects())
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.scene] clear_collision_objects failed ({exc})")
        return False
