"""What the robot is currently carrying — a tiny per-arm blackboard on ``ctx.data``.

The pick skill records what it grabbed here; the place skill reads it back. Kept in
its own module (no grasp/place imports) so both can depend on it without a circular
import. State lives under ``ctx.data["held_objects"]`` as ``{arm: HeldObject}`` —
per arm because Walkie has two and :func:`tasks.skills.grasp.pick_object` already
auto-selects a side. Mirrors the ``setdefault``/``get``/``pop`` pattern
:mod:`tasks.skills.lift` uses for ``people_xy``.

The stored ``grasp_to_surface_offset`` is the load-bearing bit: the height of the
grasp point above the surface the object was sitting on at pick time. Placing on a
new surface at height ``z`` reconstructs the original grasp height as ``z + offset``,
so the object lands the right way up at the right height.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

Vec3 = tuple[float, float, float]


@dataclass
class HeldObject:
    """One object currently grasped in an arm, remembered for placing it back down.

    ``rotation`` is the map-frame grasp orientation (3x3) at pick time, paired with
    ``grasp_heading`` (the base's map-frame yaw at that moment). It's reused as the
    release wrist pose — the only pose a rigidly-held object can reproduce — but the
    arm is base-mounted and the base re-orients between pick and place, so the placer
    reproduces the wrist orientation *relative to the base* (rotating ``rotation`` by
    the heading delta), not its absolute map orientation. ``grasp_xyz`` is kept for
    reference/debug only; the place *position* is always recomputed from a fresh
    surface scan (robust to odometry drift). ``footprint_m`` is the object's XY span
    (for empty-space sizing); ``support_surface_z``/``grasp_to_surface_offset`` may be
    ``None`` when no support surface was found at pick time (caller falls back to a
    default offset).
    """

    label: str
    arm: str
    grasp_xyz: Vec3
    rotation: np.ndarray  # (3, 3) map-frame grasp orientation at pick time
    width: float
    grasp_heading: float | None = None  # base yaw (map frame, rad) at pick time
    footprint_m: float | None = None
    support_surface_z: float | None = None
    grasp_to_surface_offset: float | None = None
    ts: float = 0.0


def record_held_object(
    ctx,
    *,
    label: str,
    arm: str,
    grasp_xyz: Vec3,
    rotation: np.ndarray,
    width: float,
    grasp_heading: float | None = None,
    footprint_m: float | None = None,
    support_surface_z: float | None = None,
    grasp_to_surface_offset: float | None = None,
) -> HeldObject:
    """Remember that *arm* is now holding *label*. Returns the stored record."""
    held = HeldObject(
        label=label,
        arm=arm,
        grasp_xyz=tuple(float(v) for v in grasp_xyz),
        rotation=np.asarray(rotation, dtype=float),
        width=float(width),
        grasp_heading=(None if grasp_heading is None else float(grasp_heading)),
        footprint_m=(None if footprint_m is None else float(footprint_m)),
        support_surface_z=(None if support_surface_z is None else float(support_surface_z)),
        grasp_to_surface_offset=(
            None if grasp_to_surface_offset is None else float(grasp_to_surface_offset)
        ),
        ts=time.time(),
    )
    ctx.data.setdefault("held_objects", {})[arm] = held
    print(
        f"[held] {arm} now holding {label!r} "
        f"(offset={held.grasp_to_surface_offset}, footprint={held.footprint_m})"
    )
    return held


def recall_held_object(ctx, arm: str) -> HeldObject | None:
    """The object held in *arm*, or ``None`` if that arm is empty."""
    return ctx.data.get("held_objects", {}).get(arm)


def held_arms(ctx) -> list[str]:
    """Arms that are currently holding something (in insertion order)."""
    return [a for a, h in ctx.data.get("held_objects", {}).items() if h is not None]


def clear_held_object(ctx, arm: str) -> HeldObject | None:
    """Forget what *arm* was holding (call after a successful release). Returns it."""
    return ctx.data.get("held_objects", {}).pop(arm, None)
