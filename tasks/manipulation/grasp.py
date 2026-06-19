"""Grasp planning: GraspNet (DB cloud) with a heuristic stub fallback.

``plan_grasp`` dispatches on ``WALKIE_GRASP_PLANNER``:

  graspnet — pack the object's stored walkie_graphs cloud into a PointCloud2 and
             call ``walkie.robot.grasp.from_cloud``; the best returned pose
             (position + quaternion + approach waypoint) becomes the GraspPlan.
  stub     — the original heuristic: hand at the object's centroid with a fixed
             configured orientation. No GPU / no DB needed (offline arm testing).

The GraspNet path degrades to the stub whenever the DB cloud or the grasp server
is unavailable, so a run never hard-fails on a missing dependency.
"""

from __future__ import annotations

import os

from tasks.base import TaskContext

from . import db
from .types import DetectedObject, GraspPlan, _arm_frame, _parse3, world_to_base


def _envf(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _planner() -> str:
    return os.getenv("WALKIE_GRASP_PLANNER", "graspnet").strip().lower()


# --- GraspNet (DB cloud) ----------------------------------------------------
def _plan_grasp_graspnet(ctx: TaskContext, obj: DetectedObject) -> GraspPlan | None:
    """Plan via GraspNet from the object's stored DB cloud. None -> caller falls back."""
    graphs = getattr(ctx, "graphs", None)
    if graphs is None:
        print("[manipulation.grasp] no ctx.graphs; cannot run GraspNet path")
        return None
    node = graphs.get(obj.node_id) if obj.node_id else db.resolve_object_node(graphs, obj.class_name)
    if node is None:
        print(f"[manipulation.grasp] no DB node for {obj.class_name!r}; GraspNet path unavailable")
        return None
    cloud = db.object_cloud_pc2(graphs, node, frame_id="map")
    if cloud is None:
        print(f"[manipulation.grasp] no stored cloud for node {node.id}; GraspNet path unavailable")
        return None
    try:
        result = ctx.walkie.robot.grasp.from_cloud(
            cloud,
            score_threshold=_envf("WALKIE_GRASP_FROM_CLOUD_SCORE_TH", "0.0"),
            max_grasps=int(_envf("WALKIE_GRASP_FROM_CLOUD_MAX", "20")),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.grasp] grasp.from_cloud failed ({exc})")
        return None
    if not result or not result.get("success") or not result.get("grasps"):
        msg = (result or {}).get("message", "no grasps")
        print(f"[manipulation.grasp] GraspNet returned no usable grasp ({msg})")
        return None
    best = result["grasps"][0]  # ranked best-first by the server
    frame = result.get("planning_frame") or "base_footprint"
    pos = tuple(float(v) for v in best["position"])
    quat = tuple(float(v) for v in best["orientation"])
    approach = best.get("approach_position")
    approach_pos = tuple(float(v) for v in approach) if approach else None
    return GraspPlan(
        position=pos,
        rotation=(0.0, 0.0, 0.0),  # unused when quaternion is set
        frame_id=frame,
        quaternion=quat,
        approach_position=approach_pos,
        width=best.get("width"),
        score=best.get("score"),
    )


# --- heuristic stub (offline / no-GPU fallback) -----------------------------
def _plan_grasp_stub(ctx: TaskContext, obj: DetectedObject) -> GraspPlan | None:
    """Heuristic planner: object 3D centroid -> hand pose with a configured RPY.

    Does NOT run a grasp network. Places the end-effector at the object's centroid
    with a config orientation (top-down by default, or a front/horizontal
    approach). Returns None when the object was never lifted to 3D.
    """
    if obj.world_xyz is None:
        return None
    frame = _arm_frame()
    centroid = obj.world_xyz if frame == "map" else world_to_base(ctx, obj.world_xyz)
    cx, cy, cz = centroid
    approach = os.getenv("WALKIE_GRASP_APPROACH", "top_down").strip().lower()
    z_off = _envf("WALKIE_GRASP_Z_OFFSET_M", "0.0")
    if approach == "front":
        rot = _parse3(os.getenv("WALKIE_GRASP_RPY_FRONT", "-0.8,0.0,-1.5708"))
        return GraspPlan((cx, cy, cz + z_off), rot, frame_id=frame, approach="front")
    rot = _parse3(os.getenv("WALKIE_GRASP_RPY_TOPDOWN", "-2.623,-0.033,-1.468"))
    return GraspPlan((cx, cy, cz + z_off), rot, frame_id=frame, approach="top_down")


def plan_grasp(ctx: TaskContext, obj: DetectedObject) -> GraspPlan | None:
    """Plan a grasp for *obj* using the configured planner, stub-fallback on miss.

    With ``WALKIE_GRASP_PLANNER=graspnet`` (default) it tries the DB-cloud GraspNet
    path and, if that yields nothing, falls back to the heuristic stub. With
    ``=stub`` it uses only the heuristic. Returns None when neither can plan.
    """
    if _planner() == "stub":
        return _plan_grasp_stub(ctx, obj)
    plan = _plan_grasp_graspnet(ctx, obj)
    if plan is not None:
        return plan
    print("[manipulation.grasp] GraspNet path unavailable; falling back to stub planner")
    return _plan_grasp_stub(ctx, obj)
