"""Grasp planning: GraspNet (switchable input source) with a heuristic stub fallback.

``plan_grasp`` dispatches on ``WALKIE_GRASP_PLANNER``:

  graspnet — run GraspNet and turn the best returned pose (position + quaternion +
             approach waypoint) into a GraspPlan. The GraspNet *input* is chosen by
             ``WALKIE_GRASP_SOURCE``:
               cloud — the object's stored walkie_graphs cloud -> ``grasp.from_cloud``
               pos   — the stored cloud + a live crop, antipodal-validated -> ``grasp.from_pos``
               mask  — a fresh live segmentation mask over the camera view -> ``grasp.from_mask``
  stub     — the original heuristic: hand at the object's centroid with a fixed
             configured orientation. No GPU / no DB needed (offline arm testing).

The GraspNet path degrades to the stub whenever its input or the grasp server is
unavailable, so a run never hard-fails on a missing dependency.
"""

from __future__ import annotations

import os

from tasks.base import TaskContext

from . import db
from .types import BBox, DetectedObject, GraspPlan, _arm_frame, _parse3, world_to_base


def _envf(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _planner() -> str:
    return os.getenv("WALKIE_GRASP_PLANNER", "graspnet").strip().lower()


def _source() -> str:
    return os.getenv("WALKIE_GRASP_SOURCE", "cloud").strip().lower()


# --- shared result handling -------------------------------------------------
def _result_to_plan(result: dict | None) -> GraspPlan | None:
    """Best grasp from a GraspNet result (cloud/pos/mask all share this shape)."""
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


# --- GraspNet sources -------------------------------------------------------
def _grasp_from_cloud(ctx: TaskContext, graphs, node) -> dict | None:
    """GraspNet from the object's stored DB point cloud."""
    cloud = db.object_cloud_pc2(graphs, node, frame_id="map")
    if cloud is None:
        print(f"[manipulation.grasp] no stored cloud for node {node.id}; cloud source unavailable")
        return None
    try:
        return ctx.walkie.robot.grasp.from_cloud(
            cloud,
            score_threshold=_envf("WALKIE_GRASP_FROM_CLOUD_SCORE_TH", "0.0"),
            max_grasps=int(_envf("WALKIE_GRASP_FROM_CLOUD_MAX", "20")),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.grasp] grasp.from_cloud failed ({exc})")
        return None


def _grasp_from_pos(ctx: TaskContext, graphs, node) -> dict | None:
    """GraspNet from the stored DB cloud + a live crop, antipodal-validated."""
    cloud = db.object_cloud_pc2(graphs, node, frame_id="map")
    if cloud is None:
        print(f"[manipulation.grasp] no stored cloud for node {node.id}; pos source unavailable")
        return None
    try:
        return ctx.walkie.robot.grasp.from_pos(
            cloud,
            crop_margin_m=_envf("WALKIE_GRASP_FROM_POS_CROP_MARGIN_M", "0.0"),
            score_threshold=_envf("WALKIE_GRASP_FROM_CLOUD_SCORE_TH", "0.0"),
            max_grasps=int(_envf("WALKIE_GRASP_FROM_CLOUD_MAX", "20")),
            antipodal_weight=_envf("WALKIE_GRASP_FROM_POS_ANTIPODAL_W", "0.0"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.grasp] grasp.from_pos failed ({exc})")
        return None


def _xyxy_to_cxcywh(bbox: BBox) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def _live_detection_for(ctx: TaskContext, obj: DetectedObject):
    """Best live detection (with mask) matching *obj*'s class, or None.

    Re-detects on a fresh snapshot scoped to the object's class, then picks the
    detection whose bbox center is nearest *obj*'s (falling back to highest
    confidence) so a multi-object view still grasps the intended one.
    """
    snap = ctx.snapshot()
    if snap is None:
        return None
    try:
        dets = ctx.walkieAI.image.detect(snap.img, prompts=[obj.class_name], return_mask=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.grasp] live detect for mask source failed ({exc})")
        return None
    if not dets:
        return None
    ox1, oy1, ox2, oy2 = obj.bbox_xyxy
    ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0

    def _key(d):
        x1, y1, x2, y2 = d.bbox
        dist = ((x1 + x2) / 2.0 - ocx) ** 2 + ((y1 + y2) / 2.0 - ocy) ** 2
        return (dist, -(d.confidence or 0.0))

    # No usable obj bbox -> fall back to most confident detection.
    if (ox1, oy1, ox2, oy2) == (0, 0, 0, 0):
        return max(dets, key=lambda d: d.confidence or 0.0)
    return min(dets, key=_key)


def _grasp_from_mask(ctx: TaskContext, obj: DetectedObject) -> dict | None:
    """GraspNet from a fresh live segmentation mask over the camera view."""
    det = _live_detection_for(ctx, obj)
    if det is None:
        print("[manipulation.grasp] no live detection; mask source unavailable")
        return None
    from walkie_sdk.utils.converters import numpy_to_mono8_image

    mask_img = None
    if getattr(det, "mask", None) is not None:
        try:
            mask_img = numpy_to_mono8_image(det.mask)
        except Exception as exc:  # noqa: BLE001
            print(f"[manipulation.grasp] mask encode failed ({exc}); using bbox region")
    bbox = _xyxy_to_cxcywh(det.bbox)
    try:
        return ctx.walkie.robot.grasp.from_mask(
            mask=mask_img,
            bbox=bbox,  # fallback region when the mask is empty/omitted
            num_frames=int(_envf("WALKIE_GRASP_MASK_NUM_FRAMES", "0")),
            score_threshold=_envf("WALKIE_GRASP_FROM_CLOUD_SCORE_TH", "0.0"),
            max_grasps=int(_envf("WALKIE_GRASP_FROM_CLOUD_MAX", "20")),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.grasp] grasp.from_mask failed ({exc})")
        return None


# --- GraspNet planner -------------------------------------------------------
def _plan_grasp_graspnet(ctx: TaskContext, obj: DetectedObject) -> GraspPlan | None:
    """Plan via GraspNet using the configured source. None -> caller falls back."""
    source = _source()
    if source == "mask":
        return _result_to_plan(_grasp_from_mask(ctx, obj))

    # cloud / pos both need the object's stored DB cloud.
    graphs = getattr(ctx, "graphs", None)
    if graphs is None:
        print(f"[manipulation.grasp] no ctx.graphs; {source!r} source unavailable")
        return None
    node = graphs.get(obj.node_id) if obj.node_id else db.resolve_object_node(graphs, obj.class_name)
    if node is None:
        print(f"[manipulation.grasp] no DB node for {obj.class_name!r}; {source!r} source unavailable")
        return None
    if source == "pos":
        return _result_to_plan(_grasp_from_pos(ctx, graphs, node))
    return _result_to_plan(_grasp_from_cloud(ctx, graphs, node))


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

    With ``WALKIE_GRASP_PLANNER=graspnet`` (default) it runs the GraspNet source
    selected by ``WALKIE_GRASP_SOURCE`` (cloud|mask|pos) and, if that yields
    nothing, falls back to the heuristic stub. With ``=stub`` it uses only the
    heuristic. Returns None when neither can plan.
    """
    if _planner() == "stub":
        return _plan_grasp_stub(ctx, obj)
    plan = _plan_grasp_graspnet(ctx, obj)
    if plan is not None:
        return plan
    print("[manipulation.grasp] GraspNet path unavailable; falling back to stub planner")
    return _plan_grasp_stub(ctx, obj)
