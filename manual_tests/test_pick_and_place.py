"""Manual, on-robot smoke test for the GraspNet pick pipeline.

Needs the robot + walkie-ai-server (with the grasp service) up. The cloud/pos
sources also need the object already scanned into the walkie_graphs store; the
mask source needs only the live camera. It resolves the object, prints the
GraspNet result for the selected source, and (optionally) executes one real pick.

Run as a module so the repo root is on sys.path:

    # A/B the three GraspNet inputs against the same object:
    SOURCE=cloud uv run python -m manual_tests.test_pick_and_place "water bottle"
    SOURCE=mask  uv run python -m manual_tests.test_pick_and_place "water bottle"
    SOURCE=pos   uv run python -m manual_tests.test_pick_and_place "water bottle"
    # Then perform the pick with the chosen source. With EXECUTE the tester steps
    # through every arm/base motion (Enter=do, s=skip, q=abort); CONFIRM=0 disables.
    SOURCE=pos EXECUTE=1 uv run python -m manual_tests.test_pick_and_place "water bottle"

RViz markers are published by default (VIZ=0 to disable): the ranked grasp
candidates as arrows (best green -> red), the best approach waypoint as a sphere
with a score label (in base_footprint), and the table collision box as a
translucent cube (in map). Open RViz and add a MarkerArray display on
'walkie/viz_markers'; MoveIt's own displays then show the planned arm motion and
the attached/table collision objects when EXECUTE=1.

A live mask overlay of the selected object is also written to MASK_VIZ_PATH
(default pick_mask_viz.jpg; MASK_VIZ=0 to disable): the chosen detection's
segmentation mask painted green over the camera view (literally what SOURCE=mask
feeds GraspNet; a live preview of the resolved object for cloud/pos), with other
same-class detections dimmed red and bboxes/labels on top. Open the JPG to
confirm the right object/mask was picked before executing.

Deliberately outside tests/ (pyproject testpaths) so pytest never collects it —
it drives real hardware.
"""

from __future__ import annotations

import math
import os
import sys

from dotenv import load_dotenv

from client import WalkieAIClient
from tasks.base import TaskContext
from tasks.common import initialize_graphs, initialize_llm_model, initialize_robot, load_task_config
from tasks.manipulation import db, grasp, pick_object
from tasks.manipulation.types import DetectedObject

# RViz marker shape constants (re-exported by the SDK).
from walkie_sdk import ARROW, CUBE, SPHERE, TEXT_VIEW_FACING

_VIZ_NS = "pnp_test"


def _fmt(v):
    """Compact repr for arm-call logging: round floats, recurse into lists/tuples."""
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return repr(v)


def _rpy_to_quat(rpy) -> list[float]:
    """Roll,pitch,yaw (rad) -> [x,y,z,w], ZYX intrinsic (yaw*pitch*roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r / 2.0), math.sin(r / 2.0)
    cp, sp = math.cos(p / 2.0), math.sin(p / 2.0)
    cy, sy = math.cos(y / 2.0), math.sin(y / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _ee_target(name, args, kwargs):
    """Absolute EE pose target of a pose command, as (position, quat, frame).

    Covers go_to_pose_quat (position, quat) and go_to_pose (position, rpy).
    Returns None for relative/other calls (go_to_pose_relative is an offset from
    the live EE pose, so it can't be drawn absolutely without FK).
    """
    if name == "go_to_pose_quat" and len(args) >= 2:
        return list(args[0]), [float(v) for v in args[1]], kwargs.get("frame_id", "map")
    if name == "go_to_pose" and len(args) >= 2:
        return list(args[0]), _rpy_to_quat(args[1]), kwargs.get("frame_id", "map")
    return None


class _LoggingArm:
    """Transparent proxy over an ArmGroup that prints + draws every actuator call.

    Every method invocation (gripper/grasp/go_to_pose*/go_to_home, …) is logged
    with its args and return value before/after it runs, so the tester sees the
    exact arm commands the GraspNet executor issues. For absolute pose commands
    (go_to_pose_quat / go_to_pose) it also publishes an RViz marker at the target
    EE pose — a magenta arrow + sphere + step label on walkie/viz_markers — so the
    tester can see where the hand is being sent. Non-callable attributes pass
    straight through. Best-effort: a raising call is logged then re-raised so the
    executor's own error handling is unchanged; viz failures never abort a move.
    """

    def __init__(self, inner, *, label: str, viz=None):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_viz", viz)
        object.__setattr__(self, "_draw_n", 0)

    def _draw_target(self, name, position, quat, frame) -> None:
        viz = object.__getattribute__(self, "_viz")
        label = object.__getattribute__(self, "_label")
        if viz is None or os.getenv("VIZ", "1").strip().lower() not in ("1", "true", "yes"):
            return
        n = object.__getattribute__(self, "_draw_n")
        object.__setattr__(self, "_draw_n", n + 1)
        base = 500 + 10 * n  # past grasps(100s)/approach(200)/table(300)/nav(400s)
        pos = [float(v) for v in position]
        try:
            viz.draw_marker(
                position=pos, quaternion=quat, frame_id=frame, marker_type=ARROW,
                scale=[0.12, 0.02, 0.02], color=[1.0, 0.0, 1.0, 0.9],
                marker_id=base, ns=f"{_VIZ_NS}/ee_target",
            )
            viz.draw_marker(
                position=pos, frame_id=frame, marker_type=SPHERE,
                scale=[0.03, 0.03, 0.03], color=[1.0, 0.0, 1.0, 0.9],
                marker_id=base + 1, ns=f"{_VIZ_NS}/ee_target",
            )
            viz.draw_marker(
                position=[pos[0], pos[1], pos[2] + 0.06], frame_id=frame,
                marker_type=TEXT_VIEW_FACING, scale=[0.0, 0.0, 0.035],
                color=[1.0, 1.0, 1.0, 1.0], marker_id=base + 2,
                ns=f"{_VIZ_NS}/ee_target", text=f"{n}:{name}",
            )
            print(f"[arm:{label}] viz: EE target #{n} ({name}) at "
                  f"{frame}({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
        except Exception as exc:  # noqa: BLE001
            print(f"[arm:{label}] viz: EE target draw failed ({exc})")

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_inner"), name)
        if not callable(attr):
            return attr
        label = object.__getattribute__(self, "_label")
        draw_target = object.__getattribute__(self, "_draw_target")

        def _logged(*args, **kwargs):
            argstr = ", ".join(
                [_fmt(a) for a in args] + [f"{k}={_fmt(v)}" for k, v in kwargs.items()]
            )
            print(f"[arm:{label}] -> {name}({argstr})")
            target = _ee_target(name, args, kwargs)
            if target is not None:
                draw_target(name, *target)
            try:
                result = attr(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"[arm:{label}] <- {name} raised {exc!r}")
                raise
            print(f"[arm:{label}] <- {name} = {_fmt(result)}")
            return result

        return _logged


def install_arm_logging() -> None:
    """Wrap the executor's arm group so every actuator call is printed + drawn.

    Monkeypatches ``tasks.manipulation.execute._arm_group`` (robustly — works
    whether ``arm.left``/``arm.right`` are plain attributes or read-only
    properties) so all of pick_object's arm/gripper commands go through
    :class:`_LoggingArm`, which also draws each EE pose target in RViz (it pulls
    ``viz`` from the ctx passed to ``_arm_group``). Idempotent.
    """
    from tasks.manipulation import execute as _execute

    if getattr(_execute._arm_group, "_arm_logging", False):
        return
    orig = _execute._arm_group

    def _logging_arm_group(ctx):
        side = os.getenv("WALKIE_ARM", "left").strip().lower()
        viz = None
        try:
            viz = ctx.walkie.robot.viz
        except Exception:  # noqa: BLE001
            viz = None
        return _LoggingArm(orig(ctx), label=side, viz=viz)

    _logging_arm_group._arm_logging = True
    _execute._arm_group = _logging_arm_group
    print("[arm] actuator call logging enabled")


def _yaw_to_quat(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _rank_color(i: int, n: int) -> list[float]:
    """Best grasp (i=0) green, fading to red down the ranking. RGBA."""
    t = 0.0 if n <= 1 else i / (n - 1)
    return [t, 1.0 - t, 0.0, 1.0]


def draw_grasps(walkie, result: dict, *, max_show: int = 5) -> None:
    """Draw GraspNet candidates as arrows (ranked color) + the best approach pose.

    Grasps come back in ``planning_frame`` (map). The best grasp is
    drawn green with a SPHERE at its approach waypoint and a score label; lower
    -ranked grasps fade to red. Best-effort — viz failures never abort the test.
    """
    viz = walkie.robot.viz
    frame = result.get("planning_frame") or "map"
    grasps = result.get("grasps", [])[:max_show]
    try:
        viz.clear_markers()
    except Exception:  # noqa: BLE001
        pass
    for i, g in enumerate(grasps):
        color = _rank_color(i, len(grasps))
        try:
            viz.draw_marker(
                position=[float(v) for v in g["position"]],
                quaternion=[float(v) for v in g["orientation"]],
                frame_id=frame, marker_type=ARROW,
                scale=[0.10, 0.02, 0.02], color=color,
                marker_id=100 + i, ns=f"{_VIZ_NS}/grasps",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] grasp arrow {i} failed ({exc})")
        ap = g.get("approach_position")
        if i == 0 and ap is not None:
            try:
                viz.draw_marker(
                    position=[float(v) for v in ap], frame_id=frame,
                    marker_type=SPHERE, scale=[0.04, 0.04, 0.04],
                    color=[0.1, 0.4, 1.0, 0.9], marker_id=200, ns=f"{_VIZ_NS}/approach",
                )
                px, py, pz = (float(v) for v in g["position"])
                viz.draw_marker(
                    position=[px, py, pz + 0.08], frame_id=frame,
                    marker_type=TEXT_VIEW_FACING, scale=[0.0, 0.0, 0.04],
                    color=[1.0, 1.0, 1.0, 1.0], marker_id=201, ns=f"{_VIZ_NS}/label",
                    text=f"score={g.get('score')}",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[viz] approach/label failed ({exc})")
    print(f"[viz] drew {len(grasps)} grasp(s) in '{frame}' on walkie/viz_markers")


def draw_table_box(walkie, node) -> None:
    """Draw the surface collision box (floor -> top_z) as a translucent CUBE in map."""
    if node is None:
        return
    box = db.node_table_box(node)
    if box is None:
        return
    (cx, cy, top_z, yaw), (dx, dy) = box
    try:
        walkie.robot.viz.draw_marker(
            position=[cx, cy, top_z / 2.0], quaternion=_yaw_to_quat(yaw),
            frame_id="map", marker_type=CUBE,
            scale=[max(dx, 0.01), max(dy, 0.01), max(top_z, 0.01)],
            color=[0.6, 0.6, 0.6, 0.3], marker_id=300, ns=f"{_VIZ_NS}/table",
        )
        print(f"[viz] drew table box center=({cx:.2f},{cy:.2f},{top_z/2:.2f}) "
              f"size=({dx:.2f},{dy:.2f},{top_z:.2f}) in 'map'")
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] table box failed ({exc})")


def _obj_from_node(node) -> DetectedObject:
    return DetectedObject(
        bbox_xyxy=(0, 0, 0, 0),
        class_name=node.class_name,
        confidence=1.0,
        world_xy=(node.centroid[0], node.centroid[1]),
        world_xyz=tuple(node.centroid),
        node_id=node.id,
    )


def _select_index(dets, obj: DetectedObject) -> int:
    """Index of the detection the grasp pipeline would pick for *obj*.

    Mirrors ``grasp._live_detection_for``: the detection whose bbox center is
    nearest *obj*'s, or the most confident when *obj* has no usable bbox
    (cloud/pos resolve from the DB and carry a zero bbox).
    """
    ox1, oy1, ox2, oy2 = obj.bbox_xyxy
    if (ox1, oy1, ox2, oy2) == (0, 0, 0, 0):
        return max(range(len(dets)), key=lambda i: dets[i].confidence or 0.0)
    ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0

    def _key(i: int):
        x1, y1, x2, y2 = dets[i].bbox
        dist = ((x1 + x2) / 2.0 - ocx) ** 2 + ((y1 + y2) / 2.0 - ocy) ** 2
        return (dist, -(dets[i].confidence or 0.0))

    return min(range(len(dets)), key=_key)


def save_mask_overlay(ctx, obj: DetectedObject, *, path: str, source: str) -> None:
    """Render the live camera view with the SELECTED object's mask highlighted.

    Re-detects ``obj.class_name`` on a fresh snapshot and writes *path*: the
    chosen detection's mask in green (the one GraspNet is fed for SOURCE=mask, or
    stands in for it on cloud/pos), other same-class detections dimmed red, and
    bboxes + labels on top. The selected mask/box is drawn last so it stays on
    top where detections overlap. Best-effort — any failure logs and returns
    without aborting the test.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except Exception as exc:  # noqa: BLE001
        print(f"[mask-viz] numpy/PIL unavailable ({exc}); skipping")
        return
    snap = ctx.snapshot()
    if snap is None or getattr(snap, "img", None) is None:
        print("[mask-viz] no snapshot/img; skipping")
        return
    try:
        dets = ctx.walkieAI.image.detect(snap.img, prompts=[obj.class_name], return_mask=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[mask-viz] detect({obj.class_name!r}) failed ({exc})")
        return
    if not dets:
        print(f"[mask-viz] no live detections for {obj.class_name!r}; nothing to draw")
        return

    sel = _select_index(dets, obj)
    order = [i for i in range(len(dets)) if i != sel] + [sel]  # selected drawn last
    arr = np.array(snap.img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    sel_rgb = np.array([0, 255, 0], dtype=np.float32)
    oth_rgb = np.array([255, 80, 80], dtype=np.float32)
    alpha = 0.45
    for i in order:
        mask = getattr(dets[i], "mask", None)
        if mask is None:
            continue
        if mask.shape[:2] != (h, w):  # resize without cv2: PIL NEAREST
            mask = np.array(
                Image.fromarray((mask.astype(np.uint8) * 255)).resize((w, h), Image.NEAREST)
            ) > 0
        m = mask.astype(bool)
        color = sel_rgb if i == sel else oth_rgb
        arr[m] = (arr[m] * (1.0 - alpha) + color * alpha).astype(np.uint8)

    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    for i in order:
        det = dets[i]
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        is_sel = i == sel
        rgb = (0, 255, 0) if is_sel else (255, 80, 80)
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3 if is_sel else 1)
        conf = det.confidence if det.confidence is not None else 0.0
        label = f"{det.class_name or 'object'} {conf:.2f}" + (" <-PICK" if is_sel else "")
        draw.text((x1 + 2, max(y1 - 12, 2)), label, fill=rgb)

    try:
        out.save(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[mask-viz] save to {path!r} failed ({exc})")
        return
    sel_det = dets[sel]
    has_mask = getattr(sel_det, "mask", None) is not None
    print(f"[mask-viz] {source!r}: {len(dets)} detection(s) of {obj.class_name!r}; "
          f"selected #{sel} conf={sel_det.confidence} "
          f"bbox={tuple(int(v) for v in sel_det.bbox)} mask={'yes' if has_mask else 'NO'} "
          f"-> wrote {path}")


def _raw_result(ctx, graphs, node, obj, source):
    """Call the source-specific GraspNet entry directly, for rich printing."""
    if source == "mask":
        return grasp._grasp_from_mask(ctx, obj)
    if source == "pos":
        return grasp._grasp_from_pos(ctx, graphs, node)
    return grasp._grasp_from_cloud(ctx, graphs, node)


def main() -> None:
    load_dotenv()
    load_task_config(os.path.dirname(__file__))

    query = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    source = os.getenv("SOURCE", os.getenv("WALKIE_GRASP_SOURCE", "cloud")).strip().lower()
    os.environ["WALKIE_GRASP_SOURCE"] = source  # so plan_grasp/pick_object agree
    execute = os.getenv("EXECUTE", "0").lower() in ("1", "true", "yes")
    viz = os.getenv("VIZ", "1").lower() in ("1", "true", "yes")
    mask_viz = os.getenv("MASK_VIZ", "1").lower() in ("1", "true", "yes")
    mask_viz_path = os.getenv("MASK_VIZ_PATH", "pick_mask_viz.jpg")
    # CONFIRM=1 -> step through every arm/base motion (Enter=do, s=skip, q=abort).
    # Default on when EXECUTE so a tester gates each real move; CONFIRM=0 disables.
    if os.getenv("CONFIRM", "1" if execute else "0").lower() in ("1", "true", "yes"):
        os.environ["WALKIE_MANIP_CONFIRM"] = "1"

    walkie = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()
    graphs = initialize_graphs(model, walkie_ai, walkie)
    ctx = TaskContext(walkie=walkie, walkieAI=walkie_ai, model=model, graphs=graphs)

    # Log every arm/gripper command the executor issues (no-op on dry runs,
    # which return before any arm motion).
    install_arm_logging()

    try:
        node = db.resolve_object_node(graphs, query)
        if node is None and source in ("cloud", "pos"):
            print(f"[test] no DB node for {query!r}; scan it first (or use SOURCE=mask).")
            return
        if node is not None:
            print(f"[test] node {node.id} class={node.class_name} centroid={node.centroid} "
                  f"aabb={node.aabb_min}..{node.aabb_max}")
        obj = _obj_from_node(node) if node is not None else DetectedObject(
            bbox_xyxy=(0, 0, 0, 0), class_name=query, confidence=1.0,
        )

        # Live mask overlay of the selected object — written even if GraspNet
        # later returns no grasps, so the tester can see what was perceived.
        if mask_viz:
            save_mask_overlay(ctx, obj, path=mask_viz_path, source=source)

        print(f"[test] GraspNet source = {source!r}")
        result = _raw_result(ctx, graphs, node, obj, source)
        if not result or not result.get("grasps"):
            print(f"[test] GraspNet returned no grasps: {result}")
            return
        print(f"[test] planning_frame={result.get('planning_frame')} "
              f"n_grasps={len(result['grasps'])} object_size={result.get('object_size')}")
        for i, g in enumerate(result["grasps"][:5]):
            print(f"  grasp[{i}] score={g.get('score')} width={g.get('width')} "
                  f"antipodal={g.get('antipodal_score')} pos={g['position']} "
                  f"quat={g['orientation']} approach={g.get('approach_position')}")

        if viz:
            # Resolve the surface node the executor would use, so the drawn table
            # box matches what gets added to the MoveIt planning scene.
            surface = db.resolve_surface_node(
                graphs, os.getenv("WALKIE_SURFACE_CLASS", "table"),
                near=obj.world_xy,
            ) if graphs is not None else None
            draw_grasps(walkie, result)
            draw_table_box(walkie, surface)
            print("[test] markers published — open RViz, add MarkerArray on "
                  "'walkie/viz_markers' (grasps in base_footprint, table in map).")

        if not execute:
            print("[test] dry run (set EXECUTE=1 to perform the pick).")
            return

        grasped = pick_object(ctx, obj)
        print(f"[test] pick_object -> grasped={grasped}")
    finally:
        if graphs is not None:
            graphs.stop()
        walkie.close()


if __name__ == "__main__":
    main()
