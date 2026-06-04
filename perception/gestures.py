"""Single-frame gesture / posture reasoning from pose keypoints (HRI C7).

Pure functions over a list of COCO ``PoseKeypoint``-shaped objects (anything
with ``.index``, ``.x``, ``.y``, ``.confidence``). No camera, no server, no
state — so the whole module is offline-testable with synthetic keypoints, the
same way the human-agent pose heuristics are.

This lifts the ``arm_raised`` / ``posture`` heuristics that used to live inline
in ``agents/human_agent/tools.py`` (they are re-exported there for backward
compatibility) and adds the gestures the GPSR / Restaurant tasks ask about:

- **waving / hand raised** — a wrist above its shoulder (a wave or hand-raise).
- **pointing left / right** — an arm extended roughly horizontally to one side.
- **posture** — sitting / standing / **lying** / unknown.

Everything is a *single-frame* heuristic over noisy keypoints — treat the output
as a hint, not ground truth. A real wave (vs. a static raised hand) needs
temporal information we don't have here; "waving" therefore means "hand up".
"""

from __future__ import annotations

from typing import Optional

# COCO keypoint indices (pose_estimation returns 17 of these per person).
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

KP_CONF = 0.3  # minimum keypoint confidence to trust a coordinate

# Pointing: how far past the shoulders (in shoulder-widths) a wrist must reach
# sideways before we call it a point, and how horizontal the arm must be.
_POINT_EXTEND = 1.2


def kp_map(pose) -> dict:
    """``index -> keypoint`` for the keypoints this pose actually carries."""
    return {kp.index: kp for kp in pose.keypoints}


def _get(kpts, index):
    """The keypoint at *index* if present and confident, else ``None``."""
    kp = kpts.get(index)
    return kp if kp and kp.confidence > KP_CONF else None


def arm_raised(kpts) -> bool:
    """True if either wrist is above (smaller image-y than) its shoulder."""
    for shoulder_i, wrist_i in ((LEFT_SHOULDER, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_WRIST)):
        s, w = _get(kpts, shoulder_i), _get(kpts, wrist_i)
        if s and w and w.y < s.y:
            return True
    return False


def _shoulder_width(kpts) -> Optional[float]:
    ls, rs = _get(kpts, LEFT_SHOULDER), _get(kpts, RIGHT_SHOULDER)
    if not ls or not rs:
        return None
    return abs(ls.x - rs.x)


def pointing(kpts) -> Optional[str]:
    """Image-side a person is pointing to: ``'left'`` / ``'right'`` / ``None``.

    An arm counts as pointing when its wrist reaches more than ``_POINT_EXTEND``
    shoulder-widths sideways of its shoulder and the arm is more horizontal than
    vertical (so a raised hand isn't mistaken for a point). Directions are in the
    **camera image frame** — with the robot facing forward, image-left is the
    robot's left. If both arms qualify, the more extended one wins.
    """
    width = _shoulder_width(kpts)
    if not width or width <= 0:
        return None
    best_dir, best_dx = None, _POINT_EXTEND * width
    for shoulder_i, wrist_i in ((LEFT_SHOULDER, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_WRIST)):
        s, w = _get(kpts, shoulder_i), _get(kpts, wrist_i)
        if not s or not w:
            continue
        dx, dy = w.x - s.x, abs(w.y - s.y)
        if abs(dx) > best_dx and abs(dx) > dy:  # far enough sideways and horizontal
            best_dir, best_dx = ("left" if dx < 0 else "right"), abs(dx)
    return best_dir


def posture(kpts) -> str:
    """'sitting' / 'standing' / 'lying' / 'unknown' from torso & leg geometry.

    Heuristic only — pose keypoints are noisy and the lower body is often
    occluded. Order matters: a horizontal torso (shoulders and hips side-by-side
    rather than stacked) reads as **lying** before the sit/stand test runs.
    Otherwise we compare the hip->knee vertical drop to the shoulder->hip drop:
    when the legs are folded (sitting) the knees sit near hip height, so the
    ratio collapses. Returns 'unknown' whenever the needed keypoints are missing.
    """
    def avg(index_a, index_b, axis):
        pts = [getattr(p, axis) for p in (_get(kpts, index_a), _get(kpts, index_b)) if p]
        return sum(pts) / len(pts) if pts else None

    shoulder_y = avg(LEFT_SHOULDER, RIGHT_SHOULDER, "y")
    hip_y = avg(LEFT_HIP, RIGHT_HIP, "y")
    if shoulder_y is None or hip_y is None:
        return "unknown"

    shoulder_x = avg(LEFT_SHOULDER, RIGHT_SHOULDER, "x")
    hip_x = avg(LEFT_HIP, RIGHT_HIP, "x")
    if shoulder_x is not None and hip_x is not None:
        if abs(shoulder_x - hip_x) > abs(hip_y - shoulder_y):
            return "lying"

    knee_y = avg(LEFT_KNEE, RIGHT_KNEE, "y")
    if knee_y is None:
        return "unknown"
    torso = hip_y - shoulder_y
    if torso <= 0:
        return "unknown"
    leg_drop = (knee_y - hip_y) / torso
    return "sitting" if leg_drop < 0.5 else "standing"


def summarize_person(pose) -> dict:
    """The minimal {arm_raised, posture} summary used by ``count_people``.

    Kept narrow on purpose so callers that compare it by equality stay stable;
    use :func:`describe_gestures` for the richer breakdown.
    """
    kpts = kp_map(pose)
    return {"arm_raised": arm_raised(kpts), "posture": posture(kpts)}


def describe_gestures(pose) -> dict:
    """Full single-frame gesture breakdown for one person.

    Returns ``{"waving": bool, "pointing": 'left'|'right'|None,
    "posture": 'sitting'|'standing'|'lying'|'unknown'}``. ``waving`` is a raised
    hand (no temporal motion available from one frame).
    """
    kpts = kp_map(pose)
    return {
        "waving": arm_raised(kpts),
        "pointing": pointing(kpts),
        "posture": posture(kpts),
    }


def gesture_phrase(g: dict) -> str:
    """A short human phrase for one person's :func:`describe_gestures` dict."""
    bits = []
    if g["waving"]:
        bits.append("waving / hand raised")
    if g["pointing"]:
        bits.append(f"pointing to your {g['pointing']}")
    if g["posture"] != "unknown":
        bits.append(g["posture"])
    return ", ".join(bits) if bits else "no clear gesture"
