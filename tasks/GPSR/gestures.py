"""Gesture / pose recognition from COCO keypoints — pure and offline-testable.

GPSR commands reference people by gesture/pose ("the waving person", "the person
raising their left arm", "how many people are sitting"). The detector gives 17
named COCO keypoints per person (client.image.PersonPose); these
heuristics turn keypoints into the canonical gesture ids the world model uses
(world.toml [gestures]). No robot, no network — `PersonPose` is a plain
dataclass — so the rules are unit-tested directly.

Conventions: image coordinates, y grows DOWNWARD (a raised hand has the SMALLER
y). "left/right arm" is the person's OWN side (anatomical, matching COCO's
left_/right_ naming and the generator's "their left arm"). "pointing left/right"
is in the ROBOT's view (which way the arm extends across the image), since that
is what the robot can act on. Thresholds are scale-invariant (fractions of torso
height) so they hold near and far.
"""

from __future__ import annotations

import os

from client.image import PersonPose, PoseKeypoint


def _conf() -> float:
    return float(os.getenv("GPSR_KP_MIN_CONF", "0.3"))


def _kp(person: PersonPose, name: str) -> PoseKeypoint | None:
    """The named keypoint if present and confident enough, else None."""
    thresh = _conf()
    for k in person.keypoints:
        if k.name == name and k.confidence >= thresh:
            return k
    return None


def _torso_h(person: PersonPose) -> float | None:
    """Vertical shoulder→hip distance (scale reference); None if unavailable.

    Uses whichever shoulder/hip pair is present; falls back to |y| span so a
    person lying down (small vertical torso) still yields a usable, if small,
    reference. Returns None only when no shoulder+hip pair exists.
    """
    for s_name, h_name in (("left_shoulder", "left_hip"), ("right_shoulder", "right_hip")):
        s, h = _kp(person, s_name), _kp(person, h_name)
        if s and h:
            return abs(h.y - s.y) or 1.0
    return None


def _arm_raised(person: PersonPose, side: str) -> bool:
    """The side's wrist is clearly above (smaller y than) its shoulder."""
    wrist, shoulder = _kp(person, f"{side}_wrist"), _kp(person, f"{side}_shoulder")
    torso = _torso_h(person)
    if not (wrist and shoulder and torso):
        return False
    margin = float(os.getenv("GPSR_ARM_RAISE_MARGIN", "0.15")) * torso
    return wrist.y < shoulder.y - margin


def _arm_pointing(person: PersonPose, direction: str) -> bool:
    """An arm extended roughly horizontally toward image-left/right (robot view).

    Either arm counts; the wrist must be far from its shoulder horizontally and
    near shoulder height (not raised, not hanging). `direction` is "left" (wrist
    to smaller x) or "right" (larger x) in the image.
    """
    torso = _torso_h(person)
    if not torso:
        return False
    h_margin = float(os.getenv("GPSR_POINT_X_MARGIN", "0.6")) * torso
    v_tol = float(os.getenv("GPSR_POINT_Y_TOL", "0.5")) * torso
    for side in ("left", "right"):
        wrist, shoulder = _kp(person, f"{side}_wrist"), _kp(person, f"{side}_shoulder")
        if not (wrist and shoulder):
            continue
        if abs(wrist.y - shoulder.y) > v_tol:
            continue  # raised or hanging, not a horizontal point
        dx = wrist.x - shoulder.x
        if direction == "right" and dx > h_margin:
            return True
        if direction == "left" and dx < -h_margin:
            return True
    return False


def _vspan(person: PersonPose, names: list[str]) -> tuple[float, float] | None:
    ys = [k.y for n in names if (k := _kp(person, n))]
    return (min(ys), max(ys)) if ys else None


def is_waving(person: PersonPose) -> bool:
    """A raised hand on either side (single frame can't see the motion)."""
    return _arm_raised(person, "left") or _arm_raised(person, "right")


def is_sitting(person: PersonPose) -> bool:
    """Thighs roughly horizontal: small hip→knee vertical span vs. the torso."""
    torso = _torso_h(person)
    if not torso:
        return False
    ratio = float(os.getenv("GPSR_SIT_RATIO", "0.6"))
    for side in ("left", "right"):
        hip, knee = _kp(person, f"{side}_hip"), _kp(person, f"{side}_knee")
        if hip and knee and (knee.y - hip.y) < ratio * torso:
            return True
    return False


def is_standing(person: PersonPose) -> bool:
    """Hips→ankles extended vertically and the body upright (taller than wide)."""
    torso = _torso_h(person)
    if not torso:
        return False
    ratio = float(os.getenv("GPSR_STAND_RATIO", "1.2"))
    legs_ok = False
    for side in ("left", "right"):
        hip, ankle = _kp(person, f"{side}_hip"), _kp(person, f"{side}_ankle")
        if hip and ankle and (ankle.y - hip.y) > ratio * torso:
            legs_ok = True
    if not legs_ok:
        return False
    return not is_lying_down(person)


def is_lying_down(person: PersonPose) -> bool:
    """Body horizontal: overall x-span exceeds y-span (wider than tall)."""
    pts = [k for k in person.keypoints if k.confidence >= _conf()]
    if len(pts) < 4:
        return False
    xs, ys = [k.x for k in pts], [k.y for k in pts]
    x_span, y_span = max(xs) - min(xs), max(ys) - min(ys)
    ratio = float(os.getenv("GPSR_LIE_RATIO", "1.3"))
    return x_span > ratio * max(y_span, 1.0)


# gesture id (world.toml) -> predicate
_PREDICATES = {
    "waving": is_waving,
    "raising_left_arm": lambda p: _arm_raised(p, "left"),
    "raising_right_arm": lambda p: _arm_raised(p, "right"),
    "pointing_left": lambda p: _arm_pointing(p, "left"),
    "pointing_right": lambda p: _arm_pointing(p, "right"),
    "sitting": is_sitting,
    "standing": is_standing,
    "lying_down": is_lying_down,
}


def matches_gesture(person: PersonPose, gesture: str) -> bool:
    """True if *person*'s keypoints satisfy the canonical *gesture* id."""
    pred = _PREDICATES.get(gesture)
    return bool(pred and pred(person))


def classify_gestures(person: PersonPose) -> set[str]:
    """All canonical gestures this person currently matches (may overlap)."""
    return {g for g, pred in _PREDICATES.items() if pred(person)}
