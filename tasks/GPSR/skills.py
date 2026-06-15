"""GPSR Tier-1 skills: one deterministic function per primitive (the §7 table).

Each skill has the signature ``skill(ctx, step, world, state) -> bool`` and
returns True when it executed without a hard error (it speaks honest results,
including negative ones like "I couldn't find it" or a count of zero — those
still *ran*, so they don't trigger the Tier-2 agent fallback). It returns False
only on an execution error (no camera frame, exception), which lets dispatch.py
hand the clause to the agent stack.

Phase 1 covers the no-arm primitives (navigate, find_object, find_person, count,
say, greet, get_person_info, get_object_property). Manipulation (pick/place/
deliver) is gated off and falls through to Tier-2 until Phase 2.

`state` is the per-command scratch dict — find_object/find_person stash the
located target's map point so a following greet/say can face it.

Reuses HRI's geometry helpers (lift, face, heading) and the pure gesture
heuristics (gestures.py). Imports tasks.base, so this module is robot-side and
not offline-importable (the dev box has no CUDA) — the testable logic lives in
gestures.py / plan.py.
"""

from __future__ import annotations

import os

from langchain.messages import HumanMessage

from tasks.base import TaskContext
from tasks.HRI.skills import cxcywh_to_xyxy, face_point, lift_bbox_world_xy

from . import gestures
from .plan import PlanStep
from .world import WorldModel


# --- low-level helpers ------------------------------------------------------

def _detect(ctx: TaskContext, img, classes: list[str], *, return_mask: bool = False):
    try:
        return ctx.walkieAI.object_detection.detect(img, prompts=classes, return_mask=return_mask)
    except Exception as exc:
        print(f"[gpsr.skill] object detection failed ({exc})")
        return []


def _people(ctx: TaskContext, img):
    try:
        return ctx.walkieAI.pose_estimation.estimate(img)
    except Exception as exc:
        print(f"[gpsr.skill] pose estimation failed ({exc})")
        return []


def _caption(ctx: TaskContext, crop, prompt: str) -> str | None:
    try:
        return ctx.walkieAI.image_caption.caption(crop, prompt=prompt)
    except Exception as exc:
        print(f"[gpsr.skill] caption failed ({exc})")
        return None


def _llm_line(ctx: TaskContext, prompt: str) -> str:
    """One short spoken line from the LLM (for answers/greetings). '' on failure."""
    try:
        reply = ctx.model.invoke([HumanMessage(content=prompt)])
        return (str(reply.content) or "").strip()
    except Exception as exc:
        print(f"[gpsr.skill] llm line failed ({exc})")
        return ""


def go_to_named(ctx: TaskContext, name: str | None, world: WorldModel) -> bool:
    """Navigate to a canonical room/location by its world-model pose."""
    if not name:
        return False
    pose = world.location_pose(name)
    if pose is None:
        print(f"[gpsr.skill] no pose for {name!r}")
        return False
    return ctx.goto(*pose)


def _object_classes(obj: str | None, category: str | None, world: WorldModel) -> list[str]:
    """Detector prompt classes for an object/category (underscores -> spaces)."""
    if obj:
        return [obj.replace("_", " ")]
    if category and world.categories.get(category):
        return [o.replace("_", " ") for o in world.categories[category]]
    return []


def _crop(img, bbox_xyxy):
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    return img.crop((max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)))


# --- primitives -------------------------------------------------------------

def navigate(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    return go_to_named(ctx, step.args.get("target"), world)


def find_object(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    obj = step.args.get("object")
    where = step.args.get("location") or step.args.get("room")
    if where:
        go_to_named(ctx, where, world)
    snap = ctx.snapshot()
    if snap is None:
        return False
    classes = _object_classes(obj, step.args.get("category"), world) or ["object"]
    dets = _detect(ctx, snap.img, classes)
    label = (obj or "object").replace("_", " ")
    if not dets:
        ctx.say(f"I could not find the {label}.")
        return True  # the search ran; nothing to find is an honest result
    best = max(dets, key=lambda d: d.confidence or 0.0)
    xy = lift_bbox_world_xy(ctx, snap, best.bbox)
    if xy:
        state["target_xy"] = xy
        state["found_object"] = obj
    ctx.say(f"I found the {label}.")
    return True


def _find_person_match(ctx: TaskContext, step: PlanStep):
    """Return (snap, person_bbox_xyxy) for the descriptor, or (snap, None)."""
    snap = ctx.snapshot()
    if snap is None:
        return None, None
    people = _people(ctx, snap.img)
    if not people:
        return snap, None
    kind = step.args.get("kind")
    descriptor = step.args.get("descriptor")
    if kind in ("gesture", "pose") and descriptor:
        matches = [p for p in people if gestures.matches_gesture(p, descriptor)]
        people = matches or []
    # name / clothing / no-match: best-effort nearest (largest bbox = closest).
    if not people:
        return snap, None
    target = max(people, key=lambda p: p.bbox[2] * p.bbox[3])
    return snap, cxcywh_to_xyxy(target.bbox)


def find_person(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    room = step.args.get("room")
    if room:
        go_to_named(ctx, room, world)
    snap, bbox = _find_person_match(ctx, step)
    if snap is None:
        return False
    who = step.args.get("descriptor") or "person"
    if bbox is None:
        ctx.say(f"I could not find {who}.")
        return True
    xy = lift_bbox_world_xy(ctx, snap, bbox)
    if xy:
        state["target_xy"] = xy
        face_point(ctx, *xy)
    ctx.say("I found you.")
    return True


def count(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    where = step.args.get("location") or step.args.get("room")
    if where:
        go_to_named(ctx, where, world)
    snap = ctx.snapshot()
    if snap is None:
        return False
    if step.args.get("what") == "persons":
        people = _people(ctx, snap.img)
        descriptor = step.args.get("descriptor")
        if step.args.get("kind") in ("gesture", "pose") and descriptor:
            people = [p for p in people if gestures.matches_gesture(p, descriptor)]
            ctx.say(f"I count {len(people)} {descriptor.replace('_', ' ')} people.")
        else:
            ctx.say(f"I count {len(people)} people.")
        return True
    classes = _object_classes(step.args.get("object"), step.args.get("category"), world) or ["object"]
    dets = _detect(ctx, snap.img, classes)
    noun = (step.args.get("object") or step.args.get("category") or "objects").replace("_", " ")
    ctx.say(f"I count {len(dets)} {noun}.")
    return True


def say(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    info = (step.args.get("info") or "").strip()
    if not info:
        return False
    # Resolve dynamic asks (the time, a joke, an answer) via the LLM; speak
    # literal statements as-is.
    dynamic = info.endswith("?") or any(
        w in info.lower() for w in ("time", "date", "day", "joke", "name of", "what", "who", "how")
    )
    text = _llm_line(ctx, f"In one short spoken sentence, respond to this request: {info}") if dynamic else info
    ctx.say(text or info)
    return True


def greet(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    room = step.args.get("room")
    if room:
        go_to_named(ctx, room, world)
    snap, bbox = _find_person_match(ctx, step)
    if snap is not None and bbox is not None:
        xy = lift_bbox_world_xy(ctx, snap, bbox)
        if xy:
            face_point(ctx, *xy)
    name = step.args.get("descriptor") if step.args.get("kind") == "name" else None
    ctx.say(f"Hello {name}, nice to meet you!" if name else "Hello, nice to meet you!")
    return True


def get_person_info(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    which = step.args.get("which")
    snap = ctx.snapshot()
    if snap is None:
        return False
    people = _people(ctx, snap.img)
    if not people:
        ctx.say("I do not see anyone to describe.")
        return True
    target = max(people, key=lambda p: p.bbox[2] * p.bbox[3])
    if which in ("pose", "gesture"):
        found = gestures.classify_gestures(target)
        ctx.say(f"The person seems to be {', '.join(sorted(found)).replace('_', ' ')}." if found else "I cannot tell the person's pose.")
    elif which == "clothing":
        cap = _caption(ctx, _crop(snap.img, cxcywh_to_xyxy(target.bbox)),
                       "Describe this person's clothing in one short sentence.")
        ctx.say(cap or "I cannot make out the person's clothing.")
    else:  # name — needs to ask (no prior enrollment in GPSR)
        answer = ctx.ask("Hello, what is your name?")
        ctx.say(f"Your name is {answer}." if answer else "I did not catch the name.")
    return True


def get_object_property(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    which = step.args.get("which")
    obj = step.args.get("object")
    # Category is known from the world model — no perception needed.
    if which == "category" and obj:
        cat = world.objects.get(obj)
        ctx.say(f"The {obj.replace('_', ' ')} is a {cat.replace('_', ' ')}." if cat else
                f"I am not sure what category the {obj.replace('_', ' ')} is in.")
        return True
    snap = ctx.snapshot()
    if snap is None:
        return False
    classes = _object_classes(obj, None, world) or ["object"]
    dets = _detect(ctx, snap.img, classes)
    if not dets:
        ctx.say(f"I could not find the {(obj or 'object').replace('_', ' ')}.")
        return True
    best = max(dets, key=lambda d: d.confidence or 0.0)
    cap = _caption(ctx, _crop(snap.img, best.bbox),
                   f"In one short phrase, describe the {which or 'appearance'} of the main object in this image.")
    ctx.say(cap or f"I cannot tell the {which or 'property'} of that object.")
    return True


# primitive value -> skill. Manipulation primitives are intentionally absent:
# they fall through to the Tier-2 agent fallback until Phase 2.
SKILLS = {
    "navigate": navigate,
    "find_object": find_object,
    "find_person": find_person,
    "count": count,
    "say": say,
    "greet": greet,
    "get_person_info": get_person_info,
    "get_object_property": get_object_property,
}
