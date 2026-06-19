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

`state` is the **per-command** scratch dict. `state["at"]` is the nav-dedup
(canonical name of where the robot is; in interleaved runs only this key crosses
commands). find_object/find_person also stash the located target's map point
(`target_xy`/`found_object`) for a *later step of the same command* to use (e.g. a
future place/deliver) — these are currently written but not yet read, so they must
stay per-command (dispatch.execute_interleaved keeps them isolated).

Reuses HRI's geometry helpers (lift, face, heading) and the pure gesture
heuristics (gestures.py). Imports tasks.base, so this module is robot-side and
not offline-importable (the dev box has no CUDA) — the testable logic lives in
gestures.py / plan.py.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from langchain.messages import HumanMessage

from tasks.base import TaskContext
from tasks.skills import (
    cxcywh_to_xyxy,
    face_point,
    follow_person,
    lift_bbox_world_xy,
    select_largest_person,
)

from . import gestures
from .plan import PlanStep, _person_phrase
from .tracking import ArrivalStopper
from .world import WorldModel


# --- low-level helpers ------------------------------------------------------

def _detect(ctx: TaskContext, img, classes: list[str], *, return_mask: bool = False):
    try:
        return ctx.walkieAI.image.detect(img, prompts=classes, return_mask=return_mask)
    except Exception as exc:
        print(f"[gpsr.skill] object detection failed ({exc})")
        return []


def _people(ctx: TaskContext, img):
    try:
        return ctx.walkieAI.image.estimate_poses(img)
    except Exception as exc:
        print(f"[gpsr.skill] pose estimation failed ({exc})")
        return []


def _caption(ctx: TaskContext, crop, prompt: str) -> str | None:
    try:
        return ctx.walkieAI.image.caption(crop, prompt=prompt)
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


def _facts_context() -> str:
    """Known facts the robot can be asked to 'tell' (the §11 say/tell source).

    RoboCup 'tell' commands ask for things general knowledge can't give reliably
    — the day/date/time, the team name, who the robot is. We ground the LLM with
    these instead of letting it hallucinate or refuse. Identity is config-driven
    (GPSR_ROBOT_NAME/GPSR_TEAM_NAME/GPSR_TEAM_AFFILIATION); the clock is read live.
    """
    now = datetime.now()
    robot = os.getenv("GPSR_ROBOT_NAME", "Walkie")
    team = os.getenv("GPSR_TEAM_NAME", "EIC")
    affiliation = os.getenv("GPSR_TEAM_AFFILIATION", "Chulalongkorn University")
    return (
        f"You are {robot}, a domestic service robot competing in RoboCup@Home for "
        f"team {team} from {affiliation}. "
        # `now.day` (not %-d) so the format is portable, not glibc-only.
        f"Today is {now:%A}, {now.day} {now:%B %Y}; the current time is {now:%H:%M}."
    )


def go_to_named(
    ctx: TaskContext, name: str | None, world: WorldModel, state: dict | None = None
) -> bool:
    """Navigate to a canonical room/location by its world-model pose.

    Idempotent within a command: if `state` already records the robot at `name`,
    the redundant drive is skipped. This dedups the parser's *explicit* navigate
    step against a following find/greet/count step that also names the same place
    (the parser is told to make navigation explicit, but still fills the find
    step's room/location) — otherwise the robot drives there twice.
    """
    if not name:
        return False
    if state is not None and state.get("at") == name:
        return True  # already here this command — don't drive again
    pose = world.location_pose(name)
    if pose is None:
        print(f"[gpsr.skill] no pose for {name!r}")
        return False
    ok = ctx.goto(*pose)
    if ok and state is not None:
        state["at"] = name  # remember so the next step doesn't re-navigate
    return ok


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
    return go_to_named(ctx, step.args.get("target"), world, state)


def find_object(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    obj = step.args.get("object")
    where = step.args.get("location") or step.args.get("room")
    if where:
        go_to_named(ctx, where, world, state)
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


def _pick_index(reply: str | None, n: int) -> int | None:
    """First integer in *reply*, if it is a valid 0..n-1 candidate index."""
    m = re.search(r"-?\d+", reply or "")
    if not m:
        return None
    i = int(m.group())
    return i if 0 <= i < n else None


def _match_attire(ctx: TaskContext, snap, people, descriptor: str):
    """Pick the person whose clothing best matches a free-text descriptor.

    Clothing is NOT a re-ID problem — GPSR enrolls no face/appearance gallery,
    so the HRI people store is empty and ``locate_people`` can't help. It's
    attribute matching: caption each candidate's crop, then let the LLM choose
    the index that matches '<descriptor>' (or none). Returns the chosen person,
    or None when nobody matches — so the caller can honestly report a miss
    instead of grabbing a bystander.
    """
    captions = [
        _caption(ctx, _crop(snap.img, cxcywh_to_xyxy(p.bbox)),
                 "Describe this person's clothing and its colors in one short phrase.") or ""
        for p in people
    ]
    listing = "\n".join(f"{i}: {c}" for i, c in enumerate(captions))
    reply = _llm_line(
        ctx,
        f'A person is described as: "{descriptor}".\n'
        f"People currently visible, by index and their clothing:\n{listing}\n"
        "Reply with ONLY the index of the best match, or -1 if none clearly matches.",
    )
    idx = _pick_index(reply, len(people))
    return people[idx] if idx is not None else None


def _find_person_match(ctx: TaskContext, step: PlanStep):
    """Return (snap, person_bbox_xyxy) for the descriptor, or (snap, None).

    Matching is per descriptor *kind* (parse._ground_person):
      gesture/pose -> COCO-keypoint heuristics (gestures.py);
      clothing     -> caption each candidate and LLM-pick the best attire match;
      name         -> GPSR enrolls no face gallery, so identity can't be
                      verified; fall back to the nearest person (best effort) and
                      let the caller address them by name;
      (none given) -> the nearest person.
    snap is None only on a camera failure; bbox is None when nobody matches.
    """
    snap = ctx.snapshot()
    if snap is None:
        return None, None
    people = _people(ctx, snap.img)
    if not people:
        return snap, None
    kind = step.args.get("kind")
    descriptor = step.args.get("descriptor")

    if kind in ("gesture", "pose") and descriptor:
        people = [p for p in people if gestures.matches_gesture(p, descriptor)]
        if not people:
            return snap, None  # nobody is doing that gesture/pose

    elif kind == "clothing" and descriptor:
        match = _match_attire(ctx, snap, people, descriptor)
        return snap, (cxcywh_to_xyxy(match.bbox) if match is not None else None)

    # name / no descriptor / gesture-matched: nearest person (largest = closest).
    target = max(people, key=lambda p: p.bbox[2] * p.bbox[3])
    return snap, cxcywh_to_xyxy(target.bbox)


def find_person(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    room = step.args.get("room")
    if room:
        go_to_named(ctx, room, world, state)
    snap, bbox = _find_person_match(ctx, step)
    if snap is None:
        return False  # no camera frame -> let Tier-2 try
    who = _person_phrase(step.args.get("descriptor"), step.args.get("kind"))
    if bbox is None:
        ctx.say(f"I could not find {who}.")
        return True
    xy = lift_bbox_world_xy(ctx, snap, bbox)
    if xy:
        state["target_xy"] = xy
        face_point(ctx, *xy)
    ctx.say(f"I found {who}.")
    return True


def count(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    where = step.args.get("location") or step.args.get("room")
    if where:
        go_to_named(ctx, where, world, state)
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
    # Route every 'tell' through the LLM with the known-facts context: it answers
    # information requests (the day, time, a joke, who/what the robot is) using
    # the facts, and otherwise just announces the phrase. This replaces a brittle
    # keyword heuristic and grounds the answer (§11 say/tell knowledge source).
    prompt = (
        f"{_facts_context()}\n\n"
        f'The operator asked you to convey: "{info}".\n'
        "Reply with ONE short, natural spoken sentence. If it requests information "
        "(the day, date, time, a joke, a fact about you or your team), answer it "
        "using the facts above. If it is simply a phrase to announce, say it as "
        "given. Output only the sentence to speak, no quotes."
    )
    text = _llm_line(ctx, prompt)
    ctx.say(text or info)  # fall back to the literal request if the LLM fails
    return True


def greet(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    room = step.args.get("room")
    if room:
        go_to_named(ctx, room, world, state)
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


def follow(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    """Follow a person, reusing HRI's ``follow_person`` tracking loop (off main).

    GPSR enrolls nobody, so we track whoever is in front via
    ``select_largest_person`` — i.e. the person who just issued the command. When
    the command names a destination (``follow me to X``), an
    :class:`tracking.ArrivalStopper` ends the loop the moment the robot reaches
    X's pose (``follow_person`` returns ``'stopped'``); otherwise it runs until
    the person is lost or ``HRI_FOLLOW_TIMEOUT_SEC``. Returns True unless
    follow_person raises.
    """
    to = step.args.get("to")
    target_xy = None
    if to:
        pose = world.location_pose(to)
        if pose is not None:
            target_xy = (pose[0], pose[1])
    stopper = ArrivalStopper(ctx, target_xy) if target_xy is not None else None
    reason = follow_person(
        ctx,
        select_largest_person,
        stopper=stopper,
        on_warmup=lambda: ctx.say("Okay, please lead the way slowly and I will follow you."),
    )
    print(f"[gpsr.skill] follow exit reason: {reason}")
    if reason == "stopped" and to:  # the arrival stopper fired -> we reached `to`
        ctx.say(f"We have arrived at {to.replace('_', ' ')}.")
    elif reason == "lost":
        ctx.say("I lost track of you.")
    else:  # timed out, or stopped with no named destination
        ctx.say("I have stopped following.")
    return True


def guide(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    """Guide (lead) a person to a destination — nav + best-effort person confirm.

    Optionally goes to ``from`` first (where the person is), confirms/faces them,
    then leads to ``to`` and announces arrival. We lead with our back to the person,
    so confirming they arrived with us needs **mid-route re-acquire** (looking back
    at the trailing follower), the open follow-back TODO — a forward-facing arrival
    frame can't see them. Returns False (→ Tier-2) only when the destination is
    missing or unreachable — matching `navigate`.
    """
    to = step.args.get("to")
    frm = step.args.get("from")
    descriptor = step.args.get("descriptor")
    kind = step.args.get("kind")
    # Meet the person at the start beacon, if one was named.
    if frm:
        go_to_named(ctx, frm, world, state)
    # Confirm + face the person (best-effort; GPSR enrolls nobody).
    snap, bbox = _find_person_match(ctx, step)
    if snap is not None and bbox is not None:
        xy = lift_bbox_world_xy(ctx, snap, bbox)
        if xy:
            face_point(ctx, *xy)
    dest = (to or "").replace("_", " ")
    addr = f"Hello {descriptor}, " if kind == "name" and descriptor else ""
    ctx.say(f"{addr}please follow me, and I will guide you to {dest}.")
    if not to or not go_to_named(ctx, to, world, state):
        ctx.say("I am sorry, I could not find the way there.")
        return False
    # We led with our back to the person (they trail behind), so the forward camera
    # frame here CANNOT confirm they arrived — checking it would false-negative on a
    # compliant follower. Confirming the companion needs looking back / mid-route
    # re-acquire (tracking.companion_present; the open follow-back TODO). Announce.
    ctx.say(f"We have arrived at {dest}.")
    return True


# primitive value -> skill. The manipulation primitives (pick/place/deliver) are
# intentionally absent: they fall through to the Tier-2 agent fallback until
# Phase 2. `follow`/`guide` reuse HRI's follow_person / nav helpers.
SKILLS = {
    "navigate": navigate,
    "find_object": find_object,
    "find_person": find_person,
    "follow": follow,
    "guide": guide,
    "count": count,
    "say": say,
    "greet": greet,
    "get_person_info": get_person_info,
    "get_object_property": get_object_property,
}
