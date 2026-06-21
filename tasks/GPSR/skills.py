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

import math
import os
import re
import time
from datetime import datetime

from langchain.messages import HumanMessage

from tasks.base import TaskContext
from tasks.skills import (
    approach_point,
    cxcywh_to_xyxy,
    face_point,
    follow_person,
    go_to_through_door,
    lift_bbox_world_xy,
    select_largest_person,
)

from . import gestures
from .plan import PlanStep, _person_phrase
from .tracking import ArrivalStopper, companion_present, heading_between, segment_route
from .world import WorldModel


# --- low-level helpers ------------------------------------------------------

def _conf_floor() -> float:
    """Minimum detector confidence for an object detection to count (0 = keep all).

    Open-vocab detectors emit low-confidence boxes even for an absent class; with
    no floor those inflate ``count`` (which is ``len(dets)``) and let
    ``find_object`` call a spurious box a "found" object. The floor drops them.
    Default 0 reproduces the pre-gate behaviour exactly — the useful value is a
    robot-tuning question (try ~0.3 once the real detector's score distribution is
    known), so it ships off by default and is set per-arena like the other knobs.
    """
    try:
        return float(os.getenv("GPSR_DETECT_CONF_MIN", "0") or 0)
    except ValueError:
        return 0.0


def _above_floor(dets, floor: float):
    """Keep only detections at/above the confidence floor (missing conf -> 0).

    Pure (no robot/LLM) so it is unit-tested directly. ``floor <= 0`` is a no-op
    short-circuit that returns every detection — the default, behaviour-preserving
    path."""
    if floor <= 0:
        return list(dets)
    return [d for d in dets if (getattr(d, "confidence", None) or 0.0) >= floor]


def _detect(ctx: TaskContext, img, classes: list[str], *, return_mask: bool = False):
    try:
        dets = ctx.walkieAI.image.detect(img, prompts=classes, return_mask=return_mask)
    except Exception as exc:
        print(f"[gpsr.skill] object detection failed ({exc})")
        return []
    return _above_floor(dets, _conf_floor())


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
    # If a closed door blocks the route, ask a human to open it and retry
    # (go_to_through_door only asks when the way is actually blocked + not seen
    # open; gate OFF with GPSR_NAV_DOOR_RETRY=0 if it false-asks on the robot).
    # For a place flagged `barrier = true` in world.toml (a door/partition that the
    # depth check reads as "open" because it can't see a too-narrow gap), ask on the
    # block even when depth says open — nav success is the real signal.
    if os.getenv("GPSR_NAV_DOOR_RETRY", "1").lower() in ("1", "true", "yes"):
        ok = go_to_through_door(ctx, *pose, ask_even_if_open=world.is_barrier(name))
    else:
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


# --- counting / measurement helpers (pure -> unit-tested) -------------------

def _median_count(counts: list[int]) -> int:
    """Median of per-frame detection counts, rounded to int. The open-vocab
    detector flickers a spurious/dropped box frame to frame, so a single-shot
    ``len(dets)`` over- or under-counts; the median across a few frames is robust
    to one bad frame. Empty -> 0."""
    if not counts:
        return 0
    s = sorted(counts)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return round((s[mid - 1] + s[mid]) / 2)


def _count_objects_stable(ctx: TaskContext, classes: list[str]) -> int | None:
    """Count detections of *classes* in the current view, stabilized over a few
    frames at the same heading (objects on a placement are static, so we re-shoot
    rather than scan). Returns the per-frame median count, or None if not one
    frame was captured (caller -> Tier-2)."""
    frames = max(1, int(os.getenv("GPSR_COUNT_OBJ_FRAMES", "3")))
    settle = float(os.getenv("GPSR_COUNT_OBJ_SETTLE_SEC", "0.2"))
    counts: list[int] = []
    for i in range(frames):
        if i > 0 and settle > 0:
            time.sleep(settle)
        snap = ctx.snapshot()
        if snap is None:
            continue
        counts.append(len(_detect(ctx, snap.img, classes)))
    if not counts:
        return None
    return _median_count(counts)


_SUPERLATIVE_LARGE = ("biggest", "largest", "heaviest", "thickest", "tallest", "longest")
_SUPERLATIVE_SMALL = ("smallest", "thinnest", "lightest", "tiniest", "shortest")


def _superlative_dir(text: str | None) -> str | None:
    """'large' for a biggest/largest/... query, 'small' for smallest/thinnest/...,
    else None. The parser carries the superlative DIRECTION only in the raw clause
    (the object grounds to a generic placeholder), so the skill reads it from there
    to pick the right object by image-size."""
    t = (text or "").lower()
    if any(w in t for w in _SUPERLATIVE_SMALL):
        return "small"
    if any(w in t for w in _SUPERLATIVE_LARGE):
        return "large"
    return None


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _pick_by_size(dets, direction: str):
    """The largest ('large') or smallest ('small') detection by image-bbox area —
    a proxy for physical size when the world has no measurements. Pure -> tested."""
    return (max if direction == "large" else min)(dets, key=lambda d: _bbox_area(d.bbox))


# --- primitives -------------------------------------------------------------

def navigate(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    return go_to_named(ctx, step.args.get("target"), world, state)


def _memory_graphs(ctx: TaskContext):
    """The walkie_graphs CLIP scene memory for object recall, or None when
    GPSR_FIND_USE_MEMORY is off / perception isn't running. Object *positions* come
    from this scene graph — world.toml holds only fixed places, and objects move /
    vary, so a static world lookup can't locate them."""
    if os.getenv("GPSR_FIND_USE_MEMORY", "1") != "1":
        return None
    brain = getattr(ctx, "data", {}).get("brain")
    return getattr(brain, "graphs", None) if brain is not None else None


def _detect_here(ctx: TaskContext, obj, category, world: WorldModel):
    """Look for `obj` in the current camera view. Returns (status, xy): status is
    'found' | 'none' | 'no_frame'; xy is the lifted world (x, y) or None."""
    snap = ctx.snapshot()
    if snap is None:
        return "no_frame", None
    classes = _object_classes(obj, category, world) or ["object"]
    dets = _detect(ctx, snap.img, classes)
    if not dets:
        return "none", None
    best = max(dets, key=lambda d: d.confidence or 0.0)
    return "found", lift_bbox_world_xy(ctx, snap, best.bbox)


def _stash_found(state: dict, obj, xy) -> None:
    """Record a found object's map point for a later step of the same command."""
    if xy:
        state["target_xy"] = xy
        state["found_object"] = obj


def _find_via_memory(ctx: TaskContext, obj, category, label: str, world: WorldModel, state: dict) -> bool:
    """Fallback (option A): recall where `obj` was last seen from the scene graph,
    drive there, and confirm with a live detection. The position comes from the
    CLIP scene memory, not world.toml. True only if the re-detect confirms it (so a
    stale recall never makes a false 'found' claim)."""
    graphs = _memory_graphs(ctx)
    if graphs is None:
        return False
    try:
        hits = graphs.query_text(obj or label, k=1)
    except Exception as exc:
        print(f"[gpsr.skill] scene-graph query failed ({exc})")
        return False
    if not hits:
        return False
    cx, cy = float(hits[0].centroid[0]), float(hits[0].centroid[1])
    print(f"[gpsr.skill] find_object: memory recalls {label!r} near ({cx:.2f}, {cy:.2f}); approaching")
    state.pop("at", None)  # we leave the named place -> nav-dedup must re-evaluate
    approach_point(ctx, cx, cy, stop_distance=float(os.getenv("GPSR_FIND_APPROACH_M", "0.8")))
    status, xy = _detect_here(ctx, obj, category, world)
    if status == "found":
        _stash_found(state, obj, xy)
        return True
    return False


def find_object(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    """Find an object. Primary: where the command says (or the current view if it
    named no place). On a miss — or when no place was named — fall back to the
    scene-graph memory (option A) for where it was last seen, then confirm with a
    live detection. Object positions come from the scene graph, not world.toml."""
    obj = step.args.get("object")
    category = step.args.get("category")
    where = step.args.get("location") or step.args.get("room")
    label = (obj or "object").replace("_", " ")

    if where:
        go_to_named(ctx, where, world, state)
    status, xy = _detect_here(ctx, obj, category, world)
    if status == "no_frame":
        return False  # camera failure -> Tier-2 (don't drive blind on a recall)
    if status == "found":
        _stash_found(state, obj, xy)
        ctx.say(f"I found the {label}.")
        return True
    if _find_via_memory(ctx, obj, category, label, world, state):
        ctx.say(f"I found the {label}.")
        return True
    ctx.say(f"I could not find the {label}.")
    return True  # the search ran; nothing to find is an honest result


def _pick_index(reply: str | None, n: int) -> int | None:
    """First integer in *reply*, if it is a valid 0..n-1 candidate index."""
    m = re.search(r"-?\d+", reply or "")
    if not m:
        return None
    i = int(m.group())
    return i if 0 <= i < n else None


_GENERIC_PERSON_WORDS = {
    "person", "people", "someone", "somebody", "anyone", "anybody",
    "human", "guy", "man", "woman", "girl", "boy", "individual", "",
}


def _is_generic_person(descriptor: str) -> bool:
    """True when the descriptor is a bare person-noun, not a distinguishing
    attribute. ``find a person`` parses to a *clothing* descriptor of "a person";
    that word names nobody's attire, so the attire LLM rejects every candidate and
    a present person reads as "not found". Detect it so we fall back to the nearest
    person instead. Strips a leading article (a/an/the)."""
    d = re.sub(r"^(a|an|the)\s+", "", (descriptor or "").strip().lower())
    return d in _GENERIC_PERSON_WORDS


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


def _detect_person_in_view(ctx: TaskContext, step: PlanStep):
    """One snapshot at the CURRENT heading: detect + match per descriptor kind.

    Matching is per descriptor *kind* (parse._ground_person):
      gesture/pose -> COCO-keypoint heuristics (gestures.py);
      clothing     -> caption each candidate and LLM-pick the best attire match;
      name         -> GPSR enrolls no face gallery, so identity can't be
                      verified; fall back to the nearest person (best effort) and
                      let the caller address them by name;
      (none given) -> the nearest person.
    snap is None only on a camera failure; bbox is None when nobody in this view
    matches.
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
        for i, p in enumerate(people):
            conf = [k.name for k in p.keypoints if k.confidence >= gestures._conf()]
            print(f"[gpsr.skill] person {i}: gestures={sorted(gestures.classify_gestures(p))} "
                  f"want={descriptor!r} match={gestures.matches_gesture(p, descriptor)} conf_kps={conf}")
        people = [p for p in people if gestures.matches_gesture(p, descriptor)]
        if not people:
            return snap, None  # nobody in this view is doing that gesture/pose

    elif kind == "clothing" and descriptor and not _is_generic_person(descriptor):
        match = _match_attire(ctx, snap, people, descriptor)
        return snap, (cxcywh_to_xyxy(match.bbox) if match is not None else None)

    # generic ("a person") / name / no descriptor / gesture-matched: nearest person.
    target = max(people, key=lambda p: p.bbox[2] * p.bbox[3])
    return snap, cxcywh_to_xyxy(target.bbox)


def _find_person_match(ctx: TaskContext, step: PlanStep):
    """Find a person matching the descriptor, scanning the room if the forward
    view is empty. Returns (snap, person_bbox_xyxy) or (snap, None)/(None, None).

    The arrival heading rarely points the camera straight at the person, so a
    single forward frame misses anyone off to the side or behind. When the forward
    view comes up empty we rotate in place through a full turn
    (``GPSR_PERSON_SCAN_STEPS`` stops, default 6 -> 60 deg each), re-detecting at
    each stop and stopping early on the first match. People are static for ``find``
    (unlike ``follow``'s moving target), so an in-place spin covers the room; we
    end back near the arrival heading when nobody is found.
    """
    snap, bbox = _detect_person_in_view(ctx, step)
    if snap is None or bbox is not None:
        return snap, bbox

    steps = int(os.getenv("GPSR_PERSON_SCAN_STEPS", "6"))
    if steps <= 0:
        return snap, None  # scanning disabled -> honest negative from the forward view
    # rotate_to blocks until the base reaches the heading, but the base still sways
    # for a moment and the camera pipeline lags, so an immediate frame is motion-
    # blurred (or stale from mid-rotation) and the pose detector misses. Let it
    # settle before grabbing the frame.
    settle = float(os.getenv("GPSR_PERSON_SCAN_SETTLE_SEC", "0.7"))
    base = ctx.current_pose()["heading"]
    ctx.say("I do not see anyone here yet; let me look around.")
    for i in range(1, steps + 1):
        target = base + 2 * math.pi * i / steps
        ctx.rotate_to(math.atan2(math.sin(target), math.cos(target)))  # normalize to [-pi, pi]
        if settle > 0:
            time.sleep(settle)
        snap_i, bbox_i = _detect_person_in_view(ctx, step)
        if snap_i is not None:
            snap = snap_i
        if bbox_i is not None:
            return snap, bbox_i
    return snap, None


def find_person(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    where = step.args.get("location") or step.args.get("room")  # room or a beacon
    if where:
        go_to_named(ctx, where, world, state)
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


def _count_people_scanning(ctx: TaskContext, step: PlanStep) -> int | None:
    """Rotate in place through a full turn, detecting people at each stop and
    counting UNIQUE bodies. People are spread around a room, so a single forward
    frame at the arrival heading misses everyone off to the side — we scan like
    `_find_person_match` does. Adjacent frames overlap (the camera FOV is wider
    than the per-stop rotation), so a naive sum double-counts; we dedup by each
    person's lifted world (x, y) within `GPSR_COUNT_DEDUP_M`. Honours the gesture
    filter for "how many sitting/waving people". Returns the count, or None on a
    total camera failure (no frame at any stop) so the caller can fall to Tier-2.
    """
    gesture = step.args.get("descriptor") if step.args.get("kind") in ("gesture", "pose") else None
    steps = max(1, int(os.getenv("GPSR_COUNT_SCAN_STEPS", os.getenv("GPSR_PERSON_SCAN_STEPS", "6"))))
    settle = float(os.getenv("GPSR_PERSON_SCAN_SETTLE_SEC", "0.7"))
    dedup_m = float(os.getenv("GPSR_COUNT_DEDUP_M", "0.4"))
    base = ctx.current_pose()["heading"]
    seen: list[tuple[float, float]] = []
    unlocated = 0  # matched people we couldn't lift to a world point -> can't dedup
    got_frame = False
    for i in range(steps):
        if i > 0:  # i == 0 is the arrival heading; no need to rotate first
            target = base + 2 * math.pi * i / steps
            ctx.rotate_to(math.atan2(math.sin(target), math.cos(target)))  # normalize to [-pi, pi]
            if settle > 0:
                time.sleep(settle)
        snap = ctx.snapshot()
        if snap is None:
            continue
        got_frame = True
        people = _people(ctx, snap.img)
        if gesture:
            people = [p for p in people if gestures.matches_gesture(p, gesture)]
        for p in people:
            xy = lift_bbox_world_xy(ctx, snap, cxcywh_to_xyxy(p.bbox))
            if xy is None:
                unlocated += 1
                continue
            if not any(math.dist(xy, s) < dedup_m for s in seen):
                seen.append(xy)
    ctx.rotate_to(math.atan2(math.sin(base), math.cos(base)))  # restore arrival heading
    if not got_frame:
        return None
    return len(seen) + unlocated


def count(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    where = step.args.get("location") or step.args.get("room")
    if where:
        go_to_named(ctx, where, world, state)
    if step.args.get("what") == "persons":
        # People scatter around a room -> scan the full turn and count unique bodies.
        total = _count_people_scanning(ctx, step)
        if total is None:
            return False  # no camera frame at any stop -> Tier-2
        descriptor = step.args.get("descriptor")
        if step.args.get("kind") in ("gesture", "pose") and descriptor:
            ctx.say(f"I count {total} {descriptor.replace('_', ' ')} people.")
        else:
            ctx.say(f"I count {total} people.")
        return True
    # Objects sit on a single placement we're already facing -> re-shoot the same
    # view a few times and take the median count (kills detector flicker).
    classes = _object_classes(step.args.get("object"), step.args.get("category"), world) or ["object"]
    total = _count_objects_stable(ctx, classes)
    if total is None:
        return False  # no camera frame -> Tier-2
    noun = (step.args.get("object") or step.args.get("category") or "objects").replace("_", " ")
    ctx.say(f"I count {total} {noun}.")
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
    where = step.args.get("location") or step.args.get("room")  # room or a beacon
    if where:
        go_to_named(ctx, where, world, state)
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
    # A "biggest/smallest … object on the placement" query: the parser leaves the
    # object generic (obj is None) and carries the direction in the raw clause.
    # Detect over all candidate objects, pick by image-size, and NAME the winner
    # (not a property of an arbitrary box). A concrete named object (obj set) or a
    # non-size property falls through to the plain describe path.
    direction = _superlative_dir(step.raw) if which == "size" and not obj else None
    if direction:
        classes = [o.replace("_", " ") for o in world.objects] or ["object"]
        dets = _detect(ctx, snap.img, classes)
        if not dets:
            ctx.say("I could not find any objects there.")
            return True
        winner = _pick_by_size(dets, direction)
        word = "largest" if direction == "large" else "smallest"
        name = _caption(ctx, _crop(snap.img, winner.bbox),
                        "In two or three words, name the object in this image.")
        ctx.say(f"The {word} object I see is {name}." if name
                else f"I see the {word} object but cannot identify it.")
        return True
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


def _check_follower(ctx: TaskContext, came_from) -> None:
    """Mid-lead: turn back toward ``came_from`` and (bounded) wait for the guided
    person to re-appear in the forward frame. Best-effort — after the retries we
    lead on regardless, since arriving with a chance they catch up beats aborting
    (which also scores nothing). Robot-side (turn + camera + sleep)."""
    here = ctx.current_pose()
    ctx.rotate_to(heading_between((here["x"], here["y"]), came_from))
    if companion_present(ctx):
        return
    misses = int(os.getenv("GPSR_GUIDE_MAX_MISSES", "3"))
    wait_s = float(os.getenv("GPSR_GUIDE_REACQUIRE_WAIT_SEC", "2.0"))
    for _ in range(max(0, misses)):
        ctx.say("Please keep up with me; I will wait for you.")
        time.sleep(wait_s)
        if companion_present(ctx):
            ctx.say("Thank you. Let us continue.")
            return
    print("[gpsr.skill] guide: follower not re-acquired; leading on best-effort")


def _lead_with_reacquire(ctx: TaskContext, dest_pose) -> bool:
    """Lead to ``dest_pose`` in segments, looking back between hops to keep the
    trailing follower along (the guide mid-route re-acquire). Each hop drives
    facing the direction of travel; at the destination it faces the surveyed
    heading. Robot-side. Returns False if any hop's nav fails."""
    here = ctx.current_pose()
    prev = (here["x"], here["y"])
    end = (dest_pose[0], dest_pose[1])
    final_heading = dest_pose[2] if len(dest_pose) > 2 else 0.0
    seg_m = float(os.getenv("GPSR_GUIDE_SEGMENT_M", "2.0"))
    waypoints = segment_route(prev, end, seg_m)
    for i, wp in enumerate(waypoints):
        last = i == len(waypoints) - 1
        heading = final_heading if last else heading_between(prev, wp)
        if not ctx.goto(wp[0], wp[1], heading):
            return False
        if not last:
            _check_follower(ctx, prev)
        prev = wp
    return True


def _lead_to(ctx: TaskContext, to_name, world: WorldModel, state: dict) -> bool:
    """Drive to ``to_name`` while leading a follower. With GPSR_GUIDE_REACQUIRE on,
    lead in segments with mid-route look-back; otherwise one blocking drive
    (``go_to_named``, the legacy behaviour). Honours the per-command nav dedup."""
    if os.getenv("GPSR_GUIDE_REACQUIRE", "0") != "1":
        return go_to_named(ctx, to_name, world, state)
    if state is not None and state.get("at") == to_name:
        return True  # already here this command — nothing to lead
    pose = world.location_pose(to_name)
    if pose is None:
        print(f"[gpsr.skill] no pose for {to_name!r}")
        return False
    ok = _lead_with_reacquire(ctx, pose)
    if ok and state is not None:
        state["at"] = to_name
    return ok


def guide(ctx: TaskContext, step: PlanStep, world: WorldModel, state: dict) -> bool:
    """Guide (lead) a person to a destination — nav + best-effort person confirm.

    Optionally goes to ``from`` first (where the person is), confirms/faces them,
    then leads to ``to`` and announces arrival. We lead with our back to the
    person, so a forward-facing arrival frame can't confirm they kept up — with
    GPSR_GUIDE_REACQUIRE on, `_lead_to` instead leads in segments and looks back
    between hops to re-acquire a trailing follower (`_check_follower`). Returns
    False (→ Tier-2) only when the destination is missing or unreachable —
    matching `navigate`.
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
    if not to or not _lead_to(ctx, to, world, state):
        ctx.say("I am sorry, I could not find the way there.")
        return False
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
