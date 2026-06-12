"""HRI (receptionist) subtasks and the build_hri_task factory.

Flow (rulebook ch. 5.1, doorbell + torque sensing intentionally out of scope):
two guests arrive separately at the door; for each: greet, learn name +
favorite drink (no confirmation questions), guide to the living room, offer a
free seat; describe guest 1 to guest 2; introduce the seated guests to each
other while facing them. Bag handover / follow-host are gated stubs.

Blackboard layout (ctx.data):
    guests: {1: {"name", "drink", "appearance"}, 2: {...}}
    seats:  {guest_number: (SeatCandidate, img_w, world_xy | None)}
            # from the offer-seat scan; world_xy is the seat's map-frame
            # position when the 3D lift succeeded
    host:   {"appearance": str | None}  # captured during OfferSeat(1) — the
            # only seated person then is the host; name/drink come from
            # HRI_HOST_NAME / HRI_HOST_DRINK (known from the briefing)

Person identities (face + attire embeddings) live in ctx.people
(perception.PeopleStore) under stable ids "guest-1"/"guest-2"/"host" —
enrolled at the door (GreetAndLearn) and at the first seat scan (the host),
recognized again in IntroduceGuests so seat switches can't mis-aim the robot.
"""

from __future__ import annotations

import os
import time

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .identity import enroll_guest, enroll_person_in_box, locate_people
from .skills import (
    describe_seated_person,
    face_point,
    find_seated_person_bbox,
    heading_to_point,
    llm_pick_seat,
    parse_pose,
    scan_seats,
    bboxes_world_position,
)


def _guest(ctx: TaskContext, n: int) -> dict:
    return ctx.data.setdefault("guests", {}).setdefault(
        n, {"name": None, "drink": None, "appearance": None}
    )


def _bag_enabled() -> bool:
    return os.getenv("HRI_ENABLE_BAG", "0").lower() in ("1", "true", "yes")


class GoToDoor(SubTask):
    def __init__(self, guest: int):
        super().__init__(f"GoToDoor(guest {guest})")
        self.critical = guest == 1  # if nav is dead on step 1, nothing works

    def run(self, ctx: TaskContext) -> StepResult:
        x, y, heading = parse_pose(os.getenv("HRI_DOOR_POSE", "0.0,0.0,0"))
        return StepResult.DONE if ctx.goto(x, y, heading) else StepResult.RETRY


class GreetAndLearn(SubTask):
    """Greet at the door, learn name + drink, capture guest 1's appearance."""

    def __init__(self, guest: int):
        super().__init__(f"GreetAndLearn(guest {guest})")
        self.guest = guest

    def run(self, ctx: TaskContext) -> StepResult:
        record = _guest(ctx, self.guest)

        answer = ctx.ask(prompts.GREET_ASK_BOTH)
        info = ctx.extract(prompts.GuestInfo, prompts.EXTRACT_GUEST_INFO_INSTRUCTIONS, answer) if answer else None
        if info:
            record["name"], record["drink"] = info.name, info.drink

        # One targeted follow-up per missing field (asking for genuinely
        # missing info is not a penalized confirmation question).
        for field, question in (("name", prompts.ASK_MISSING_NAME), ("drink", prompts.ASK_MISSING_DRINK)):
            if record[field]:
                continue
            answer = ctx.ask(question, retries=0)
            info = ctx.extract(prompts.GuestInfo, prompts.EXTRACT_GUEST_INFO_INSTRUCTIONS, answer) if answer else None
            if info and getattr(info, field):
                record[field] = getattr(info, field)

        img = ctx.capture()
        # Visual description: only guest 1's is needed (told to guest 2 later).
        if self.guest == 1 and img is not None:
            try:
                record["appearance"] = ctx.walkieAI.image_caption.caption(
                    img, prompt=prompts.APPEARANCE_CAPTION_PROMPT
                )
            except Exception as exc:
                print(f"[HRI] appearance caption failed ({exc})")
        # Remember this guest's face + attire under a stable id, so the
        # introduction step can find them again even after a seat switch.
        if img is not None:
            enroll_guest(
                ctx, img, f"guest-{self.guest}",
                name=record["name"] or "", drink=record["drink"] or "",
            )

        name = record["name"] or "there"
        ctx.say(f"Nice to meet you, {name}!")
        return StepResult.DONE  # partial info still scores — never block here


class GuideToLivingRoom(SubTask):
    def __init__(self, guest: int):
        super().__init__(f"GuideToLivingRoom(guest {guest})")

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.FOLLOW_ME)
        x, y, heading = parse_pose(os.getenv("HRI_LIVING_ROOM_POSE", "0.0,0.0,0"))
        return StepResult.DONE if ctx.goto(x, y, heading) else StepResult.RETRY


class OfferSeat(SubTask):
    """Scan the room, let the LLM pick a seat and word the offer, face the
    seat, then speak."""

    def __init__(self, guest: int):
        super().__init__(f"OfferSeat(guest {guest})")
        self.guest = guest

    def run(self, ctx: TaskContext) -> StepResult:
        time.sleep(2) # wait for navigation to settle
        seats, persons, img = scan_seats(ctx)
        img_w = img.width if img is not None else 0
        host = ctx.data.setdefault("host", {})
        # First offer: the only seated person can be the host (guest 1 is
        # still standing next to the robot) — remember what they look like,
        # and enroll their face/attire as a distractor identity so a guest
        # match at introduction time must beat the host's.
        if self.guest == 1 and img is not None and not host.get("appearance"):
            host["appearance"] = describe_seated_person(ctx, img, persons, seats)
            host_box = find_seated_person_bbox(persons, seats)
            if host_box is not None:
                enroll_person_in_box(
                    ctx, img, host_box, "host",
                    name=os.getenv("HRI_HOST_NAME", "").strip(),
                    drink=os.getenv("HRI_HOST_DRINK", "").strip(),
                    attributes=host["appearance"] or "",
                )
        seat, announcement = llm_pick_seat(
            ctx, seats, persons, img_w,
            guest=self.guest,
            guest_name=_guest(ctx, self.guest)["name"],
            host_name=os.getenv("HRI_HOST_NAME", "").strip() or None,
            host_drink=os.getenv("HRI_HOST_DRINK", "").strip() or None,
            host_appearance=host.get("appearance"),
            prior_seats=ctx.data.get("seats"),
        )
        if seat is None:
            ctx.say(prompts.OFFER_SEAT_FALLBACK)
            return StepResult.DONE
        # Lift the seat to a map-frame point while the camera still sees the
        # scanned scene; facing then uses odometry + atan2 instead of the
        # pixel/HFOV approximation. Pixel facing stays as the fallback.
        world_xy = bboxes_world_position(ctx, seat.bbox_xyxy)
        ctx.data.setdefault("seats", {})[self.guest] = (seat, img_w, world_xy)
        faced = face_point(ctx, *world_xy) if world_xy else False
        if announcement:  # LLM-worded offer (may refer to the host)
            ctx.say(announcement)
        elif faced:
            # The rotation landed, so the seat is now centered ahead.
            ctx.say(prompts.OFFER_SEAT_TEMPLATE.format(
                seat_class=seat.class_name, direction=prompts.OFFER_SEAT_FACING))
        else:
            # No map-frame point to face (3D lift failed) — name the seat
            # without a stale left/right phrase and let the guest find it.
            ctx.say(prompts.OFFER_SEAT_FALLBACK)
        return StepResult.DONE


class ReceiveBag(SubTask):
    """Timed gripper handover (no torque sensing). Gated by HRI_ENABLE_BAG."""

    def run(self, ctx: TaskContext) -> StepResult:
        if not _bag_enabled():
            return StepResult.DONE
        wait_sec = float(os.getenv("HRI_BAG_HANDOVER_WAIT_SEC", "8"))
        ctx.say(prompts.BAG_ASK_HANDOVER)
        try:
            ctx.walkie.arm.control_gripper(1.0)  # open
            time.sleep(wait_sec)
            ctx.say(prompts.BAG_CLOSING_WARNING)
            time.sleep(3)
            ctx.walkie.arm.control_gripper(0.0)  # close
        except Exception as exc:
            print(f"[HRI] gripper handover failed ({exc})")
            return StepResult.DONE  # degrade: continue the flow bagless
        ctx.data["has_bag"] = True
        ctx.say(prompts.BAG_RECEIVED)
        return StepResult.DONE


class IntroduceGuests(SubTask):
    """Face each seated guest while stating the other guest's name + drink."""

    def run(self, ctx: TaskContext) -> StepResult:
        g1, g2 = _guest(ctx, 1), _guest(ctx, 2)
        seats: dict = ctx.data.get("seats", {})

        # Anchor each guest to where they actually ARE: recognize the enrolled
        # faces/attire in one scan frame (guests may have switched seats — the
        # rulebook allows it) and lift each matched person's bbox to a
        # map-frame point while the camera still sees the scanned scene. A
        # guest who isn't recognized falls back to their offered seat's stored
        # world point. World-point headings survive the robot's own rotations,
        # so both stay valid across both turns; a guest with neither anchor is
        # addressed without rotating.
        img = ctx.capture()
        located = (
            locate_people(ctx, img, ["guest-1", "guest-2"]) if img is not None else {}
        )
        headings: dict[int, float] = {}
        for n in (1, 2):
            world_xy = None
            box = located.get(f"guest-{n}")
            if box is not None:
                world_xy = bboxes_world_position(ctx, box)
            if world_xy is None and n in seats:
                _seat, _img_w, world_xy = seats[n]
            if world_xy is None:
                continue
            heading = heading_to_point(ctx, *world_xy)
            if heading is not None:
                headings[n] = heading

        def intro_line(listener: dict, other: dict) -> str:
            return prompts.INTRO_TEMPLATE.format(
                listener_name=listener["name"] or prompts.GENERIC_GUEST,
                other_name=other["name"] or prompts.GENERIC_GUEST,
                other_drink=other["drink"] or prompts.GENERIC_DRINK,
            )

        for listener_n, listener, other in ((1, g1, g2), (2, g2, g1)):
            if listener_n in headings:
                ctx.rotate_to(headings[listener_n])
            ctx.say(intro_line(listener, other))
        return StepResult.DONE


class FollowHostAndDropBag(SubTask):
    """Extension point: needs a person-following primitive that doesn't exist
    yet. For now, announces the limitation and releases the bag in place."""

    def run(self, ctx: TaskContext) -> StepResult:
        if not (_bag_enabled() and ctx.data.get("has_bag")):
            return StepResult.DONE
        ctx.say(prompts.FOLLOW_HOST_NOT_AVAILABLE)
        try:
            ctx.walkie.arm.control_gripper(1.0)  # open: release the bag
        except Exception as exc:
            print(f"[HRI] bag release failed ({exc})")
        return StepResult.DONE


def build_hri_task(ctx: TaskContext) -> Task:
    # Guests differ every run — stale identities must never match today's.
    if ctx.people is not None and os.getenv("HRI_PEOPLE_RESET", "1").lower() in ("1", "true", "yes"):
        try:
            ctx.people.clear()
            print("[HRI] people memory cleared for a fresh run")
        except Exception as exc:
            print(f"[HRI] people memory reset failed ({exc})")
    return Task(
        "HRI",
        [
            GoToDoor(1),
            GreetAndLearn(1),
            GuideToLivingRoom(1),
            OfferSeat(1),
            GoToDoor(2),
            GreetAndLearn(2),
            ReceiveBag(),
            GuideToLivingRoom(2),
            OfferSeat(2),
            IntroduceGuests(),
            FollowHostAndDropBag(),
        ],
        ctx,
    )
