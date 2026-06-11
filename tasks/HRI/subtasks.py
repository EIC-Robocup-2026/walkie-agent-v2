"""HRI (receptionist) subtasks and the build_hri_task factory.

Flow (rulebook ch. 5.1, doorbell + torque sensing intentionally out of scope):
two guests arrive separately at the door; for each: greet, learn name +
favorite drink (no confirmation questions), guide to the living room, offer a
free seat; describe guest 1 to guest 2; introduce the seated guests to each
other while facing them. Bag handover / follow-host are gated stubs.

Blackboard layout (ctx.data):
    guests: {1: {"name", "drink", "appearance"}, 2: {...}}
    seats:  {guest_number: (SeatCandidate, img_w)}  # from the offer-seat scan
"""

from __future__ import annotations

import os
import time

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .skills import (
    direction_phrase,
    face_pixel,
    find_persons,
    heading_to_pixel,
    parse_pose,
    pick_free_seat,
    scan_seats,
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

        # Visual description: only guest 1's is needed (told to guest 2 later).
        if self.guest == 1:
            img = ctx.capture()
            if img is not None:
                try:
                    record["appearance"] = ctx.walkieAI.image_caption.caption(
                        img, prompt=prompts.APPEARANCE_CAPTION_PROMPT
                    )
                except Exception as exc:
                    print(f"[HRI] appearance caption failed ({exc})")

        name = record["name"] or "there"
        ctx.say(f"Nice to meet you, {name}!")
        return StepResult.DONE  # partial info still scores — never block here


class DescribeGuestOneToGuestTwo(SubTask):
    """At the door, tell guest 2 who they will meet inside."""

    def run(self, ctx: TaskContext) -> StepResult:
        g1 = _guest(ctx, 1)
        name = g1["name"] or prompts.GENERIC_GUEST
        appearance = g1["appearance"] or ""
        ctx.say(prompts.DESCRIBE_GUEST1_TEMPLATE.format(name=name, appearance=appearance).strip())
        return StepResult.DONE


class GuideToLivingRoom(SubTask):
    def __init__(self, guest: int):
        super().__init__(f"GuideToLivingRoom(guest {guest})")

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.FOLLOW_ME)
        x, y, heading = parse_pose(os.getenv("HRI_LIVING_ROOM_POSE", "0.0,0.0,0"))
        return StepResult.DONE if ctx.goto(x, y, heading) else StepResult.RETRY


class OfferSeat(SubTask):
    """Scan for a free seat, face it, and announce it to the guest."""

    def __init__(self, guest: int):
        super().__init__(f"OfferSeat(guest {guest})")
        self.guest = guest

    def run(self, ctx: TaskContext) -> StepResult:
        seats, img_w = scan_seats(ctx)
        seat = pick_free_seat(seats)
        if seat is None:
            ctx.say(prompts.OFFER_SEAT_FALLBACK)
            return StepResult.DONE
        ctx.data.setdefault("seats", {})[self.guest] = (seat, img_w)
        direction = direction_phrase(seat.center_px[0], img_w)
        face_pixel(ctx, seat.center_px[0], img_w)  # best-effort look-at
        ctx.say(prompts.OFFER_SEAT_TEMPLATE.format(seat_class=seat.class_name, direction=direction))
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

        # Anchor each guest to a pixel column: the live person nearest their
        # offered seat, falling back to the stored seat center itself.
        anchors: dict[int, tuple[float, int]] = {}  # guest -> (px_x, img_w)
        persons = find_persons(ctx)
        for n in (1, 2):
            if n not in seats:
                continue
            seat, img_w = seats[n]
            px = seat.center_px[0]
            if persons:
                nearest = min(persons, key=lambda p: abs(p.bbox[0] - px))
                if abs(nearest.bbox[0] - px) < img_w / 4:  # plausibly on that seat
                    px = nearest.bbox[0]
            anchors[n] = (px, img_w)

        # Compute both headings from the same pose BEFORE the first rotation —
        # rotating invalidates the pixel->heading mapping of the second anchor.
        headings = {
            n: heading_to_pixel(ctx, px, img_w) for n, (px, img_w) in anchors.items()
        }

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
            DescribeGuestOneToGuestTwo(),
            GuideToLivingRoom(2),
            OfferSeat(2),
            IntroduceGuests(),
            FollowHostAndDropBag(),
        ],
        ctx,
    )
