"""HRI (receptionist) subtasks and the build_hri_task factory.

Flow (rulebook ch. 5.1, doorbell + torque sensing intentionally out of scope):
two guests arrive separately at the door; for each: greet, learn name +
favorite drink (no confirmation questions), guide to the living room, offer a
free seat; then introduce everyone — host first, then both guests — facing
each person while speaking their LLM-worded introduction (one LLM call words
all three). Bag handover / follow-host are gated stubs.

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
    lift_bbox_world_xy,
    llm_intro_speeches,
    llm_pick_seat,
    match_people_to_seats,
    parse_pose,
    scan_seats,
    sweep_snapshots,
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
        time.sleep(5) # wait for navigation to settle

        original_head_tilt = ctx.walkie.robot.head.get_angle()
        ctx.walkie.robot.head.tilt(0)
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
        ctx.walkie.robot.head.tilt(original_head_tilt)  # restore the head tilt for better nav
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
        time.sleep(5) # wait for navigation to settle
        seats, persons, snap = scan_seats(ctx)
        img = snap.img if snap is not None else None
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
        # Recognize who is already seated (everyone enrolled except the guest
        # standing next to the robot) and tie each to the seat they hold, so the
        # picker seats the new guest near the host and steers clear of taken
        # seats. A recognized person the seat detector found no chair under is
        # surfaced too (match_people_to_seats' seatless map), so that spot still
        # isn't offered as free.
        seated_ids = [
            pid for pid in ("host", "guest-1", "guest-2")
            if pid != f"guest-{self.guest}"
        ]
        located = locate_people(ctx, [img], seated_ids) if img is not None else {}
        seat_occupants, seatless = match_people_to_seats(located, seats)
        person_names = {
            "host": os.getenv("HRI_HOST_NAME", "").strip() or None,
            "guest-1": _guest(ctx, 1)["name"],
            "guest-2": _guest(ctx, 2)["name"],
        }
        seat, announcement = llm_pick_seat(
            ctx, seats, persons, img_w,
            guest=self.guest,
            guest_name=_guest(ctx, self.guest)["name"],
            host_name=os.getenv("HRI_HOST_NAME", "").strip() or None,
            host_drink=os.getenv("HRI_HOST_DRINK", "").strip() or None,
            host_appearance=host.get("appearance"),
            prior_seats=ctx.data.get("seats"),
            seat_occupants=seat_occupants,
            seatless_people=seatless,
            person_names=person_names,
        )
        if seat is None:
            ctx.say(prompts.OFFER_SEAT_FALLBACK)
            return StepResult.DONE
        # Lift the seat to a map-frame point against the SCAN-TIME geometry
        # frozen in the snapshot — exact despite the slow llm_pick_seat call
        # above; facing then uses odometry + atan2.
        world_xy = lift_bbox_world_xy(ctx, snap, seat.bbox_xyxy)
        ctx.data.setdefault("seats", {})[self.guest] = (seat, img_w, world_xy)
        ctx.walkie.robot.arm.left.go_to_pose_relative([0.3, 0, 0.2], [0, -1.57, 0], blocking=False)
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
        ctx.walkie.robot.arm.left.go_to_home(blocking=False)  # reset the arm for better nav after facing
        return StepResult.DONE


class ReceiveBag(SubTask):
    """Timed gripper handover (no torque sensing). Gated by HRI_ENABLE_BAG."""

    def run(self, ctx: TaskContext) -> StepResult:
        if not _bag_enabled():
            return StepResult.DONE
        ctx.say(prompts.BAG_ASK_HANDOVER)
        ctx.walkie.robot.arm.left.gripper(1.0, blocking=False)  # open: ready to receive
        ctx.walkie.robot.arm.left.go_to_pose([0.5, 0.15, 1.1], [0, -0.6, 3.14], blocking=True)

        _, _, efforts = ctx.walkie.robot.arm.left.get_joint_states()
        initial_effort = efforts[3] if efforts else 0.0
        # Get joint 4's effort, which spikes when the bag is placed in the hand.
        threshold = float(os.getenv("HRI_BAG_EFFORT_THRESHOLD", "1.0").strip())
        # Give up waiting after this long so a no-show guest can't stall the run.
        timeout = float(os.getenv("HRI_BAG_WAIT_TIMEOUT_SEC", "20").strip())

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, _, efforts = ctx.walkie.robot.arm.left.get_joint_states()
            print(f"[HRI] waiting for bag handover... joint 4 effort: {abs(efforts[3] - initial_effort) if efforts else 'N/A'}")
            if abs(efforts[3] - initial_effort) > threshold:
                break

        ctx.data["has_bag"] = True
        time.sleep(1.5)  # let the nav settle after the arm movement and possible wait
        ctx.walkie.robot.arm.left.go_to_home(pose_name="standby", blocking=False)  # reset the arm for better nav after receiving
        ctx.walkie.robot.arm.left.gripper(0.0)  # open: ready to receive
        ctx.say(prompts.BAG_RECEIVED)
        return StepResult.DONE


class IntroduceEveryone(SubTask):
    """Introduce each person — host first, then both guests — facing each one.

    All three introductions are worded by ONE LLM call (llm_intro_speeches);
    the robot then turns to the person being introduced and speaks their part.
    """

    ORDER = ("host", "guest-1", "guest-2")

    def run(self, ctx: TaskContext) -> StepResult:
        host = ctx.data.setdefault("host", {})
        people = {
            "host": {
                "name": os.getenv("HRI_HOST_NAME", "").strip() or None,
                "drink": os.getenv("HRI_HOST_DRINK", "").strip() or None,
                "appearance": host.get("appearance"),
            },
            "guest-1": _guest(ctx, 1),
            "guest-2": _guest(ctx, 2),
        }

        # Anchor each person to where they actually ARE. Guests may have
        # switched seats (the rulebook allows it), and a single forward frame
        # can miss someone off to the side, so sweep three snapshots — left
        # 10°, center, right 10° — and recognize the enrolled people across all
        # of them. Faces are matched FIRST over the whole sweep; only people
        # still missing fall back to an attire-only pass (covers a guest turned
        # away in every view). Each match is lifted against the geometry of the
        # very snapshot it was found in, so the slow recognition round-trips and
        # the robot's own sweep rotations can't skew the map-frame positions. A
        # guest still not found falls back to their offered seat's stored world
        # point. World-point headings survive the robot's own rotations, so they
        # stay valid across all three turns; a person with no anchor is
        # introduced without rotating.
        sweep = float(os.getenv("HRI_INTRO_SWEEP_DEG", "10"))
        snaps = sweep_snapshots(ctx, (sweep, 0.0, -sweep))
        located = (
            locate_people(ctx, [s.img for s in snaps], list(self.ORDER))
            if snaps else {}
        )
        seats: dict = ctx.data.get("seats", {})
        headings: dict[str, float] = {}
        for pid in self.ORDER:
            world_xy = None
            found = located.get(pid)
            if found is not None:
                fi, box = found
                world_xy = lift_bbox_world_xy(ctx, snaps[fi], box)
            if world_xy is None and pid.startswith("guest-"):
                stored = seats.get(int(pid.removeprefix("guest-")))
                if stored is not None:
                    _seat, _img_w, world_xy = stored
            if world_xy is None:
                continue
            heading = heading_to_point(ctx, *world_xy)
            if heading is not None:
                headings[pid] = heading

        speeches = llm_intro_speeches(ctx, people)
        for pid in self.ORDER:
            if pid in headings:
                ctx.rotate_to(headings[pid])
            ctx.say(speeches[pid])
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
            IntroduceEveryone(),
            FollowHostAndDropBag(),
        ],
        ctx,
    )
