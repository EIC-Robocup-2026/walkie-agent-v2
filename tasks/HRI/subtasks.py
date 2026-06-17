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

import math
import os
import time

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .identity import enroll_guest, enroll_person_in_box, locate_people
from .skills import (
    CommandListener,
    FaceTracker,
    classify_host_command,
    describe_seated_person,
    face_point,
    find_seated_person_bbox,
    follow_target,
    heading_to_point,
    lift_bbox_world_xy,
    llm_intro_speeches,
    llm_pick_seat,
    match_people_to_seats,
    nearest_person_sample,
    parse_pose,
    person_bboxes,
    scan_seats,
    sweep_snapshots,
    wait_for_person,
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
        if not ctx.goto(x, y, heading):
            return StepResult.RETRY
        # Wait for the guest to come stand in front before greeting. Look
        # straight ahead so a standing person's face is in frame (nav may have
        # left the head tilted down), then poll for a face bigger than the area
        # floor; restore the tilt afterwards for the greeting/next nav.
        original_tilt = ctx.walkie.robot.head.get_angle()
        ctx.walkie.robot.head.tilt(0)
        if wait_for_person(ctx):
            print("[HRI] guest detected at the door")
        else:
            print("[HRI] no guest detected before timeout; greeting anyway")
        ctx.walkie.robot.head.tilt(original_tilt if original_tilt is not None else 0.0)
        return StepResult.DONE


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
        # Keep the base turned toward the guest's face for the whole exchange:
        # a background loop tracks the biggest face in view and rotates the base
        # to re-center it while we ask questions and caption below.
        with FaceTracker(ctx):
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
        ctx.walkie.robot.arm.go_to_pose_relative([0.3, 0, 0.2], [0, -1.57, 0], group_name="left_arm", blocking=False)
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
        ctx.walkie.robot.arm.go_to_pose(0.45, 0.16, 1.15, -0.8, 0, -1.57, group_name="left_arm", blocking=True)

        _, _, efforts = ctx.walkie.robot.arm.left.get_joint_states()
        initial_effort = efforts[3] if efforts else 0.0
        # Get joint 4's effort, which spikes when the bag is placed in the hand.
        threshold = float(os.getenv("HRI_BAG_EFFORT_THRESHOLD", "1.0").strip())
        # Give up waiting after this long so a no-show guest can't stall the run.
        timeout = float(os.getenv("HRI_BAG_WAIT_TIMEOUT_SEC", "20").strip())

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, _, efforts = ctx.walkie.robot.arm.left.get_joint_states()
            # print(f"[HRI] waiting for bag handover... joint 4 effort: {abs(efforts[3] - initial_effort) if efforts else 'N/A'}")
            if abs(efforts[3] - initial_effort) > threshold:
                break
        ctx.data["has_bag"] = True
        ctx.say(prompts.BAG_CLOSING_WARNING)
        time.sleep(1)  # let the nav settle after the arm movement and possible wait
        ctx.walkie.robot.arm.go_to_pose(0.3, 0.16, 1.15, -0.8, 0, -1.57, group_name="left_arm", blocking=True)
        ctx.walkie.robot.arm.left.gripper(0.0)  # close
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


def _box_area(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


class _HostSampler:
    """Two-tier host-position sampler for :class:`HostTracker`.

    The slow part of locating the host is IDENTITY recognition: per frame,
    :func:`identity.locate_people` runs a face embed + a pose pass + an OSNet
    attire embed for every person (several HTTP round-trips that grow with the
    crowd). Pose estimation alone is real-time. So this samples in two tiers:

    * **fast (every tick)** — one snapshot + pose only; the host is tracked by
      spatial continuity (the person box nearest the last locked box, within a
      jump gate). Runs at pose-estimation rate.
    * **confirm (every ``HRI_FOLLOW_CONFIRM_EVERY_N`` ticks, or whenever
      continuity is lost)** — full ``locate_people`` to re-lock onto the real
      host and correct drift / a wrong-person lock.

    :meth:`sample` returns ``(world_xy | None, side | None)`` exactly like the
    loop expects. *side* is the box's horizontal frame offset (``cx/width -
    0.5``, negative = left); it is ``None`` only when no person box was found,
    while *world_xy* may be ``None`` if the depth lift failed. Lives here, not in
    skills.py, because it calls ``locate_people`` (skills importing identity
    would be a cycle).
    """

    def __init__(self) -> None:
        self.confirm_every_n = int(os.getenv("HRI_FOLLOW_CONFIRM_EVERY_N", "8"))
        self.gate_frac = float(os.getenv("HRI_FOLLOW_TRACK_GATE_FRAC", "0.2"))
        self.debug = os.getenv("HRI_FOLLOW_TRACK_DEBUG", "0").lower() in ("1", "true", "yes")
        self._locked: tuple | None = None      # last host box (xyxy)
        self._since_confirm = 1 << 30          # force a confirm on the first sample

    def sample(self, ctx: TaskContext):
        t0 = time.monotonic()
        snap = ctx.snapshot()
        t_snap = time.monotonic()
        if snap is None:
            return None, None
        img = snap.img
        boxes = person_bboxes(ctx, img)        # fast: pose only
        t_pose = time.monotonic()

        # Re-identify periodically or whenever we have no lock; otherwise just
        # follow the locked box by spatial continuity (cheap).
        confirm = self._locked is None or self._since_confirm >= self.confirm_every_n
        mode = "confirm" if confirm else "track"
        if confirm:
            box = self._identify(ctx, img, boxes)
            self._since_confirm = 0
        else:
            box = self._track_box(boxes, img.width)
            self._since_confirm += 1
            if box is None:  # lost continuity early — re-identify now
                box = self._identify(ctx, img, boxes)
                self._since_confirm = 0
                mode = "track->confirm"
        t_id = time.monotonic()

        if box is None:
            self._locked = None
            if self.debug:
                print(f"[HostSampler] {mode} no-host "
                      f"snap={1e3 * (t_snap - t0):.0f}ms pose={1e3 * (t_pose - t_snap):.0f}ms "
                      f"id={1e3 * (t_id - t_pose):.0f}ms")
            return None, None
        self._locked = box
        side = (box[0] + box[2]) / 2 / img.width - 0.5
        # Lift every tick (the follow loop drives to this point each cycle); skip
        # the full-frame edge filter to keep the lift cheap enough to run flat out.
        xy = lift_bbox_world_xy(ctx, snap, box, use_edge_filter=False)
        if self.debug:
            print(f"[HostSampler] {mode} snap={1e3 * (t_snap - t0):.0f}ms "
                  f"pose={1e3 * (t_pose - t_snap):.0f}ms id={1e3 * (t_id - t_pose):.0f}ms "
                  f"lift={1e3 * (time.monotonic() - t_id):.0f}ms")
        return xy, side

    def _identify(self, ctx: TaskContext, img, boxes):
        """The host's box via full recognition; nearest person as the fallback."""
        if ctx.people is not None and ctx.people.count() > 0:
            found = locate_people(ctx, [img], ["host"]).get("host")
            if found is not None:
                return found[1]
        return max(boxes, key=_box_area) if boxes else None

    def _track_box(self, boxes, width):
        """The pose box nearest the locked one, if within the per-tick jump gate."""
        if self._locked is None or not boxes:
            return None
        lcx = (self._locked[0] + self._locked[2]) / 2
        lcy = (self._locked[1] + self._locked[3]) / 2

        def center_dist(b):
            return math.hypot((b[0] + b[2]) / 2 - lcx, (b[1] + b[3]) / 2 - lcy)

        best = min(boxes, key=center_dist)
        return best if center_dist(best) <= self.gate_frac * width else None


class FollowHostAndDropBag(SubTask):
    """Ask the host where the bag goes, follow them there, then place it.

    The host answers by walking off ("follow me"); a :class:`_HostSampler`
    tracks them in two tiers — pose-only every tick (real-time) plus a periodic
    full identity re-lock — lifts the host's box to a map-frame point, and the
    robot drives to within ``HRI_FOLLOW_DISTANCE_M`` so it never bumps them.
    When the host slips out of view, a :class:`MotionPredictor` extrapolates
    their recent map-frame trajectory so the robot drives to where they were
    heading (re-acquiring fast) instead of scanning blindly. It keeps following
    until the host tells it to put the bag down.

    Voice runs in parallel: while the loop drives, a :class:`CommandListener`
    records, transcribes, and LLM-classifies in the background so the mic is
    never dark during a nav step or an STT/LLM round-trip. The room is full of
    people, so every utterance is classified (:func:`skills.classify_host_command`)
    — only a clear host instruction acts; crowd chatter is ignored. Then it runs
    the same arm release as before. Gated by ``HRI_ENABLE_BAG`` + ``has_bag``.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        if not (_bag_enabled() and ctx.data.get("has_bag")):
            return StepResult.DONE

        listen_timeout = float(os.getenv("HRI_FOLLOW_LISTEN_TIMEOUT_SEC", "5"))

        # Ask where the bag goes — one synchronous turn (the robot isn't moving
        # yet, so blocking on the answer is fine). The host may point at a spot
        # right here, or lead off.
        ctx.say(prompts.BAG_ASK_WHERE)
        if classify_host_command(ctx, ctx.listen(timeout=listen_timeout)) == "place":
            return self._place_bag(ctx)
        # Hand the drive-toward / predict / rotate-search loop to follow_target,
        # tracking the host by identity via _HostSampler (two-tier: pose every
        # tick + a periodic full re-lock). The CommandListener is the stopper:
        # follow_target enters it AFTER the warmup ack (so neither thread
        # transcribes the robot's own voice) and ends the moment it hears
        # "place". on_stopped speaks the place ack while the threads wind down.
        reason = follow_target(
            ctx,
            _HostSampler().sample,
            stopper=CommandListener(ctx),
            on_warmup=lambda: ctx.say(prompts.FOLLOW_HOST_ACK),
            on_lost=lambda: ctx.say(prompts.FOLLOW_HOST_LOST),
            on_stopped=lambda: ctx.say(prompts.BAG_PLACE_ACK),
        )
        if reason == "lost":
            print("[HRI] lost the host past the search budget; placing here")
        return self._place_bag(ctx)

    def _place_bag(self, ctx: TaskContext) -> StepResult:
        """Lower the left arm, open the gripper to release the bag, reset."""
        try:
            ctx.walkie.robot.arm.right.go_to_home(pose_name="standby", blocking=False)  # get the arm out of the way for better nav while following
            ctx.walkie.robot.arm.left.go_to_pose([0.38, 0.16, 0.5299], [-2.6230, -0.0326, -1.4681], blocking=True)
            ctx.walkie.robot.arm.left.gripper(1.0)  # open: release the bag
            ctx.walkie.robot.arm.go_to_home(group_name="both_arms_lift", blocking=True)  # reset the arm for better nav after releasing the bag
            ctx.walkie.robot.arm.left.gripper(0, blocking=False)  # close the gripper after releasing the bag, so it's not just hanging open while following
        except Exception as exc:
            print(f"[HRI] bag release failed ({exc})")
        ctx.data["has_bag"] = False
        return StepResult.DONE


class FollowNearestPerson(SubTask):
    """Test task: follow whoever is closest — pose only, NO identity/appearance.

    Drives :func:`skills.follow_target` with the identity-free
    :func:`skills.nearest_person_sample`, so the whole follow loop (track /
    predict / rotate-search, sampling decoupled from the nav cadence) can be
    exercised on the robot without enrolling anyone or running the bag handover.
    No stopper, so it follows until the person is lost past the search budget or
    ``HRI_FOLLOW_TIMEOUT_SEC`` elapses. Ungated — runnable on its own.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        reason = follow_target(
            ctx,
            nearest_person_sample,
            on_warmup=lambda: ctx.say("Okay, please lead the way and walk slowly, and I will follow you."),
            on_lost=lambda: ctx.say(prompts.FOLLOW_HOST_LOST),
        )
        print(f"[HRI] follow-nearest-person finished ({reason})")
        ctx.say("Okay, I will stop following you now.")
        return StepResult.DONE


class TestTask(SubTask):
    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say("This is a test task. It does nothing.")
        while True:
            ctx.snapshot()  # keep the camera warms
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
            # GoToDoor(1),
            # GreetAndLearn(1),
            # GuideToLivingRoom(1),
            # OfferSeat(1),
            # GoToDoor(2),
            # GreetAndLearn(2),
            # ReceiveBag(),
            # GuideToLivingRoom(2),
            # OfferSeat(2),
            # IntroduceEveryone(),
            # FollowHostAndDropBag(),
            FollowNearestPerson(),  # follow-loop test: no identity, no bag
            # TestTask(),
        ],
        ctx,
    )
