"""HRI (receptionist) subtasks and the build_hri_task factory.

Flow (rulebook ch. 5.1, doorbell + torque sensing intentionally out of scope):
two guests arrive separately at the door; for each: greet, learn name +
favorite drink (no confirmation questions), guide to the living room, offer a
free seat. Once both are seated the robot introduces the two guests to each
other (host not introduced): for each guest it turns to FACE THE OTHER guest
and tells them who is beside them with a left/right cue from that listener's
point of view ("Bob, the person on your left is Alice..."); one LLM call words
both lines. Bag handover / follow-host are gated stubs.

Blackboard layout (ctx.data):
    guests: {1: {"name", "drink", "appearance"}, 2: {...}}
    seats:  {guest_number: (SeatCandidate, img_w, world_xy | None)}
            # from the offer-seat scan; world_xy is the seat's map-frame
            # position when the 3D lift succeeded
    host:   {"appearance": str | None}  # captured during OfferSeat(1) — the
            # only seated person then is the host; name/drink come from
            # HRI_HOST_NAME / HRI_HOST_DRINK (known from the briefing)
    people_xy: {"host"|"guest-1"|"guest-2": (x, y)}  # last seen map-frame
            # position of each seated person, refreshed (latest-wins) on every
            # scan so a seat switch updates it; used to face them later (e.g.
            # the host before the bag handover). Reset at the start of each run.

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
from .identity import (
    enroll_guest,
    enroll_person_in_box,
    locate_people,
    select_person_to_follow,
    wait_until_seated,
)
from .skills import (
    CommandListener,
    FaceTracker,
    classify_host_command,
    describe_seated_person,
    face_point,
    find_seated_person_bbox,
    follow_person,
    heading_to_point,
    lift_bbox_world_xy,
    llm_guest_intro_speeches,
    llm_pick_seat,
    match_people_to_seats,
    parse_pose,
    recall_person_xy,
    remember_located_positions,
    remember_person_xy,
    reset_people_positions,
    scan_seats,
    select_largest_person,
    side_relative_to_listener,
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
        ctx.walkie.robot.arm.go_to_home(group_name="both_arms_lift", blocking=False)  # reset the arm for better nav
        x, y, heading = parse_pose(os.getenv("HRI_DOOR_POSE", "0.0,0.0,0"))
        if not ctx.goto(x, y, heading):
            return StepResult.RETRY
        # Wait for the guest to come stand in front before greeting. Look
        # straight ahead so a standing person's face is in frame (nav may have
        # left the head tilted down), then poll for a face bigger than the area
        # floor; restore the tilt afterwards for the greeting/next nav.
        return StepResult.DONE


class GreetAndLearn(SubTask):
    """Greet at the door, learn name + drink, capture guest 1's appearance."""

    def __init__(self, guest: int):
        super().__init__(f"GreetAndLearn(guest {guest})")
        self.guest = guest

    def run(self, ctx: TaskContext) -> StepResult:
        record = _guest(ctx, self.guest)
        time.sleep(5) # wait for navigation to settle
        ctx.say(prompts.LOOKING_FOR_GUEST)

        ctx.walkie.robot.head.tilt(0)
        if wait_for_person(ctx):
            print("[HRI] guest detected at the door")
        else:
            print("[HRI] no guest detected before timeout; greeting anyway")
        # Keep the base turned toward the guest's face for the whole exchange:
        # a background loop tracks the biggest face in view and rotates the base
        # to re-center it while we ask questions and caption below.
        with FaceTracker(ctx):
            answer = ctx.ask(prompts.GREET_ASK_BOTH)
            info = ctx.extract(prompts.GuestInfo, prompts.EXTRACT_GUEST_INFO_INSTRUCTIONS, answer) if answer else None
            if info:
                record["name"], record["drink"] = info.name, info.drink

            # Keep following up on each genuinely missing field, re-asking up to
            # HRI_ASK_RETRIES times until we get it (asking for missing info is
            # not a penalized confirmation question). Each attempt re-asks AND
            # re-extracts, so an answer that was heard but unparseable also retries.
            ask_retries = int(os.getenv("HRI_ASK_RETRIES", "4"))
            for field, question in (("name", prompts.ASK_MISSING_NAME), ("drink", prompts.ASK_MISSING_DRINK)):
                for _ in range(ask_retries):
                    if record[field]:
                        break
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
        # Persist where each already-seated person (the host, and any earlier
        # guest) is right now, so later steps can face them without re-scanning.
        # Latest-wins, so a re-scan after someone switched seats refreshes it.
        if snap is not None:
            remember_located_positions(ctx, located, [snap])
        seat_occupants, seatless = match_people_to_seats(located, seats)
        person_names = {
            "host": os.getenv("HRI_HOST_NAME", "").strip() or None,
            "guest-1": _guest(ctx, 1)["name"],
            "guest-2": _guest(ctx, 2)["name"],
        }
        seat, part, announcement = llm_pick_seat(
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
        # On a sofa, face and offer the specific free cushion the picker chose
        # (its own bbox), not the whole couch — otherwise a guest sent to a
        # 3-seater aims at the middle, possibly onto whoever's already there.
        target_bbox = part.bbox_xyxy if part is not None else seat.bbox_xyxy
        # Lift the target to a map-frame point against the SCAN-TIME geometry
        # frozen in the snapshot — exact despite the slow llm_pick_seat call
        # above; facing then uses odometry + atan2.
        world_xy = lift_bbox_world_xy(ctx, snap, target_bbox)
        ctx.data.setdefault("seats", {})[self.guest] = (seat, img_w, world_xy)
        ctx.walkie.robot.arm.go_to_pose_relative(0.3, 0, 0.2, 0, -1.57, 0, group_name="right_arm", blocking=False)
        faced = face_point(ctx, *world_xy) if world_xy else False
        if announcement:  # LLM-worded offer (may refer to the host)
            ctx.say(announcement)
        elif faced:
            # The rotation landed, so the seat is now centered ahead. Name the
            # cushion side on a sofa so the guest knows which spot to take.
            seat_class = (f"{part.label.lower()} side of the {seat.class_name}"
                          if part is not None else seat.class_name)
            ctx.say(prompts.OFFER_SEAT_TEMPLATE.format(
                seat_class=seat_class, direction=prompts.OFFER_SEAT_FACING))
        else:
            # No map-frame point to face (3D lift failed) — name the seat
            # without a stale left/right phrase and let the guest find it.
            ctx.say(prompts.OFFER_SEAT_FALLBACK)
        # Before finishing, confirm the guest actually sat down: watch the seat
        # (arm still pointing) and treat them staying recognized in the frame
        # for HRI_SEATED_DWELL_SEC as seated. Persist wherever they ended up —
        # their real position (which may differ from the offered seat if they
        # picked another), falling back to the offered seat's point.
        _seated, guest_xy = wait_until_seated(ctx, f"guest-{self.guest}")
        final_xy = guest_xy or world_xy
        if final_xy is not None:
            remember_person_xy(ctx, f"guest-{self.guest}", final_xy)
        ctx.walkie.robot.arm.right.go_to_home(blocking=False)  # reset the arm for better nav after facing
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
        ctx.walkie.robot.arm.left.gripper(0.0)  # close
        ctx.walkie.robot.arm.go_to_pose(0.1413, 0.0481, 0.9640, -1.2972, -0.3040, -3.0219, group_name="left_arm", blocking=True)
        ctx.say(prompts.BAG_RECEIVED)
        return StepResult.DONE


class IntroduceGuests(SubTask):
    """Introduce the two guests to each other, once both are seated.

    For each guest the robot turns to FACE THE OTHER guest (the listener) and
    tells them who is sitting beside them, with a left/right cue from the
    listener's own point of view, e.g. "Bob, the person on your left is Alice,
    and her favorite drink is cola." Both lines are worded by ONE LLM call
    (llm_guest_intro_speeches); the host is not introduced.

    Each guest is anchored to where they actually ARE (they may have switched
    seats), then the side is a 2D cross product of the listener's facing
    (assumed toward the robot) with the vector to the other guest.
    """

    GUESTS = (1, 2)

    def run(self, ctx: TaskContext) -> StepResult:
        guests = {n: _guest(ctx, n) for n in self.GUESTS}

        # Anchor each guest to where they actually ARE. Guests may have switched
        # seats (the rulebook allows it), and a single forward frame can miss
        # one off to the side, so sweep three snapshots — left 10°, center,
        # right 10° — and recognize the enrolled guests across all of them.
        # Faces are matched FIRST over the whole sweep; only a guest still
        # missing falls back to an attire-only pass (covers one turned away in
        # every view). Each match is lifted against the geometry of the very
        # snapshot it was found in, so the slow recognition round-trips and the
        # robot's own sweep rotations can't skew the map-frame positions. A
        # guest still not found falls back to their offered seat's stored world
        # point. World-point headings survive the robot's own rotations, so they
        # stay valid across both turns; a guest with no anchor is introduced
        # without rotating and without a left/right cue.
        ids = [f"guest-{n}" for n in self.GUESTS]
        sweep = float(os.getenv("HRI_INTRO_SWEEP_DEG", "10"))
        snaps = sweep_snapshots(ctx, (sweep, 0.0, -sweep))
        located = locate_people(ctx, [s.img for s in snaps], ids) if snaps else {}
        # Refresh the persisted positions from this sweep (latest-wins), so a
        # guest who switched seats since OfferSeat is now recorded where they are.
        remember_located_positions(ctx, located, snaps)
        seats: dict = ctx.data.get("seats", {})
        world: dict[int, tuple[float, float]] = {}
        for n in self.GUESTS:
            world_xy = None
            found = located.get(f"guest-{n}")
            if found is not None:
                fi, box = found
                world_xy = lift_bbox_world_xy(ctx, snaps[fi], box)
            if world_xy is None:
                stored = seats.get(n)
                if stored is not None:
                    _seat, _img_w, world_xy = stored
            if world_xy is not None:
                world[n] = world_xy

        # Two acts: introduce guest 1 while facing guest 2, then guest 2 while
        # facing guest 1. The side is where the introduced (subject) guest sits
        # from the facing (listener) guest's perspective.
        acts = []
        for subject, listener in ((1, 2), (2, 1)):
            side = None
            if listener in world and subject in world:
                side = side_relative_to_listener(ctx, world[listener], world[subject])
            acts.append({
                "listener": listener,
                "listener_name": guests[listener]["name"],
                "subject_name": guests[subject]["name"],
                "subject_drink": guests[subject]["drink"],
                "side": side,
            })

        speeches = llm_guest_intro_speeches(ctx, acts)
        for act in acts:
            listener = act["listener"]
            if listener in world:
                heading = heading_to_point(ctx, *world[listener])
                if heading is not None:
                    ctx.rotate_to(heading)
            ctx.say(speeches[listener])
        return StepResult.DONE


class FollowHostAndDropBag(SubTask):
    """Ask the host where the bag goes, follow them there, then place it.

    The host answers by walking off ("follow me"); :func:`skills.follow_person`
    drives the base to the host's map-frame point every tick — the host is picked
    out of the crowd by FACE first, falling back to ATTIRE
    (:func:`identity.select_person_to_follow`), and nav's heading-alignment keeps
    the robot facing them, biased toward the nearest match. When the
    host briefly drops from view the loop coasts on its :class:`MotionPredictor`
    estimate, then rotate-searches. It follows until told to put the bag down.

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

        # Turn to face the host before talking to them, using their last known
        # seated position (persisted during the seat offers / introduction).
        host_xy = recall_person_xy(ctx, "host")
        if host_xy is not None:
            face_point(ctx, *host_xy)

        # Ask where the bag goes — one synchronous turn (the robot isn't moving
        # yet, so blocking on the answer is fine). The host may point at a spot
        # right here, or lead off.
        ctx.say(prompts.BAG_ASK_WHERE)
        if classify_host_command(ctx, ctx.listen(timeout=listen_timeout)) == "place":
            return self._place_bag(ctx)
        # Follow the host (selected by face first, attire fallback). The CommandListener is the
        # stopper: follow_person enters it AFTER the warmup ack (so neither thread
        # transcribes the robot's own voice) and ends the moment it hears "place".
        # on_stopped speaks the place ack while the threads wind down.
        reason = follow_person(
            ctx,
            lambda c, snap: select_person_to_follow(c, snap, "host"),
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
            ctx.walkie.robot.arm.go_to_pose(0.45, 0.16, 1.15, -0.8, 0, -1.57, group_name="left_arm", blocking=True)
            ctx.walkie.robot.arm.left.go_to_pose([0.38, 0.16, 0.5299], [-2.6230, -0.0326, -1.4681], blocking=True)
            ctx.walkie.robot.arm.left.gripper(1.0)  # open: release the bag
            ctx.walkie.robot.arm.go_to_home(group_name="both_arms_lift", blocking=True)  # reset the arm for better nav after releasing the bag
            ctx.walkie.robot.arm.left.gripper(0, blocking=False)  # close the gripper after releasing the bag, so it's not just hanging open while following
        except Exception as exc:
            print(f"[HRI] bag release failed ({exc})")
        ctx.data["has_bag"] = False
        ctx.say(prompts.FINISH_TASK)
        return StepResult.DONE


class FollowNearestPerson(SubTask):
    """Test task: follow whoever is closest — pose only, NO identity/appearance.

    Drives :func:`skills.follow_person` with :func:`skills.select_largest_person`,
    so the whole follow loop (snapshot → lift → drive, with motion prediction to
    coast over dropouts) can be exercised on the robot without enrolling anyone
    or running the bag handover. No stopper, so it follows until the person is
    lost past the search budget or ``HRI_FOLLOW_TIMEOUT_SEC`` elapses. Ungated.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        reason = follow_person(
            ctx,
            select_largest_person,
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
    # Positions are map-frame and run-local; never carry them across runs.
    reset_people_positions(ctx)
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
            IntroduceGuests(),  # introduce the two guests to each other
            FollowHostAndDropBag(),
            # FollowNearestPerson(),  # follow-loop test: no identity, no bag
            # TestTask(),
        ],
        ctx,
    )
