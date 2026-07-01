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
    audit_identity_collisions,
    enroll_guest_frames,
    enroll_person_in_box,
    locate_people,
    make_follow_selector,
    wait_until_seated,
)
from .skills import (
    classify_host_command,
    describe_seated_person,
    host_command_listener,
    llm_guest_intro_speeches,
    llm_pick_seat,
)
from tasks.skills import (
    FaceTracker,
    cxcywh_to_xyxy,
    face_point,
    find_seated_person_bbox,
    follow_person,
    heading_to_point,
    lift_bbox_world_xy,
    match_people_to_seats,
    move_base_relative,
    person_seat_anchor,
    recall_person_xy,
    remember_located_positions,
    remember_person_xy,
    reset_people_positions,
    scan_seats,
    select_largest_person,
    side_relative_to_listener,
    sweep_snapshots,
    tilt_head,
    wait_for_person,
)
from walkie_world.map.locations import resolve_pose


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
        # Wait start task button
        # ctx.say("I'm waiting for the start button to be pressed.")
        # print("[HRI] waiting for start button to be pressed...")
        # while not ctx.walkie.robot.button.is_pressed:
        #     print(ctx.walkie.robot.button.is_pressed)
        #     pass
        # ctx.say("Start button pressed. Heading to the door.")
        print("[HRI] start button pressed")
        ctx.walkie.robot.arm.go_to_home(group_name="both_arms_lift", blocking=False)  # reset the arm for better nav
        # Greeting waypoint = the mapped `entrance` door (world.toml). Its heading is
        # the passage direction (faces into the arena); GreetAndLearn re-centers on the
        # guest via face-tracking, so the initial heading is not critical. Falls back to
        # an optional HRI_DOOR_POSE env override, then origin, on a box with no map.
        door = ctx.world.doors.get("entrance") if ctx.world else None
        if door is not None:
            x, y, heading = door.pose
        x, y, heading = resolve_pose(None, env_fallback="HRI_DOOR_POSE", default=f"{x},{y},{heading}")
        print(x, y, heading)
        ctx.goto(x, y, heading)
        # Wait for the guest to come stand in front before greeting. Look
        # straight ahead so a standing person's face is in frame (nav may have
        # left the head tilted down), then poll for a face bigger than the area
        # floor; restore the tilt afterwards for the greeting/next nav.
        return StepResult.DONE


class GreetAndLearn(SubTask):
    """Greet at the door, learn name + drink, then take a posed photo.

    After the Q&A the robot steps back ~30 cm, says "say cheese", and captures a
    burst of face frames (head level) plus body/attire frames (head tilted down)
    that are averaged into one robust enrollment for BOTH guests. Only guest 1's
    appearance is also captioned in text (told to guest 2 at the introduction)."""

    def __init__(self, guest: int):
        super().__init__(f"GreetAndLearn(guest {guest})")
        self.guest = guest

    def run(self, ctx: TaskContext) -> StepResult:
        record = _guest(ctx, self.guest)
        time.sleep(5) # wait for navigation to settle
        ctx.say(prompts.LOOKING_FOR_GUEST)

        tilt_head(ctx, -0.15)
        if wait_for_person(ctx):
            print("[HRI] guest detected at the door")
        else:
            print("[HRI] no guest detected before timeout; greeting anyway")
        # Keep the base turned toward the guest's face for the whole exchange:
        # a background loop tracks the biggest face in view and rotates the base
        # to re-center it while we ask questions. The base must be still and free
        # for the posed photo afterward, so the capture runs AFTER this block —
        # moving the base inside it would fight the tracker's nav goals.
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

            name = record["name"] or "there"
            ctx.say(f"Nice to meet you, {name}!")
        # FaceTracker has now stopped: the base is free to move for the photo.
        ctx.score("gaze_greeting")  # tracked + faced the guest throughout the Q&A
        self._posed_capture(ctx, record)
        return StepResult.DONE  # partial info still scores — never block here

    def _posed_capture(self, ctx: TaskContext, record: dict) -> None:
        """Step back, warn the guest, capture face (level) + body (tilted-down)
        frames, caption guest-1's appearance, and enroll the averaged embeddings.

        Runs AFTER FaceTracker exits (it moves the base, which would fight the
        tracker's concurrent nav goals). Stepping back ~30 cm frames the whole
        body so the OSNet attire embedding is strong; the face is shot head-level
        and the body with the head tilted down. Everything is best-effort: a
        failure leaves whatever was already captured/enrolled and never blocks.
        """
        count = int(os.getenv("HRI_GREET_CAPTURE_COUNT", "3"))
        backup_m = float(os.getenv("HRI_GREET_BACKUP_M", "0.30"))
        face_tilt = float(os.getenv("HRI_GREET_FACE_TILT", "0.0"))
        app_tilt = float(os.getenv("HRI_GREET_APPEARANCE_TILT", "0.15"))
        tilt_settle = float(os.getenv("HRI_GREET_TILT_SETTLE_SEC", "0.8"))
        cap_gap = float(os.getenv("HRI_GREET_CAPTURE_GAP_SEC", "0.3"))

        def _burst() -> list:
            imgs = []
            for _ in range(count):
                img = ctx.capture()
                if img is not None:
                    imgs.append(img)
                if cap_gap > 0:
                    time.sleep(cap_gap)
            return imgs

        ctx.say(prompts.PHOTO_SAY_CHEESE)
        # Step back first so the whole body fits, then cue the guest right before
        # the face burst (so the "cheese" smile is fresh when the shutter fires).
        move_base_relative(ctx, -backup_m)  # blocking goto; no extra settle needed

        # FACE frames: head level (needs the face for the embedding + the caption).
        tilt_head(ctx, face_tilt, settle=tilt_settle)
        face_imgs = _burst()
        # ATTIRE frames: head tilted down to frame the body for a strong OSNet crop.
        tilt_head(ctx, app_tilt, settle=tilt_settle)
        app_imgs = _burst()
        # Restore the level head for the next nav step.
        tilt_head(ctx, 0.0)

        # Guest-1 only: TEXT appearance description told to guest 2 later. Use a
        # head-LEVEL face frame (it needs hair/glasses/face), not a tilted-down one.
        if self.guest == 1 and face_imgs:
            try:
                record["appearance"] = ctx.walkieAI.image.caption(
                    face_imgs[0], prompt=prompts.APPEARANCE_CAPTION_PROMPT
                )
            except Exception as exc:
                print(f"[HRI] appearance caption failed ({exc})")

        # Remember this guest's face + attire under a stable id (averaged over the
        # burst), so the introduction step can find them again after a seat switch.
        if face_imgs or app_imgs:
            enroll_guest_frames(
                ctx, face_imgs, app_imgs, f"guest-{self.guest}",
                name=record["name"] or "", drink=record["drink"] or "",
            )
        else:
            print("[HRI] posed capture produced no frames; skipping enrollment")

        ctx.say(prompts.YAY_I_REMEMBER)


class GuideToLivingRoom(SubTask):
    def __init__(self, guest: int):
        super().__init__(f"GuideToLivingRoom(guest {guest})")

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.FOLLOW_ME)
        x, y, heading = resolve_pose("living_room", default="0.0,0.0,0")
        if not ctx.goto(x, y, heading):
            return StepResult.RETRY
        ctx.score("gaze_navigation")  # guided the guest, facing the navigation goal
        return StepResult.DONE


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
        ctx.score("seat_offer")  # a free seat was found + offered to this guest
        # Before finishing, confirm the guest actually sat down: watch the seat
        # (arm still pointing) and treat them staying recognized in the frame
        # for HRI_SEATED_DWELL_SEC as seated. Persist wherever they ended up —
        # their real position (which may differ from the offered seat if they
        # picked another), falling back to the offered seat's point.
        _seated, guest_xy = wait_until_seated(ctx, f"guest-{self.guest}")
        final_xy = guest_xy or world_xy
        if final_xy is not None:
            remember_person_xy(ctx, f"guest-{self.guest}", final_xy)
        x, y, heading = resolve_pose("living_room", default="0.0,0.0,0")
        ctx.goto(x, y, heading)
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
        ctx.score("bag_handover")  # arm: received the bag via handover
        ctx.say(prompts.BAG_CLOSING_WARNING)
        time.sleep(1)  # let the nav settle after the arm movement and possible wait
        ctx.walkie.robot.arm.left.gripper(0.0)  # close
        ctx.walkie.robot.arm.go_to_pose(0.1413, 0.0481, 0.9640, -1.2972, -0.3040, -3.0219, group_name="left_arm", blocking=True)
        ctx.say(prompts.BAG_RECEIVED)
        return StepResult.DONE


class AuditIdentities(SubTask):
    """Cross-guest near-duplicate audit, run once after everyone is enrolled.

    Compares the stored face and attire embeddings of every enrolled pair
    (guest-1, guest-2, host) and logs a WARNING for any pair that is suspiciously
    similar (likely the same person, or two guests hard to tell apart). Two
    DISTINCT guests can't be safely merged, so this never deletes a record; with
    ``HRI_DUP_ACTION=widen`` it raises the attire-match margin for the rest of the
    run on an ATTIRE collision, pushing recognition to lean on the more
    discriminative FACE. Cheap (pure vector math, no camera/LLM); best-effort.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        collisions = audit_identity_collisions(ctx)
        if not collisions:
            return StepResult.DONE
        action = os.getenv("HRI_DUP_ACTION", "log").lower()
        if action == "widen" and any(kind == "appearance" for *_ids, kind, _s in collisions):
            cur = float(os.getenv("HRI_FOLLOW_APPEARANCE_MARGIN", "0.05"))
            widened = float(os.getenv("HRI_DUP_WIDENED_APP_MARGIN", "0.10"))
            if widened > cur:
                os.environ["HRI_FOLLOW_APPEARANCE_MARGIN"] = str(widened)
                print(f"[HRI] attire collision: widened HRI_FOLLOW_APPEARANCE_MARGIN {cur}->{widened}")
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
            faced = False
            if listener in world:
                heading = heading_to_point(ctx, *world[listener])
                if heading is not None:
                    faced = ctx.rotate_to(heading)
            ctx.say(speeches[listener])
            # Claimed scoring (attempted): we voiced the subject's name/drink and,
            # when the listener could be anchored, turned to face the right guest.
            ctx.score("intro_name_drink",
                      (1 if act["subject_name"] else 0) + (1 if act["subject_drink"] else 0))
            if faced:
                ctx.score("intro_gaze_correct")
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

    Voice runs in parallel: while the loop drives, a :func:`skills.host_command_listener`
    records, transcribes, and LLM-classifies in the background so the mic is
    never dark during a nav step or an STT/LLM round-trip. The room is full of
    people, so every utterance is classified (:func:`skills.classify_host_command`)
    — only a clear host instruction acts; crowd chatter is ignored. Then it runs
    the same arm release as before. Gated by ``HRI_ENABLE_BAG`` + ``has_bag``.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        if not (_bag_enabled() and ctx.data.get("has_bag")):
            return StepResult.DONE
        ctx.walkie.robot.head.set_auto_tilt(False)  # keep the head steady during follow, so the face/attire match is stable; we'll set a fixed tilt for the handover instead
        tilt_head(ctx, -0.15)
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
        # Follow the host (selected by face first, attire fallback). The command
        # listener is the stopper: follow_person enters it AFTER the warmup ack
        # (and the speaker pauses the mic while it talks, so neither thread
        # transcribes the robot's own voice) and ends the moment it hears "place".
        # on_stopped speaks the place ack while the threads wind down.
        reason = follow_person(
            ctx,
            make_follow_selector("host"),
            stopper=host_command_listener(ctx),
            on_warmup=lambda: ctx.say(prompts.FOLLOW_HOST_ACK),
            on_lost=lambda: ctx.say(prompts.FOLLOW_HOST_LOST),
            on_stopped=lambda: ctx.say(prompts.BAG_PLACE_ACK),
        )
        ctx.walkie.robot.head.set_auto_tilt(True)
        if reason == "lost":
            print("[HRI] lost the host past the search budget; placing here")
        else:
            ctx.score("follow_host")  # followed the host to the bag-drop area
        return self._place_bag(ctx)

    def _place_bag(self, ctx: TaskContext) -> StepResult:
        """Lower the left arm, open the gripper to release the bag, reset."""
        try:
            result = ctx.walkie.robot.arm.right.go_to_home(pose_name="standby", blocking=False)  # get the arm out of the way for better nav while following
            print(result)
            result = ctx.walkie.robot.arm.go_to_pose(0.45, 0.16, 1.15, -0.8, 0, -1.57, group_name="left_arm", blocking=True)
            print(result)
            result = ctx.walkie.robot.arm.left.go_to_pose([0.38, 0.16, 0.5299], [-2.6230, -0.0326, -1.4681], blocking=True)
            print(result)
            result = ctx.walkie.robot.arm.left.gripper(1.0)  # open: release the bag
            print(result)
            result = ctx.walkie.robot.arm.go_to_home(group_name="both_arms_lift", blocking=True)  # reset the arm for better nav after releasing the bag
            print(result)
            result = ctx.walkie.robot.arm.left.gripper(0, blocking=False)  # close the gripper after releasing the bag, so it's not just hanging open while following
            print(result)
        except Exception as exc:
            print(f"[HRI] bag release failed ({exc})")
        ctx.data["has_bag"] = False
        ctx.score("drop_correct_area")  # arm: released the bag at the drop area
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


class TestRememberAndFollowHost(SubTask):
    """Test harness: remember the host in front, then follow them + drop the bag.

    A focused exercise of the follow-host path (:class:`FollowHostAndDropBag`)
    WITHOUT the full receptionist run — for debugging when the robot "won't
    follow anyone". Two phases:

    1. *Remember the host.* The host stands in front; the robot takes a posed
       burst — face frames (head level) then attire frames (head tilted down to
       frame the body) — and enrolls the burst-averaged face + OSNet attire
       embeddings under the id ``"host"`` via :func:`enroll_guest_frames` (the
       SAME id and path the real OfferSeat host enrollment uses), tagged with
       the host's name/drink from ``HRI_HOST_NAME`` / ``HRI_HOST_DRINK``. Those
       two modalities are exactly what :func:`identity.select_person_to_follow`
       scores each tick — FACE first, ATTIRE fallback — so a weak enrollment
       here is the usual reason the follow loop never locks on.
    2. *Follow + drop.* "Assume we have a bag": mark ``has_bag`` and force
       ``HRI_ENABLE_BAG`` on, then delegate to :class:`FollowHostAndDropBag`, so
       the ask-where -> follow -> place-bag sequence runs completely unchanged.

    Run it on its own with ``HRI_TEST_FOLLOW_HOST=1`` (see :func:`build_hri_task`).
    ``HRI_FOLLOW_TRACK_DEBUG=1`` / ``HRI_FOLLOW_VIZ=1`` (both already on in
    config.toml) log the per-tick stage costs and write follow_viz.jpg showing
    which body the loop is locked onto.
    """

    def __init__(self) -> None:
        super().__init__("TestRememberAndFollowHost")

    def run(self, ctx: TaskContext) -> StepResult:
        self._remember_host(ctx)
        # Assume the bag is already in hand: skip ReceiveBag's handover but mark
        # the bag held and force the gate on, so FollowHostAndDropBag runs end
        # to end instead of short-circuiting on the _bag_enabled()/has_bag check.
        os.environ["HRI_ENABLE_BAG"] = "1"
        ctx.data["has_bag"] = True
        return FollowHostAndDropBag().run(ctx)

    def _remember_host(self, ctx: TaskContext) -> None:
        """Posed burst of the standing host -> enroll face + attire under "host".

        Mirrors GreetAndLearn's posed capture (same ``HRI_GREET_*`` knobs): step
        back so the whole body fits, shoot face frames head-level, then attire
        frames head-down, and enroll the burst-averaged embeddings. Best-effort —
        a capture/enroll failure logs and leaves whatever was already captured.
        """
        count = int(os.getenv("HRI_GREET_CAPTURE_COUNT", "3"))
        backup_m = float(os.getenv("HRI_GREET_BACKUP_M", "0.30"))
        face_tilt = float(os.getenv("HRI_GREET_FACE_TILT", "0.0"))
        app_tilt = float(os.getenv("HRI_GREET_APPEARANCE_TILT", "0.15"))
        tilt_settle = float(os.getenv("HRI_GREET_TILT_SETTLE_SEC", "0.8"))
        cap_gap = float(os.getenv("HRI_GREET_CAPTURE_GAP_SEC", "0.3"))

        def _burst() -> list:
            imgs = []
            for _ in range(count):
                img = ctx.capture()
                if img is not None:
                    imgs.append(img)
                if cap_gap > 0:
                    time.sleep(cap_gap)
            return imgs

        ctx.say("Hello! I am going to remember you as the host. "
                "Please stand in front of me and look at me.")
        tilt_head(ctx, face_tilt, settle=tilt_settle)
        if not wait_for_person(ctx):
            print("[HRI] no host face detected before timeout; capturing anyway")
        move_base_relative(ctx, -backup_m)  # frame the whole body for the attire crop
        # FACE frames: head level (needs the face for the embedding + caption).
        ctx.say(prompts.PHOTO_SAY_CHEESE)
        face_imgs = _burst()
        # ATTIRE frames: head tilted down to frame the body for a strong OSNet crop.
        tilt_head(ctx, app_tilt, settle=tilt_settle)
        app_imgs = _burst()
        tilt_head(ctx, 0.0)  # restore the level head for the follow

        # Text appearance (best-effort), kept on the blackboard + as the record's
        # attributes, mirroring the real host capture in OfferSeat.
        appearance = None
        if face_imgs:
            try:
                appearance = ctx.walkieAI.image.caption(
                    face_imgs[0], prompt=prompts.APPEARANCE_CAPTION_PROMPT
                )
            except Exception as exc:
                print(f"[HRI] host appearance caption failed ({exc})")
        ctx.data.setdefault("host", {})["appearance"] = appearance

        if face_imgs or app_imgs:
            enroll_guest_frames(
                ctx, face_imgs, app_imgs, "host",
                name=os.getenv("HRI_HOST_NAME", "").strip(),
                drink=os.getenv("HRI_HOST_DRINK", "").strip(),
                attributes=appearance or "",
            )
        else:
            print("[HRI] host posed capture produced no frames; nothing enrolled")
        ctx.say("Got it, I will remember you. Now lead the way and I will follow.")


def _label(draw, x: int, y: int, text: str, color) -> None:
    """Draw *text* with a black backing box at (x, y) so it stays readable."""
    ty = max(0, y - 14)
    draw.rectangle((x, ty, x + 7 * len(text) + 4, ty + 13), fill=(0, 0, 0))
    draw.text((x + 2, ty + 1), text, fill=color)


_SCAN_TIMING_ORDER = ("snapshot", "detect", "pose", "assemble", "total")


def _fmt_scan_timings(timings: dict) -> str:
    """One-line 'snapshot=..ms detect=..ms ...' from a scan_seats timing dict."""
    return "  ".join(
        f"{k}={timings[k] * 1000:.0f}ms" for k in _SCAN_TIMING_ORDER if k in timings
    )


def _fmt_detect_breakdown(timings: dict) -> str:
    """Per-class detect split 'chair=..ms sofa=..ms ...' (the detect:<class>
    keys scan_seats records in detect_per_class mode), or '' when absent."""
    return "  ".join(
        f"{k.split(':', 1)[1]}={v * 1000:.0f}ms"
        for k, v in timings.items() if k.startswith("detect:")
    )


def _annotate_scan(snap, seats, persons, timings=None):
    """Draw scan_seats() output onto the captured frame and return a PIL image.

    Seats are boxed green (free) / red (occupied), each sofa cushion is boxed
    with its own occupancy and LEFT/MIDDLE/RIGHT label, and every detected person
    is boxed blue with a dot at their seating anchor (the point scan_seats uses to
    decide which seat they hold). A top banner sums the counts and, when
    *timings* is given, the per-stage benchmark on a second line plus the
    per-class detect split (if recorded) on a third.
    """
    from PIL import ImageDraw

    FREE, OCC, PERSON = (0, 200, 0), (220, 30, 30), (40, 120, 255)
    vis = snap.img.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    for i, s in enumerate(seats):
        color = OCC if s.occupied else FREE
        x1, y1, x2, y2 = (int(v) for v in s.bbox_xyxy)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        _label(draw, x1, y1,
               f"#{i} {s.class_name} {s.confidence:.2f} "
               f"{'OCC' if s.occupied else 'FREE'}", color)
        for p in s.parts or []:  # sofa cushions: each shows its own occupancy
            pc = OCC if p.occupied else FREE
            px1, py1, px2, py2 = (int(v) for v in p.bbox_xyxy)
            draw.rectangle((px1, py1, px2, py2), outline=pc, width=1)
            draw.text((px1 + 3, py2 - 14),
                      f"{p.label} {'X' if p.occupied else 'o'}", fill=pc)
    for p in persons:
        x1, y1, x2, y2 = (int(v) for v in cxcywh_to_xyxy(p.bbox))
        draw.rectangle((x1, y1, x2, y2), outline=PERSON, width=2)
        ax, ay = person_seat_anchor(p)
        draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), fill=PERSON)
    free = sum(1 for s in seats if not s.occupied)
    lines = [(f"seats: {len(seats)} ({free} free)   persons: {len(persons)}",
              (255, 255, 255))]
    if timings:
        lines.append((_fmt_scan_timings(timings), (255, 210, 80)))
        breakdown = _fmt_detect_breakdown(timings)
        if breakdown:
            lines.append(("detect: " + breakdown, (255, 210, 80)))
    draw.rectangle((0, 0, vis.width, 6 + 14 * len(lines)), fill=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        draw.text((6, 5 + 14 * i), text, fill=color)
    return vis


class TestScanSeats(SubTask):
    """Test task: scan for seats + people each tick and show what's detected.

    Calls :func:`skills.scan_seats` in a loop and renders the result with
    :func:`_annotate_scan` — seats green (free) / red (occupied), per-cushion
    occupancy on sofas, and every detected person with their seating anchor. The
    annotated frame is shown live in a cv2 window when a display is available and
    ALWAYS written to ``HRI_SCAN_VIZ_PATH`` (default ``seat_scan_viz.jpg``) so it
    works headless too; a one-line summary is printed each tick. Each scan is also
    benchmarked per stage (snapshot / detect / pose / assemble / total) — printed
    each tick and overlaid on the frame — to show which part of scan_seats
    dominates. Press ``q``/Esc in the window to stop. Ungated; run standalone with
    ``HRI_TEST_SCAN_SEATS=1`` (see :func:`build_hri_task`).
    """

    def __init__(self) -> None:
        super().__init__("TestScanSeats")

    def run(self, ctx: TaskContext) -> StepResult:
        interval = float(os.getenv("HRI_SCAN_INTERVAL_SEC", "0.5"))
        viz_path = os.getenv("HRI_SCAN_VIZ_PATH", "seat_scan_viz.jpg")
        window = "Walkie seat scan"
        # cv2 gives a live, self-updating window; missing or headless falls back
        # to writing the annotated JPEG every tick (can_show flips off on the
        # first imshow failure so we don't retry a dead display each frame).
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            print(f"[HRI] scan viz: cv2/numpy unavailable ({exc}); writing file only")
            cv2 = None
        can_show = cv2 is not None
        try:
            while True:
                timings: dict = {}
                # detect_per_class (HRI_SCAN_DETECT_PER_CLASS, default off) splits
                # the detect stage into one timed call per seat class. The detector
                # cost is mostly a fixed per-CALL overhead, so per-class is ~N times
                # slower than the single batched call the production path uses —
                # leave it off to see realistic timing, on only to attribute cost.
                seats, persons, snap = scan_seats(ctx, timings=timings)
                print(f"[HRI] scan timing: {_fmt_scan_timings(timings)}")
                breakdown = _fmt_detect_breakdown(timings)
                if breakdown:
                    print(f"[HRI]   detect breakdown: {breakdown}")
                if snap is None:
                    print("[HRI] scan_seats: capture failed; retrying")
                    time.sleep(interval)
                    continue
                free = sum(1 for s in seats if not s.occupied)
                print(f"[HRI] scan: {len(seats)} seats ({free} free), {len(persons)} persons")
                for i, s in enumerate(seats):
                    cushions = ("  cushions: " + ", ".join(
                        f"{p.label}={'occ' if p.occupied else 'free'}" for p in s.parts
                    )) if s.parts else ""
                    print(f"        #{i} {s.class_name} conf={s.confidence:.2f} "
                          f"{'OCCUPIED' if s.occupied else 'free'}{cushions}")

                vis = _annotate_scan(snap, seats, persons, timings=timings)
                vis.save(viz_path, "JPEG", quality=80)
                if can_show:
                    try:
                        bgr = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
                        cv2.imshow(window, bgr)
                        # waitKey both pumps the window and provides the inter-tick
                        # delay; q/Esc ends the test.
                        if (cv2.waitKey(max(1, int(interval * 1000))) & 0xFF) in (ord("q"), 27):
                            break
                        continue
                    except Exception as exc:
                        print(f"[HRI] scan viz: no display ({exc}); writing {viz_path} only")
                        can_show = False
                # time.sleep(interval)
        finally:
            if cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
        return StepResult.DONE


class TestTask(SubTask):
    def run(self, ctx: TaskContext) -> StepResult:
        # ctx.say("This is a test task. It does nothing.")
        while True:
            snap = ctx.snapshot()  # keep the camera warms
            seat_classes = [
                c.strip()
                for c in os.getenv("HRI_SEAT_CLASSES", "chair,sofa,armchair,stool").split(",")
                if c.strip()
            ]
            dets = ctx.walkieAI.image.detect(
                snap.img, prompts=seat_classes)
            print("[HRI] test task snapshot taken")
        return StepResult.DONE


class TestMoveBase(SubTask):
    def run(self, ctx: TaskContext) -> StepResult:
        # ctx.say("This is a test task that moves the base. Please make sure the area in front of the robot is clear.")
        move_base_relative(ctx, 0.5)
        time.sleep(1)
        move_base_relative(ctx, -0.5)
        return StepResult.DONE



class TestPlaceBag(SubTask):
    def run(self, ctx: TaskContext) -> StepResult:
        result = ctx.walkie.robot.arm.right.go_to_home(pose_name="standby", blocking=False)  # get the arm out of the way for better nav while following
        print(result)
        result = ctx.walkie.robot.arm.go_to_pose(0.45, 0.16, 1.15, -0.8, 0, -1.57, group_name="left_arm", blocking=True)
        print(result)
        result = ctx.walkie.robot.arm.left.go_to_pose([0.38, 0.16, 0.5299], [-2.6230, -0.0326, -1.4681], blocking=True)
        print(result)
        result = ctx.walkie.robot.arm.left.gripper(1.0)  # open: release the bag
        print(result)
        result = ctx.walkie.robot.arm.go_to_home(group_name="both_arms_lift", blocking=True)  # reset the arm for better nav after releasing the bag
        print(result)
        result = ctx.walkie.robot.arm.left.gripper(0, blocking=False)  # close the gripper after releasing the bag, so it's not just hanging open while following
        print(result)


def prepare_run(ctx: TaskContext) -> None:
    """Reset per-run state before building any HRI slice.

    Guests differ every run, so stale face/attire identities must never match
    today's, and map-frame person positions are run-local. Called once by the
    runner before the chosen slice is built (keeps the builders side-effect-free
    and testable). Gated by HRI_PEOPLE_RESET (default on).
    """
    if ctx.people is not None and os.getenv("HRI_PEOPLE_RESET", "1").lower() in ("1", "true", "yes"):
        try:
            ctx.people.clear()
            print("[HRI] people memory cleared for a fresh run")
        except Exception as exc:
            print(f"[HRI] people memory reset failed ({exc})")
    reset_people_positions(ctx)


def build_hri_task(ctx: TaskContext) -> Task:
    """The full Receptionist flow (rulebook 5.1) — the ``full`` slice.

    Two guests arrive in turn: greet + learn each (name, drink, appearance), guide
    to the living room, offer a free seat; receive + drop the host's bag; audit the
    two identities, introduce the guests to each other, then follow the host. Every
    step degrades rather than crashes (partial scoring). Run :func:`prepare_run`
    first (the runner does). Pure: constructs no hardware at build time.
    """
    return Task(
        "HRI",
        [
            GoToDoor(1),
            GreetAndLearn(1),
            GuideToLivingRoom(1),
            OfferSeat(1),
            GoToDoor(2),
            GreetAndLearn(2),
            ReceiveBag(),            # arm step — gated by HRI_ENABLE_BAG
            GuideToLivingRoom(2),
            OfferSeat(2),
            AuditIdentities(),       # flag cross-guest near-duplicate face/attire, lean on face
            IntroduceGuests(),       # introduce the two guests to each other
            FollowHostAndDropBag(),
        ],
        ctx,
    )


# --- Isolated slices for step-by-step on-robot bring-up (selected by HRI_SLICE).
# Order = rough bring-up order: tune seat perception, then greet+learn one guest,
# then the follow-host re-ID path, then the whole flow. Mirrors the Restaurant /
# PickAndPlace runners so no step ever needs commenting out to run a sub-path.
def build_seats_slice(ctx: TaskContext) -> Task:
    """Loop seat + people detection and show occupancy — tune seat detection alone."""
    return Task("HRI:seats", [TestScanSeats()], ctx)


def build_greet_slice(ctx: TaskContext) -> Task:
    """Greet + learn a single guest at the door (name, drink, appearance)."""
    return Task("HRI:greet", [GreetAndLearn(1)], ctx)


def build_follow_host_slice(ctx: TaskContext) -> Task:
    """Remember the host standing in front, then follow + drop the bag."""
    return Task("HRI:follow-host", [TestRememberAndFollowHost()], ctx)
