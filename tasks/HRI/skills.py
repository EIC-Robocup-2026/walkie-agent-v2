"""Prompt-coupled HRI skills (seat-pick / guest-intro / host-command).

The generic, prompt-free primitives moved to the shared tasks.skills
package; what stays here is the LLM/prompt-driven glue specific to the
Receptionist task.
"""

from __future__ import annotations

import os

from PIL import Image

from client import PersonPose
from tasks.base import TaskContext

from tasks.skills import (
    BBox,
    CommandListener,
    SeatCandidate,
    SeatPart,
    SeatSweep,
    cxcywh_to_xyxy,
    find_seated_person_bbox,
    overlap_fraction,
    pick_free_seat,
    resolve_free_part,
)

from . import prompts
from .identity import distill_appearance_caption


def describe_seated_person(
    ctx: TaskContext,
    img: Image.Image,
    persons: list[PersonPose],
    seats: list[SeatCandidate],
) -> str | None:
    """Caption the appearance of the person sitting on a detected seat.

    Crops to the person overlapping an occupied seat when pose detection found
    one (a lone detected person is accepted too); otherwise captions the whole
    frame and lets the prompt single out the seated person. The raw caption is
    LLM-distilled to person-only details (the caption model narrates the whole
    scene regardless of the prompt). None on failure.
    """
    target = find_seated_person_bbox(persons, seats)
    crop = img
    if target is not None:
        x1, y1, x2, y2 = target
        m = 20  # px padding so clothing isn't clipped at the bbox edge
        crop = img.crop((
            max(0, int(x1 - m)), max(0, int(y1 - m)),
            min(img.width, int(x2 + m)), min(img.height, int(y2 + m)),
        ))
    try:
        raw = ctx.walkieAI.image.caption(
            crop, prompt=prompts.HOST_APPEARANCE_CAPTION_PROMPT
        )
    except Exception as exc:
        print(f"[skills] seated-person appearance caption failed ({exc})")
        return None
    return distill_appearance_caption(ctx, raw)


def classify_host_command(ctx: TaskContext, heard: str) -> str:
    """LLM intent of a heard utterance: ``'follow'``, ``'place'``, or ``'other'``.

    Filters genuine host instructions from the party-crowd chatter the mic
    picks up during the bag handover. An empty/garbled transcript or an
    extraction failure maps to ``'other'`` so the robot never acts on noise.
    """
    if not heard.strip():
        return "other"
    cmd = ctx.extract(
        prompts.HostCommand, prompts.CLASSIFY_HOST_COMMAND_INSTRUCTIONS, heard
    )
    return cmd.intent if cmd is not None else "other"


def host_command_listener(ctx: TaskContext) -> CommandListener:
    """A :class:`~tasks.skills.CommandListener` wired for the bag handover.

    While the robot drives after the host it must still hear "put the bag here".
    The generic listener handles the never-go-deaf mic loop; this wires in the
    HRI decision: every utterance is LLM-classified (:func:`classify_host_command`)
    because the room is full of party chatter, and only a clear ``'place'``
    instruction sets the listener's ``triggered`` event (which ends
    :func:`tasks.skills.follow_person`). The record timeout comes from
    ``HRI_FOLLOW_RECORD_TIMEOUT_SEC``. Use as a context manager::

        with host_command_listener(ctx) as listener:
            while ...:
                ... drive toward the host ...
                if listener.triggered.is_set():
                    break
    """
    return CommandListener(
        ctx,
        on_transcript=lambda text: classify_host_command(ctx, text) == "place",
        record_timeout=float(os.getenv("HRI_FOLLOW_RECORD_TIMEOUT_SEC", "30.0")),
    )


def _person_label(pid: str, names: dict[str, str | None] | None = None) -> str:
    """Human-readable label for an enrolled-person id, with name when known."""
    name = (names or {}).get(pid)
    if pid == "host":
        return f"the host ({name})" if name else "the host"
    if pid.startswith("guest-"):
        n = pid.removeprefix("guest-")
        return f"Guest {n} ({name})" if name else f"Guest {n}"
    return name or pid


def describe_seating_scene(
    sweep: SeatSweep,
    guest: int,
    guest_name: str | None = None,
    host_name: str | None = None,
    host_drink: str | None = None,
    host_appearance: str | None = None,
    prior_seats: dict[int, tuple[SeatCandidate, int, tuple[float, float] | None]] | None = None,
    seat_occupants: dict[int, str] | None = None,
    seatless_people: dict[str, tuple[int, BBox]] | None = None,
    person_names: dict[str, str | None] | None = None,
) -> str:
    """Text rendering of a seat sweep for the LLM seat picker.

    Everything the model needs to decide and to word the offer: each seat's
    view + position in that view's frame (pixel x, where x=0 is that view's
    left edge), size, confidence and occupancy, each person's position, per-seat
    person overlap, the host (always present and seated, with drink/appearance
    when known), and where an earlier guest was seated. The sweep's seats are
    already de-duplicated across the overlapping views.

    When *seat_occupants* (seat index -> recognized person id, from
    :func:`match_people_to_seats`) is given, each named seat says WHO is sitting
    in it; *seatless_people* (id -> (frame, box)) flags people recognized in the
    sweep that no detected seat lined up with, so the model neither offers their
    spot nor double-seats them. *person_names* supplies display names for the ids.
    """
    labels = sweep.frame_labels
    multi = len(sweep.snaps) > 1
    img_w = sweep.snaps[0].img.width if sweep.snaps else 0

    def _view(fi: int) -> str:
        return f" in the {labels[fi]}" if multi and 0 <= fi < len(labels) else ""

    guest_label = f"Guest {guest}" + (f", named {guest_name}," if guest_name else "")
    host_line = (
        f"The party host{f', {host_name},' if host_name else ''} is already "
        f"in the room and seated — one of the seated people is the host."
    )
    if host_drink:
        host_line += f" The host's favorite drink is {host_drink}."
    if host_appearance:
        host_line += f" The host's appearance: {host_appearance}"
    if multi:
        frame_line = (
            f"The robot scanned the seating area from {len(labels)} camera "
            f"headings: {', '.join(labels)}. The views overlap, so the same "
            f"person may appear in more than one view; the seat list below is "
            f"already de-duplicated across views. Each frame is {img_w}px wide; "
            f"x positions are within the named view's frame, x=0 that view's "
            f"left edge."
        )
    else:
        frame_line = (
            f"The camera frame is {img_w}px wide; x=0 is the robot's far left, "
            f"x={img_w} its far right."
        )
    lines = [
        frame_line,
        f"{guest_label} has just arrived, is standing next to the robot, and "
        f"needs a seat.",
        host_line,
        "",
        f"Detected seats ({len(sweep.seats)}):",
    ]
    person_boxes_by_frame = [
        [cxcywh_to_xyxy(p.bbox) for p in ppl] for ppl in sweep.persons_by_frame
    ]
    occupants = seat_occupants or {}
    for i, seat in enumerate(sweep.seats):
        fi = sweep.seat_frames[i]
        x1, y1, x2, y2 = seat.bbox_xyxy
        overlap = max(
            (overlap_fraction(pb, seat.bbox_xyxy)
             for pb in person_boxes_by_frame[fi]),
            default=0.0,
        )
        if seat.parts:
            # A sofa: report each cushion so the model can offer a free one even
            # when someone else is on it. The whole-sofa status is "free" as long
            # as any cushion is open.
            free_parts = [p.label for p in seat.parts if not p.occupied]
            who = f" — {_person_label(occupants[i], person_names)} is on it" if i in occupants else ""
            status = ("OCCUPIED (all cushions taken)" if seat.occupied
                      else f"has free cushion(s): {', '.join(free_parts)}") + who
            cushions = "; ".join(
                f"{p.label} cushion (center x={p.center_px[0]:.0f}px) "
                f"{'taken' if p.occupied else 'FREE'}"
                for p in seat.parts
            )
            status += f" [{cushions}]"
        elif i in occupants:
            status = f"OCCUPIED — {_person_label(occupants[i], person_names)} is sitting here"
        else:
            status = "OCCUPIED" if seat.occupied else "free"
            if overlap > 0:
                status += f" (a person's box covers {overlap:.0%} of it)"
        view = f"{labels[fi]}, " if multi else ""
        lines.append(
            f"  [{i}] {seat.class_name} — {view}center "
            f"x={seat.center_px[0]:.0f}px, {x2 - x1:.0f}x{y2 - y1:.0f}px, "
            f"detection confidence {seat.confidence:.2f}, {status}"
        )
    lines.append("")
    n_persons = sum(len(ppl) for ppl in sweep.persons_by_frame)
    if n_persons:
        note = (" — overlapping views may show the same person more than once"
                if multi else "")
        lines.append(f"Detected people ({n_persons}{note}):")
        for fi, ppl in enumerate(sweep.persons_by_frame):
            for p in ppl:
                cx, _cy, w, h = p.bbox
                lines.append(
                    f"  - person at x={cx:.0f}px{_view(fi)}, {w:.0f}x{h:.0f}px"
                )
    else:
        lines.append("No people detected.")
    for pid, (fi, box) in (seatless_people or {}).items():
        cx = (box[0] + box[2]) / 2
        lines.append(
            f"{_person_label(pid, person_names)} is seated around "
            f"x={cx:.0f}px{_view(fi)}, but no detected seat lines up with them "
            f"— their seat is likely one the detector missed (a couch, stool, "
            f"or surface). They are there: don't offer that spot, and don't "
            f"seat the new guest on top of them."
        )
    for n, (seat, _w, _xy) in (prior_seats or {}).items():
        if n == guest:
            continue
        lines.append(
            f"Guest {n} was earlier offered a {seat.class_name} (from an "
            f"earlier scan, so it may have shifted) and is probably sitting "
            f"there now."
        )
    return "\n".join(lines)


def llm_pick_seat(
    ctx: TaskContext,
    sweep: SeatSweep,
    guest: int,
    guest_name: str | None = None,
    host_name: str | None = None,
    host_drink: str | None = None,
    host_appearance: str | None = None,
    prior_seats: dict[int, tuple[SeatCandidate, int, tuple[float, float] | None]] | None = None,
    seat_occupants: dict[int, str] | None = None,
    seatless_people: dict[str, tuple[int, BBox]] | None = None,
    person_names: dict[str, str | None] | None = None,
) -> tuple[SeatCandidate | None, SeatPart | None, str | None]:
    """Let the LLM choose which seat to offer and word the spoken offer.

    Returns (seat, part, announcement); the returned seat is one of
    ``sweep.seats``, so the caller can recover its view/geometry via
    ``sweep.seats.index(seat)``. *part* is the chosen sofa cushion
    (LEFT/MIDDLE/RIGHT) when the seat is a sofa, else None — the caller faces and
    announces that cushion rather than the whole sofa. The model sees the whole
    sweep (seats across all views, each sofa's free cushions, who is recognized
    where, the host, the other guest's seat) so it can suggest a free seat next
    to the host and refer to the host in the announcement. A null announcement
    means "use the default line". An explicit null seat from the model means
    "nothing suitable"; an extraction failure or out-of-range index degrades to
    the deterministic pick_free_seat (whose first free cushion is then used for
    a sofa).
    """
    seats = sweep.seats
    if not seats:
        return None, None, None
    scene = describe_seating_scene(
        sweep, guest,
        guest_name=guest_name, host_name=host_name,
        host_drink=host_drink, host_appearance=host_appearance,
        prior_seats=prior_seats,
        seat_occupants=seat_occupants,
        seatless_people=seatless_people,
        person_names=person_names,
    )
    choice = ctx.extract(prompts.SeatChoice, prompts.PICK_SEAT_INSTRUCTIONS, scene)
    if choice is None:
        print("[skills] seat choice extraction failed; using heuristic pick")
        seat = pick_free_seat(seats)
        return seat, resolve_free_part(seat), None
    if choice.seat_index is None:
        print(f"[skills] LLM declined to pick a seat ({choice.reason or 'no reason given'})")
        return None, None, None
    if not 0 <= choice.seat_index < len(seats):
        print(f"[skills] LLM seat index {choice.seat_index} out of range; using heuristic pick")
        seat = pick_free_seat(seats)
        return seat, resolve_free_part(seat), None
    seat = seats[choice.seat_index]
    part = resolve_free_part(seat, getattr(choice, "seat_part", None))
    print(f"[skills] LLM picked seat [{choice.seat_index}] {seat.class_name}"
          f"{f' ({part.label} cushion)' if part else ''}"
          f" ({choice.reason or 'no reason given'})")
    return seat, part, (choice.announcement or "").strip() or None


def _guest_intro_fallback(
    listener_name: str | None,
    subject_name: str | None,
    subject_drink: str | None,
    side: str | None,
    subject_appearance: str | None = None,
) -> str:
    """Template introduction spoken to one guest about the other beside them."""
    who = f"the person on your {side}" if side else "the guest next to you"
    name = subject_name or prompts.GENERIC_OTHER_GUEST
    lead = f"{listener_name}, " if listener_name else ""
    line = f"{lead}{who} is {name}."
    if subject_appearance:
        line += f" You can recognize them easily: {subject_appearance}"
        if not line.endswith("."):
            line += "."
    if subject_drink:
        line += f" Their favorite drink is {subject_drink}."
    return line


def llm_guest_intro_speeches(ctx: TaskContext, acts: list[dict]) -> dict[int, str]:
    """Word both guest-to-guest introductions in ONE LLM call: {listener: text}.

    *acts* is one dict per spoken line — the robot FACES that listener and
    presents the other guest beside them::

        {"listener": 1|2, "listener_name": str|None,
         "subject_name": str|None, "subject_drink": str|None,
         "subject_appearance": str|None,
         "side": "left"|"right"|None}

    ``subject_appearance`` is the presented guest's captioned look (clothing,
    hair, glasses, ...) — the LLM is instructed to describe it in detail so the
    listener can actually spot them. Returns a speech keyed by listener number,
    each falling back to a template on extraction failure.
    """
    fallback = {
        a["listener"]: _guest_intro_fallback(
            a["listener_name"], a["subject_name"], a["subject_drink"], a["side"],
            a.get("subject_appearance"),
        )
        for a in acts
    }
    lines = []
    for a in acts:
        lines.append(
            f"While facing guest {a['listener']} (name="
            f"{a['listener_name'] or 'unknown'}), present the OTHER guest "
            f"(name={a['subject_name'] or 'unknown'}; favorite drink="
            f"{a['subject_drink'] or 'unknown'}; appearance="
            f"{a.get('subject_appearance') or 'unknown'}), who is on guest "
            f"{a['listener']}'s {a['side'] or 'unknown'} side."
        )
    speeches = ctx.extract(
        prompts.GuestIntroSpeeches, prompts.GUEST_INTRO_INSTRUCTIONS, "\n".join(lines)
    )
    if speeches is None:
        print("[skills] guest intro speech extraction failed; using template lines")
        return fallback
    by_listener = {
        1: (speeches.facing_guest_1 or "").strip(),
        2: (speeches.facing_guest_2 or "").strip(),
    }
    return {a["listener"]: by_listener.get(a["listener"], "") or fallback[a["listener"]] for a in acts}
