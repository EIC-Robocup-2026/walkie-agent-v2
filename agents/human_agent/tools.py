"""Tools for the Walkie Human (HRI) sub-agent.

Live-camera people understanding for human-robot-interaction tasks
(RoboCup @Home Receptionist first). This first slice ships the capabilities
that need **no new walkie-ai-server route** — they reuse the existing
``image_caption`` and ``pose_estimation`` clients:

- ``describe_person`` — a steered natural-language description of the person in
  view (clothing, hair, glasses, posture) for the spoken introduction.
- ``count_people`` — how many people are visible, with a best-effort
  arm-raised / posture breakdown from pose keypoints.

Face enrollment / recognition (the ``/face-embed`` server route + a face-keyed
people store) land in a later slice — see ``docs/human_recognition_design.md``.

Read-only lookups are ``@parallelable_tool``; ``speak`` moves audio so it's
``@sequential_tool``.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.tools import tool

from agents.core.robot_context import RobotContext
from agents.core.tool_decorators import parallelable_tool, sequential_tool
from interfaces.walkie_interface import WalkieInterface
from perception.gestures import (  # re-exported for the existing pose tests
    arm_raised as _arm_raised,
    describe_gestures as _describe_gestures,
    gesture_phrase as _gesture_phrase,
    kp_map as _kp_map,
    posture as _posture,
    summarize_person as _summarize_person,
)

_DEFAULT_DESCRIBE_PROMPT = (
    "Describe the person in this image for a verbal introduction: their "
    "approximate clothing colors and type, hair, glasses or hat, and whether "
    "they are sitting or standing. Be concise and factual; one or two sentences."
)


def _min_det_score() -> float:
    try:
        return float(os.getenv("FACE_MIN_DET_SCORE", "0.5"))
    except ValueError:
        return 0.5


def _match_threshold() -> float:
    """Max cosine distance for a face to count as a known person."""
    try:
        return float(os.getenv("FACE_MATCH_THRESHOLD", "0.4"))
    except ValueError:
        return 0.4


def _camera_hfov() -> float:
    """Camera horizontal field of view in degrees (for pixel-offset → yaw)."""
    try:
        return float(os.getenv("CAMERA_HFOV_DEG", "70"))
    except ValueError:
        return 70.0


def _seat_classes() -> set[str]:
    raw = os.getenv("SEAT_CLASSES", "chair,couch,sofa,bench")
    return {c.strip().lower() for c in raw.split(",") if c.strip()}


def _seat_occupancy_ratio() -> float:
    """Min (person∩seat / seat) overlap fraction for a seat to count as taken."""
    try:
        return float(os.getenv("SEAT_OCCUPANCY_RATIO", "0.2"))
    except ValueError:
        return 0.2


def _cxcywh_to_xyxy(bbox) -> tuple[int, int, int, int]:
    """Pose person bbox ``(cx, cy, w, h)`` → ``(x1, y1, x2, y2)``."""
    cx, cy, w, h = bbox
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def _overlap_area(a, b) -> int:
    """Pixel area of the intersection of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _aim_phrase(center_x: float, img_w: int) -> str:
    """Direction + approximate turn from a target's image-x to a spoken phrase."""
    if img_w <= 0:
        return "straight ahead"
    yaw = (center_x / img_w - 0.5) * _camera_hfov()  # + = right of center
    if yaw < -8:
        return f"to your left (turn left ~{abs(yaw):.0f}° to face them)"
    if yaw > 8:
        return f"to your right (turn right ~{yaw:.0f}° to face them)"
    return "roughly straight ahead"


def make_human_tools(
    walkie: WalkieInterface,
    walkieAI,
    *,
    agent_name: str = "human",
    people_store=None,
):
    """Build the human sub-agent's tool list.

    ``people_store`` is accepted for signature stability with the later
    face-recognition slice; the tools in this first slice do not use it.
    """

    def _capture():
        return walkie.camera.capture_pil()

    @parallelable_tool
    @tool(parse_docstring=True)
    def describe_person(focus: Optional[str] = None) -> str:
        """Describe a person currently in view (clothing, hair, glasses, posture).

        Use for the Receptionist introduction ("what does the guest look like?")
        or any "describe the person" request. Reports on the **live camera only**.

        Args:
            focus: Optional hint when several people are visible, e.g.
                "the person on the left" or "the one in red", appended to steer
                the description toward them.

        Returns:
            A short factual description of the person.
        """
        prompt = os.getenv("HUMAN_DESCRIBE_PROMPT", _DEFAULT_DESCRIBE_PROMPT)
        if focus:
            prompt = f"{prompt} Focus on {focus}."
        img = _capture()
        caption = walkieAI.image_caption.caption(img, prompt=prompt)
        return f"Person: {caption}"

    @parallelable_tool
    @tool
    def count_people() -> str:
        """Count the people visible now, with a best-effort posture breakdown.

        Reports the total, how many have an arm raised (a wave / hand-raise
        candidate), and an approximate sitting/standing split. Posture is a
        rough heuristic from pose keypoints — treat it as a hint, not ground
        truth. Use for "how many people are here?" / "is anyone waving?".
        """
        img = _capture()
        people = walkieAI.pose_estimation.estimate(img)
        if not people:
            return "No people visible."
        summaries = [_summarize_person(p) for p in people]
        total = len(summaries)
        arms = sum(1 for s in summaries if s["arm_raised"])
        sitting = sum(1 for s in summaries if s["posture"] == "sitting")
        standing = sum(1 for s in summaries if s["posture"] == "standing")
        unknown = total - sitting - standing
        parts = [f"{total} person(s) visible."]
        parts.append(f"Arm raised (waving/hand-raise): {arms}.")
        posture = f"Posture (approx): {standing} standing, {sitting} sitting"
        if unknown:
            posture += f", {unknown} unclear"
        parts.append(posture + ".")
        return " ".join(parts)

    @parallelable_tool
    @tool
    def detect_gestures() -> str:
        """Read each visible person's gesture and posture from the live camera.

        For every person in view reports whether they are waving / raising a
        hand, pointing to your left or right, and whether they are sitting,
        standing or lying down. Use for "is anyone waving?", "who is pointing?",
        or "find the person waving at me" (Restaurant / GPSR). Single-frame
        heuristics from pose keypoints — a hint, not ground truth.
        """
        img = _capture()
        try:
            people = walkieAI.pose_estimation.estimate(img)
        except Exception as e:  # noqa: BLE001 — a server hiccup must not crash the turn
            return f"I couldn't read poses (vision error): {e}"
        if not people:
            return "No people visible."
        w = img.size[0]
        people.sort(key=lambda p: p.bbox[2] * p.bbox[3], reverse=True)  # nearest first
        lines = []
        waving = 0
        for i, p in enumerate(people):
            g = _describe_gestures(p)
            waving += int(g["waving"])
            lines.append(
                f"- person {i+1} ({_aim_phrase(p.bbox[0], w)}): {_gesture_phrase(g)}"
            )
        head = f"{len(people)} person(s) in view"
        if waving:
            head += f", {waving} waving/hand-raised"
        return head + ":\n" + "\n".join(lines)

    @sequential_tool
    @tool(parse_docstring=True)
    def enroll_person(name: str, drink: str) -> str:
        """Remember a guest's face together with their name and favorite drink.

        Use right after greeting a new guest, while they are looking at the
        robot. Stores the face so the guest can be re-identified later (even if
        they move or switch seats). Re-enrolling the same name refreshes it.

        Args:
            name: The guest's name, as they gave it.
            drink: The guest's favorite drink, as spoken (free text).

        Returns:
            Confirmation, or a message if no clear face was visible.
        """
        if people_store is None:
            return "Face memory is off — I can't remember people right now."
        img = _capture()
        try:
            faces = walkieAI.face_recognition.embed(img)
        except Exception as e:  # noqa: BLE001 — a server hiccup must not crash the turn
            return f"I couldn't read a face (face service error): {e}"
        faces = [f for f in faces if f.det_score >= _min_det_score()]
        if not faces:
            return "I don't see a clear face — ask the guest to look at me, then try again."
        face = max(faces, key=lambda f: f.area())  # the person up front
        rec = people_store.enroll(
            name, drink, face.embedding, frame=img, face_bbox_xyxy=face.bbox_xyxy
        )
        again = "" if rec.enrollments <= 1 else f" (refreshed, seen {rec.enrollments}×)"
        return f"Remembered {rec.name}, favorite drink {rec.drink!r}{again}."

    @parallelable_tool
    @tool
    def recognize_person() -> str:
        """Identify the people in view against the remembered guests.

        Returns each visible face as a known guest (name + favorite drink) or as
        unknown. Use to greet a returning guest, or to see who is sitting where
        before an introduction. Reports on the live camera only.
        """
        if people_store is None:
            return "Face memory is off — I can't recognize people right now."
        if people_store.count() == 0:
            return "I haven't remembered anyone yet."
        img = _capture()
        try:
            faces = walkieAI.face_recognition.embed(img)
        except Exception as e:  # noqa: BLE001
            return f"I couldn't read faces (face service error): {e}"
        faces = [f for f in faces if f.det_score >= _min_det_score()]
        if not faces:
            return "No clear face in view."
        threshold = _match_threshold()
        faces.sort(key=lambda f: f.area(), reverse=True)  # nearest first
        lines = []
        for i, f in enumerate(faces):
            rec = people_store.recognize(f.embedding, max_distance=threshold)
            if rec is None:
                lines.append(f"- person {i+1}: unknown (not a remembered guest)")
            else:
                lines.append(
                    f"- person {i+1}: {rec.name}, favorite drink {rec.drink!r} "
                    f"(match {rec.similarity:.2f})"
                )
        return "People in view:\n" + "\n".join(lines)

    @parallelable_tool
    @tool
    def list_known_people() -> str:
        """List every guest remembered so far, with their favorite drink.

        Use to recall both guests' details when introducing them to each other.
        """
        if people_store is None:
            return "Face memory is off."
        people = people_store.list_people()
        if not people:
            return "No guests remembered yet."
        lines = []
        for p in people:
            extra = f"; {p.attributes}" if p.attributes else ""
            lines.append(f"- {p.name}: favorite drink {p.drink!r}{extra}")
        return f"{len(people)} remembered guest(s):\n" + "\n".join(lines)

    @parallelable_tool
    @tool
    def find_empty_seat() -> str:
        """Find seats in view (chairs/sofas) that no one is sitting on.

        Use to "offer a free seat" to a guest. Detects seats and people in the
        live view and reports which seats are unoccupied, with a rough direction
        so the robot can point to one. Reports on the live camera only.
        """
        img = _capture()
        try:
            objects = walkieAI.object_detection.detect(img)
            people = walkieAI.pose_estimation.estimate(img)
        except Exception as e:  # noqa: BLE001
            return f"I couldn't check the seats (vision error): {e}"
        seat_classes = _seat_classes()
        seats = [o for o in objects if (o.class_name or "").lower() in seat_classes]
        if not seats:
            return "I don't see any seats in view."
        person_boxes = [_cxcywh_to_xyxy(p.bbox) for p in people]
        min_ratio = _seat_occupancy_ratio()
        w = img.size[0]
        free = []
        for o in seats:
            box = tuple(int(v) for v in o.bbox)  # xyxy from the detector
            seat_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            taken = any(
                _overlap_area(box, pb) / seat_area >= min_ratio for pb in person_boxes
            )
            if not taken:
                free.append((o.class_name, (box[0] + box[2]) / 2))
        if not free:
            return f"All {len(seats)} visible seat(s) look occupied."
        lines = [f"- {cls} {_aim_phrase(cx, w)}" for cls, cx in free]
        return f"{len(free)} free seat(s) of {len(seats)} visible:\n" + "\n".join(lines)

    @parallelable_tool
    @tool(parse_docstring=True)
    def locate_person(name: Optional[str] = None) -> str:
        """Find where a person is in view so the robot can turn to face them.

        Returns a direction and approximate turn angle to aim at — pass this to
        the actuator to keep gaze on the person. Give ``name`` to find a specific
        remembered guest (matched by face); omit it to face the most prominent
        (nearest) person, e.g. "look at whoever is talking".

        Args:
            name: Optional remembered guest to look for.

        Returns:
            The person's direction + approximate yaw, or a not-found message.
        """
        img = _capture()
        w = img.size[0]
        if name:
            if people_store is None:
                return (
                    "Face memory is off — I can't pick out a named person. "
                    "Omit the name to face the nearest person instead."
                )
            try:
                faces = walkieAI.face_recognition.embed(img)
            except Exception as e:  # noqa: BLE001
                return f"I couldn't read faces (face service error): {e}"
            faces = [f for f in faces if f.det_score >= _min_det_score()]
            threshold = _match_threshold()
            for f in faces:
                rec = people_store.recognize(f.embedding, max_distance=threshold)
                if rec is not None and rec.name.lower() == name.strip().lower():
                    cx = (f.bbox_xyxy[0] + f.bbox_xyxy[2]) / 2
                    return f"{rec.name} is {_aim_phrase(cx, w)}."
            return f"I don't see {name} in view."
        try:
            people = walkieAI.pose_estimation.estimate(img)
        except Exception as e:  # noqa: BLE001
            return f"I couldn't detect people (vision error): {e}"
        if not people:
            return "No one is in view."
        nearest = max(people, key=lambda p: p.bbox[2] * p.bbox[3])  # largest = nearest
        return f"The nearest person is {_aim_phrase(nearest.bbox[0], w)}."

    @sequential_tool
    @tool(parse_docstring=True)
    def speak(text: str) -> str:
        """Speak text aloud through the robot's speaker.

        Use sparingly: the main Walkie agent usually speaks the final answer.

        Args:
            text: The text to vocalize.

        Returns:
            Confirmation that the text was spoken.
        """
        stream = walkieAI.tts.synthesize_stream(text)
        walkie.speaker.play_stream(stream, blocking=True)
        try:
            RobotContext.get().add_speech(agent_name, text)
        except RuntimeError:
            pass
        return f"Spoke: {text!r}"

    return [
        describe_person,
        count_people,
        detect_gestures,
        enroll_person,
        recognize_person,
        list_known_people,
        find_empty_seat,
        locate_person,
        speak,
    ]
