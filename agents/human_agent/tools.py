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


# COCO keypoint indices (pose_estimation returns 17 of these per person).
_NOSE = 0
_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6
_LEFT_WRIST, _RIGHT_WRIST = 9, 10
_LEFT_HIP, _RIGHT_HIP = 11, 12
_LEFT_KNEE, _RIGHT_KNEE = 13, 14

_KP_CONF = 0.3  # minimum keypoint confidence to trust a coordinate

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


def _kp_map(pose):
    """index -> keypoint, for the keypoints this pose actually carries."""
    return {kp.index: kp for kp in pose.keypoints}


def _arm_raised(kpts) -> bool:
    """True if either wrist is above (smaller image-y than) its shoulder."""
    for shoulder_i, wrist_i in ((_LEFT_SHOULDER, _LEFT_WRIST), (_RIGHT_SHOULDER, _RIGHT_WRIST)):
        s, w = kpts.get(shoulder_i), kpts.get(wrist_i)
        if s and w and s.confidence > _KP_CONF and w.confidence > _KP_CONF and w.y < s.y:
            return True
    return False


def _posture(kpts) -> str:
    """Best-effort 'sitting' / 'standing' / 'unknown' from torso/leg geometry.

    Heuristic only — pose keypoints are noisy and the lower body is often
    occluded. We compare the hip->knee vertical drop to the shoulder->hip drop:
    when the legs are folded (sitting) the knees sit close to hip height, so the
    ratio collapses. Returns 'unknown' whenever the needed keypoints are missing.
    """
    def avg_y(a, b):
        pa, pb = kpts.get(a), kpts.get(b)
        ys = [p.y for p in (pa, pb) if p and p.confidence > _KP_CONF]
        return sum(ys) / len(ys) if ys else None

    shoulder_y = avg_y(_LEFT_SHOULDER, _RIGHT_SHOULDER)
    hip_y = avg_y(_LEFT_HIP, _RIGHT_HIP)
    knee_y = avg_y(_LEFT_KNEE, _RIGHT_KNEE)
    if shoulder_y is None or hip_y is None or knee_y is None:
        return "unknown"
    torso = hip_y - shoulder_y
    if torso <= 0:
        return "unknown"
    leg_drop = (knee_y - hip_y) / torso
    return "sitting" if leg_drop < 0.5 else "standing"


def _summarize_person(pose) -> dict:
    kpts = _kp_map(pose)
    return {"arm_raised": _arm_raised(kpts), "posture": _posture(kpts)}


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
        rec = people_store.enroll(name, drink, face.embedding)
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
        enroll_person,
        recognize_person,
        list_known_people,
        speak,
    ]
