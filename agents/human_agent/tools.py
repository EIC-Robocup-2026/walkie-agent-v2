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

    return [describe_person, count_people, speak]
