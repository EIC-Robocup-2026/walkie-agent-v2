from __future__ import annotations

import os
from typing import Optional

from langchain_core.tools import tool

from agents.core.tool_decorators import parallelable_tool, sequential_tool
from agents.core.robot_context import RobotContext
from agents.vision_agent.detect import find_object_in_image
from interfaces.walkie_interface import WalkieInterface


# COCO keypoint indices used to summarize "arm raised"
_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6
_LEFT_WRIST, _RIGHT_WRIST = 9, 10


def _summarize_pose(pose) -> str:
    """Compact one-line pose description from a PersonPose."""
    kpts = {kp.index: kp for kp in pose.keypoints}
    flags = []
    ls, lw = kpts.get(_LEFT_SHOULDER), kpts.get(_LEFT_WRIST)
    rs, rw = kpts.get(_RIGHT_SHOULDER), kpts.get(_RIGHT_WRIST)
    if ls and lw and lw.confidence > 0.3 and ls.confidence > 0.3 and lw.y < ls.y:
        flags.append("left arm raised")
    if rs and rw and rw.confidence > 0.3 and rs.confidence > 0.3 and rw.y < rs.y:
        flags.append("right arm raised")
    return ", ".join(flags) if flags else "standing"


def make_vision_tools(
    walkie: WalkieInterface,
    walkieAI,
    *,
    agent_name: str = "vision",
    scene_store=None,
):
    """Build the vision sub-agent's tool list.

    Detection / caption / pose tools are parallelable; speak is sequential.

    Vision is about the **live camera** only. Long-term "where have I seen X?"
    lookups belong to the Walkie Database sub-agent, so they are not exposed
    here. ``scene_store`` is accepted for signature stability but not used by
    these tools.
    """

    def _capture():
        return walkie.camera.capture_pil()

    @parallelable_tool
    @tool
    def detect_objects_from_view() -> str:
        """Detect and list all objects currently visible in the camera view.

        Use when the user asks "what objects do you see?" or "list all items in view".
        For "do you see a SPECIFIC thing (e.g. the coke)?" prefer `look_for_object`,
        which matches the arm's detector and finds brand-named items this list misses.
        """
        img = _capture()
        objects = walkieAI.image.detect(img)
        # Drop low-confidence boxes: open-vocab detectors emit spurious low-score
        # boxes for absent classes, which read as phantom objects. Mirrors the arm's
        # confidence gate so the two agree on what's "there".
        min_conf = float(os.getenv("VISION_DETECT_MIN_CONFIDENCE", "0.2"))
        objects = [o for o in objects if o.confidence is None or o.confidence >= min_conf]
        if not objects:
            return "No objects detected."
        lines = []
        for o in objects:
            conf = f" conf={o.confidence:.2f}" if o.confidence is not None else ""
            lines.append(f"- {o.class_name}{conf} bbox={tuple(o.bbox)}")
        return "Objects detected:\n" + "\n".join(lines)

    @parallelable_tool
    @tool(parse_docstring=True)
    def look_for_object(query: str) -> str:
        """Check whether a SPECIFIC object is visible right now — the way the arm sees it.

        Use for "do you see the X?" / "is there a X in front of you?" / "find the X in
        view". Unlike `detect_objects_from_view` (which lists generic detector labels),
        this expands the target into visual descriptors and CLIP-reranks the boxes, so
        it finds brand-named items (e.g. "coke") that the bare detector reports as
        "can"/"bottle". This is the same detection contract the grasp/arm system uses,
        so if the arm can grasp it, this will report it as seen.

        Args:
            query: The object to look for, e.g. "coke", "a red can", "the cereal box".

        Returns:
            Whether the object is visible, with the best match's label and confidence.
        """
        img = _capture()
        matches = find_object_in_image(walkieAI, img, query)
        if not matches:
            return f"I do not see {query} in the current view."
        cls, conf, sim, bbox = matches[0]
        conf_s = f" conf={conf:.2f}" if conf is not None else ""
        sim_s = f" clip={sim:.2f}" if sim is not None else ""
        extra = f" (+{len(matches) - 1} other candidate box(es))" if len(matches) > 1 else ""
        return (
            f"Yes, I can see {query} — detected as {cls!r}{conf_s}{sim_s}, "
            f"bbox={bbox}.{extra}"
        )

    @parallelable_tool
    @tool(parse_docstring=True)
    def image_caption(prompt: Optional[str] = None) -> str:
        """Generate a natural-language description of what the camera currently sees.

        Use when you need rich scene context beyond bare object labels.

        Args:
            prompt: Optional steering prompt, e.g. "Describe the table only".

        Returns:
            A free-text caption.
        """
        img = _capture()
        caption = walkieAI.image.caption(img, prompt=prompt)
        return f"Scene: {caption}"

    @parallelable_tool
    @tool
    def detect_people_poses() -> str:
        """Detect people in view and summarize their poses (e.g. arms raised).

        Use when checking for waves, hand-raises, or who is standing/sitting.
        """
        img = _capture()
        people = walkieAI.image.estimate_poses(img)
        if not people:
            return "No people visible."
        lines = []
        for i, p in enumerate(people):
            summary = _summarize_pose(p)
            lines.append(
                f"- person {i+1}: bbox={tuple(p.bbox)} conf={p.confidence:.2f} pose={summary}"
            )
        return "People detected:\n" + "\n".join(lines)

    @parallelable_tool
    @tool
    def get_camera_view_description() -> str:
        """Combined snapshot: detected objects + caption + people poses, in one call.

        Use when you want a full understanding of the scene right now without
        emitting multiple separate tool calls.
        """
        img = _capture()
        # One upload, three tasks fused server-side.
        res = walkieAI.image.process(img, detection=True, caption=True, pose=True)
        objects = res.detection or []
        caption = res.caption or ""
        people = res.pose or []
        parts = [f"Scene caption: {caption}"]
        if objects:
            parts.append(
                "Objects: "
                + ", ".join(
                    f"{o.class_name}({o.confidence:.2f})" if o.confidence else o.class_name
                    for o in objects
                )
            )
        else:
            parts.append("Objects: none")
        if people:
            parts.append(
                "People: "
                + "; ".join(_summarize_pose(p) for p in people)
            )
        else:
            parts.append("People: none")
        return "\n".join(parts)

    @sequential_tool
    @tool(parse_docstring=True)
    def speak(text: str) -> str:
        """Speak text aloud through the robot's speaker.

        Use sparingly: the main Walkie agent already speaks high-level results.

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
        detect_objects_from_view,
        look_for_object,
        image_caption,
        detect_people_poses,
        get_camera_view_description,
        speak,
    ]
