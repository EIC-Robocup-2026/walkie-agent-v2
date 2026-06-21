"""Task framework: linear state-machine-style tasks built from SubTask steps.

A competition task (HRI, GPSR, ...) is a `Task` — an ordered list of `SubTask`
steps run one after another over a shared `TaskContext`. The context carries
the robot handles plus generic conversation/nav/vision primitives so new steps
stay short; task-specific perception/geometry helpers live next to the task
(e.g. tasks/HRI/skills.py).

Error philosophy: challenges score partially, so a failed non-critical step
logs and the task moves on — it never raises out of `Task.run()`.
"""

from __future__ import annotations

import json
import math
import re
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Sequence

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel

if TYPE_CHECKING:  # import-time decoupling — these are type-only (annotations are
    # lazy strings under `from __future__ import annotations`), so importing them
    # at runtime would needlessly pull the hardware stack (interfaces ->
    # silero_vad -> torch -> CUDA). Keeping them out lets the pure task logic
    # (parse/ground/dispatch policy, mock-ctx dry runs) import on a GPU-less box.
    from client import WalkieAIClient
    from interfaces.walkie_interface import WalkieInterface
    from perception import PeopleStore


class StepResult(Enum):
    DONE = auto()   # success (or gracefully degraded) -> advance
    RETRY = auto()  # transient failure -> re-run, up to max_retries
    ABORT = auto()  # unrecoverable -> stop the whole task


def _log(task: str, msg: str) -> None:
    print(f"[{task}] {msg}", file=sys.stderr)


@dataclass
class TaskContext:
    """Shared state + robot primitives passed to every subtask."""

    walkie: WalkieInterface
    walkieAI: WalkieAIClient
    model: ChatOpenAI
    data: dict[str, Any] = field(default_factory=dict)  # cross-step blackboard
    disable_listening: bool = False  # DISABLE_LISTENING: type at a TTY instead of mic
    people: "PeopleStore | None" = None  # face/appearance person memory (optional)

    # --- conversation ---------------------------------------------------

    def say(self, text: str) -> None:
        """Speak through the robot speaker; degrade to print-only on audio failure."""
        print(f"[say] {text}")
        try:
            stream = self.walkieAI.tts.synthesize_stream(text)
            self.walkie.speaker.play_stream(stream, blocking=True)
        except Exception as exc:
            _log("ctx", f"say: TTS/speaker failed ({exc}); text printed only")

    def listen(self, timeout: float = 30.0) -> str:
        """One user utterance via mic+STT (or input() when listening disabled)."""
        if self.disable_listening:
            try:
                return input("[listen] > ").strip()
            except EOFError:
                return ""
        try:
            audio = self.walkie.microphone.record_until_silence(timeout=timeout)
            text = self.walkieAI.stt.transcribe(audio)
        except Exception as exc:
            _log("ctx", f"listen: mic/STT failed ({exc})")
            return ""
        text = (text or "").strip()
        print(f"[heard] {text}")
        return text

    def ask(self, question: str, retries: int = 1) -> str:
        """say() the question, listen() for the answer; re-ask on empty."""
        for _ in range(retries + 1):
            self.say(question)
            answer = self.listen()
            if answer:
                return answer
        return ""

    def extract(
        self,
        schema: type[BaseModel],
        instructions: str,
        text: str,
    ) -> BaseModel | None:
        """LLM structured extraction of `schema` from `text`.

        Tries with_structured_output first; falls back to a JSON-mode prompt
        for models without tool-calling (LLM_USE_LOCAL backends).
        """
        try:
            structured = self.model.with_structured_output(schema)
            return structured.invoke(
                [SystemMessage(content=instructions), HumanMessage(content=text)]
            )
        except Exception as exc:
            _log("ctx", f"extract: structured output failed ({exc}); trying JSON fallback")
        try:
            prompt = (
                f"{instructions}\n\nRespond ONLY with a JSON object matching this "
                f"schema:\n{json.dumps(schema.model_json_schema())}"
            )
            reply = self.model.invoke(
                [SystemMessage(content=prompt), HumanMessage(content=text)]
            )
            match = re.search(r"\{.*\}", str(reply.content), re.DOTALL)
            if match:
                return schema.model_validate(json.loads(match.group(0)))
        except Exception as exc:
            _log("ctx", f"extract: JSON fallback failed ({exc})")
        return None

    # --- vision ---------------------------------------------------------

    def capture(self) -> Image.Image | None:
        """One RGB frame from the robot camera."""
        try:
            return self.walkie.camera.capture_pil()
        except Exception as exc:
            _log("ctx", f"capture: camera failed ({exc})")
            return None

    def snapshot(self):
        """Atomic camera snapshot (img + depth + pose + intrinsics) with 3D lifting.

        Unlike capture(), the returned CameraSnapshot can lift masks/bboxes to
        map-frame points against the geometry of the capture instant — accurate
        even after slow detection/LLM round-trips. None on any failure.
        """
        try:
            return self.walkie.capture_snapshot()
        except Exception as exc:
            _log("ctx", f"snapshot failed ({exc})")
            return None

    # --- navigation -----------------------------------------------------

    def goto(self, x: float, y: float, heading_rad: float) -> bool:
        """Blocking nav.go_to to a map-frame pose."""
        _log("ctx", f"goto x={x:.2f} y={y:.2f} heading={math.degrees(heading_rad):.0f}deg")
        try:
            self.walkie.nav.go_to(x, y, heading_rad, blocking=True)
            return True
        except Exception as exc:
            _log("ctx", f"goto failed ({exc})")
            return False

    def current_pose(self) -> dict[str, float]:
        """Robot pose {"x","y","heading"} (radians); zeros if unknown."""
        try:
            pose = self.walkie.status.get_position()
            if pose:
                return pose
        except Exception as exc:
            _log("ctx", f"current_pose failed ({exc})")
        return {"x": 0.0, "y": 0.0, "heading": 0.0}

    def rotate_to(self, heading_rad: float) -> bool:
        """Rotate in place — one-shot 'look toward' for a static target."""
        pose = self.current_pose()
        return self.goto(pose["x"], pose["y"], heading_rad)


class SubTask(ABC):
    """One step of a task. Subclasses implement run(ctx) -> StepResult."""

    max_retries: int = 1   # extra attempts after the first
    critical: bool = False  # exhausted retries: critical -> abort task, else continue

    def __init__(self, name: str | None = None):
        self.name = name or type(self).__name__

    @abstractmethod
    def run(self, ctx: TaskContext) -> StepResult: ...


class Task:
    """Ordered subtasks executed linearly with retry/abort handling."""

    def __init__(self, name: str, subtasks: Sequence[SubTask], ctx: TaskContext):
        self.name = name
        self.subtasks = list(subtasks)
        self.ctx = ctx

    def run(self) -> bool:
        """Run all steps. Returns True if the end was reached without ABORT."""
        total = len(self.subtasks)
        for i, step in enumerate(self.subtasks, 1):
            for attempt in range(step.max_retries + 1):
                suffix = f" (attempt {attempt + 1})" if attempt else ""
                _log(self.name, f"step {i}/{total}: {step.name}{suffix}")
                try:
                    result = step.run(self.ctx)
                except Exception:
                    _log(self.name, f"step {step.name} raised:\n{traceback.format_exc()}")
                    result = StepResult.RETRY
                if result is StepResult.DONE:
                    break
                if result is StepResult.ABORT:
                    _log(self.name, f"step {step.name} aborted the task")
                    return False
            else:
                if step.critical:
                    _log(self.name, f"critical step {step.name} failed; stopping")
                    return False
                _log(self.name, f"step {step.name} failed; continuing")
        _log(self.name, "task complete")
        return True
