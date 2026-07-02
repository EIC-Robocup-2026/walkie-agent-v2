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

import ast
import json
import math
import os
import re
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, Sequence, Union, get_args, get_origin

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
    from services.viz import VizSession
    from tasks.scoring import ScoreTracker
    from walkie_world import WalkieWorld


class StepResult(Enum):
    DONE = auto()   # success (or gracefully degraded) -> advance
    RETRY = auto()  # transient failure -> re-run, up to max_retries
    ABORT = auto()  # unrecoverable -> stop the whole task


def _log(task: str, msg: str) -> None:
    print(f"[{task}] {msg}", file=sys.stderr)


def _schema_list_field(schema: type[BaseModel]) -> str | None:
    """Name of the schema's SOLE list-typed field, else None.

    Lets :func:`_parse_to_schema` coerce a model that answered with a bare array
    (``['coke']``) instead of the wrapping object (``{'items': ['coke']}``) — a
    common local-LLM slip — back into the schema.
    """
    lists = [name for name, f in schema.model_fields.items()
             if get_origin(f.annotation) is list]
    return lists[0] if len(lists) == 1 else None


def _literal_choices(annotation) -> list | None:
    """Allowed values if *annotation* is a ``Literal`` (also inside ``Optional``)."""
    origin = get_origin(annotation)
    if origin is Literal:
        return list(get_args(annotation))
    if origin is Union:
        for arg in get_args(annotation):
            found = _literal_choices(arg)
            if found:
                return found
    return None


def _schema_example(annotation):
    """A minimal example value for a pydantic field annotation.

    Drives the local-LLM JSON-mode prompt (:meth:`TaskContext.extract`): small local
    models echo a raw JSON Schema back verbatim (burying the real answer inside
    ``properties.<field>.items``), so instead of dumping the schema we show them a
    concrete instance of the exact shape to fill. ``Optional`` fields render as
    ``null`` (signalling "leave null when not applicable"); ``Literal`` fields render
    as their first choice; nested models recurse.
    """
    origin = get_origin(annotation)
    if origin is Union:  # Optional[X] -> null (teaches "omit/null when irrelevant")
        if type(None) in get_args(annotation):
            return None
        annotation = next(a for a in get_args(annotation) if a is not type(None))
        origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)[0]
    if annotation is str:
        return "..."
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if origin is list:
        (inner,) = get_args(annotation) or (str,)
        return [_schema_example(inner)]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {n: _schema_example(f.annotation) for n, f in annotation.model_fields.items()}
    return "..."


def _schema_field_lines(schema: type[BaseModel], indent: str = "") -> list[str]:
    """Per-field ``- "name": description`` lines, with Literal choices spelled out and
    one level of nesting for list-of-model / model fields."""
    lines: list[str] = []
    for name, f in schema.model_fields.items():
        desc = (f.description or "").strip()
        choices = _literal_choices(f.annotation)
        if choices:
            desc = (desc + " " if desc else "") + "One of: " + " | ".join(map(str, choices)) + "."
        lines.append(f'{indent}- "{name}": {desc}'.rstrip())
        inner = f.annotation
        if get_origin(inner) is list and get_args(inner):
            inner = get_args(inner)[0]
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            lines += _schema_field_lines(inner, indent + "    ")
    return lines


def _schema_prompt(instructions: str, schema: type[BaseModel]) -> str:
    """JSON-mode system prompt that shows an EXAMPLE instance, not the raw JSON Schema.

    Dumping ``schema.model_json_schema()`` makes small local models parrot the schema
    back (answer buried in ``properties.<field>.items``), which the strict validator
    then reads as an empty object — the "heard the order but said 'did not catch
    that'" bug. An example object of the exact shape + field descriptions parses
    reliably on the same models (validated against the on-robot qwen/gemma endpoint).
    """
    example = {n: _schema_example(f.annotation) for n, f in schema.model_fields.items()}
    fields = "\n".join(_schema_field_lines(schema))
    return (
        f"{instructions}\n\n"
        f"Respond with ONLY a single JSON object, no markdown fences and no other "
        f"text, in EXACTLY this shape:\n{json.dumps(example)}\n\nFields:\n{fields}"
    )


def _parse_to_schema(content: str, schema: type[BaseModel]) -> BaseModel | None:
    """Best-effort coerce raw LLM text into ``schema`` — tolerant of local-LLM quirks.

    Handles markdown code fences, a JSON object OR a bare JSON/Python array,
    single-quoted Python literals (``['coke']``, ``{'items': ['coke']}``), and a bare
    array wrapped into the schema's sole list field. Returns a validated instance, or
    None if nothing parseable is found.
    """
    text = content.strip()
    if text.startswith("```"):  # ```json ... ``` fences
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    for pattern in (r"\{.*\}", r"\[.*\]"):  # prefer an object, then a bare array
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            continue
        data = None
        for parser in (json.loads, ast.literal_eval):  # JSON first, then Python literal
            try:
                data = parser(m.group(0))
                break
            except Exception:
                continue
        if isinstance(data, dict):
            try:
                return schema.model_validate(data)
            except Exception:
                continue
        if isinstance(data, list):
            field_name = _schema_list_field(schema)
            if field_name is not None:
                try:
                    return schema.model_validate({field_name: data})
                except Exception:
                    continue
    return None


@dataclass
class TaskContext:
    """Shared state + robot primitives passed to every subtask."""

    walkie: WalkieInterface
    walkieAI: WalkieAIClient
    model: ChatOpenAI
    data: dict[str, Any] = field(default_factory=dict)  # cross-step blackboard
    disable_listening: bool = False  # DISABLE_LISTENING: type at a TTY instead of mic
    people: "PeopleStore | None" = None  # face/appearance person memory (optional)
    viz: "VizSession | None" = None  # shared 3D viz session; auto-filled below
    scorer: "ScoreTracker | None" = None  # optional live score tally (see ctx.score)
    world: "WalkieWorld | None" = None  # unified world model (rooms/objects/people); auto-filled

    def __post_init__(self) -> None:
        # Wire the shared viz session so every subtask can draw via ctx.viz (e.g.
        # ctx.viz.axes(...) for a grasp triad). Lazy import keeps tasks/base.py
        # importable on a GPU-less box: get_viz() returns a no-op stub unless
        # WALKIE_VIZ is enabled, and services.viz imports rerun only when it is.
        if self.viz is None:
            from services.viz import get_viz

            self.viz = get_viz()
        # Every task gets a world model (rooms/objects/people query engine) reachable
        # as ctx.world. Tasks that need richer wiring (people memory, a CLIP embed_text,
        # a specific map/scene dir) build their own and pass it in; the default here is
        # objects-only (no chromadb until a people method is used). Import is lazy so
        # tasks/base.py stays importable on a GPU-less box.
        if self.world is None:
            from walkie_world import WalkieWorld

            self.world = WalkieWorld(enable_people=False)

    # --- conversation ---------------------------------------------------

    def say(self, text: str) -> None:
        """Speak through the robot speaker; degrade to print-only on audio failure."""
        print(f"[say] {text}")
        try:
            stream = self.walkieAI.tts.synthesize_stream(text)
            self.walkie.speaker.play_stream(stream, blocking=True)
        except Exception as exc:
            _log("ctx", f"say: TTS/speaker failed ({exc}); text printed only")

    def listen(self, timeout: float = 30.0, min_silence_ms: int | None = None) -> str:
        """One user utterance via mic+STT (or input() when listening disabled).

        ``min_silence_ms`` overrides the mic's end-of-speech silence for THIS
        capture — raise it when the speaker is expected to pause mid-utterance
        (e.g. GPSR's referee reading three commands in one halting stream, where
        the 1 s default would cut the recording at the first stumble).
        """
        if self.disable_listening:
            try:
                return input("[listen] > ").strip()
            except EOFError:
                return ""
        try:
            print(f"[listen] Listening... (timeout {timeout:.0f}s)")
            kwargs = {}
            if min_silence_ms is not None:
                kwargs["min_silence_duration_ms"] = int(min_silence_ms)
            audio = self.walkie.microphone.record_until_silence(timeout=timeout, **kwargs)
            text = self.walkieAI.stt.transcribe(audio)
        except Exception as exc:
            _log("ctx", f"listen: mic/STT failed ({exc})")
            return ""
        text = (text or "").strip()
        print(f"[heard] {text}")
        return text

    def ask(
        self,
        question: str,
        retries: int = 1,
        timeout: float = 30.0,
        min_silence_ms: int | None = None,
    ) -> str:
        """say() the question, listen() for the answer; re-ask on empty.

        ``timeout``/``min_silence_ms`` pass through to :meth:`listen` — widen
        them when the expected answer is a long multi-part utterance.
        """
        for _ in range(retries + 1):
            self.say(question)
            answer = self.listen(timeout=timeout, min_silence_ms=min_silence_ms)
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

        Cloud models use ``with_structured_output`` (native tool-calling/json-schema).
        Local backends (``LLM_USE_LOCAL``) SKIP that and go straight to a JSON-mode
        prompt + a tolerant parser: their structured path reliably returns malformed
        values (e.g. a bare ``['coke']`` instead of ``{"items": ["coke"]}``), so
        attempting it just wastes a round-trip and logs a scary validation error.
        The JSON-mode prompt (:func:`_schema_prompt`) shows the model an EXAMPLE
        instance of the target shape rather than the raw JSON Schema — dumping the
        schema makes small local models parrot it back with the answer buried inside
        ``properties.<field>.items`` (read as an empty object: "heard the order but
        said 'did not catch that'"). Either way the reply is run through
        :func:`_parse_to_schema`, which recovers bare arrays / single-quoted literals
        the strict validator would reject.
        """
        use_local = os.getenv("LLM_USE_LOCAL", "0").strip().lower() in ("1", "true", "yes")
        try:
            structured = self.model.with_structured_output(schema, method="json_schema")
            return structured.invoke(
                [SystemMessage(content=instructions), HumanMessage(content=text)]
            )
        except Exception as exc:
            _log("ctx", f"extract: structured output failed ({exc}); trying JSON fallback")
        try:
            prompt = _schema_prompt(instructions, schema)
            reply = self.model.invoke(
                [SystemMessage(content=prompt), HumanMessage(content=text)]
            )
            parsed = _parse_to_schema(str(reply.content), schema)
            if parsed is not None:
                return parsed
            _log("ctx", f"extract: no parseable {schema.__name__} in reply "
                        f"{str(reply.content)[:160]!r}")
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
        """Blocking nav.go_to to a map-frame pose. True only if the goal was reached.

        ``nav.go_to`` reports the outcome as a *status string* ("SUCCEEDED",
        "CLOSE_ENOUGH", "FAILED", "CANCELED") and only raises on a transport/timeout
        error — a planning failure ("FAILED", e.g. a closed door blocking the route)
        comes back as a return value, not an exception. We must inspect it: returning
        True regardless would mask every nav failure as a success (the door
        ask-and-retry never fires, the robot "arrives" without moving).

        ``WALKIE_NAV_GOAL_TOLERANCE_M`` (default 0 = off) opts into the SDK's
        "close enough" promotion: when set, a Nav2 FAILED whose final
        distance_remaining is within the tolerance is reported as CLOSE_ENOUGH
        (and counts as reached). This only reinterprets the *result* — it does
        NOT loosen Nav2's own goal checker / stop the controller creeping — so an
        in-place rotation that ends ~0 m from the goal stops reading as FAILED,
        and a customer the robot got close to isn't wrongly abandoned. Left off
        by default so door-detection (which keys off a real FAILED) is unchanged
        unless a task opts in (Restaurant sets it in its config.toml).
        """
        _log("ctx", f"goto x={x:.2f} y={y:.2f} heading={math.degrees(heading_rad):.0f}deg")
        # return True
        tol = float(os.getenv("WALKIE_NAV_GOAL_TOLERANCE_M", "0.0"))
        try:
            status = self.walkie.nav.go_to(
                x, y, heading_rad, blocking=True,
                goal_tolerance=(tol if tol > 0 else None),
            )
        except Exception as exc:
            _log("ctx", f"goto failed ({exc})")
            return False
        reached = str(status).upper() in ("SUCCEEDED", "CLOSE_ENOUGH")
        if not reached:
            _log("ctx", f"goto did not reach goal (status={status})")
        return reached

    def current_pose(self) -> dict[str, float]:
        """Robot pose {"x","y","heading"} (radians); zeros if unknown."""
        try:
            pose = self.walkie.status.get_position()
            if pose:
                return pose
        except Exception as exc:
            _log("ctx", f"current_pose failed ({exc})")
        return {"x": 0.0, "y": 0.0, "heading": 0.0}

    def rotate_to(self, heading_rad: float, *, blocking: bool = True) -> bool:
        """Rotate in place — one-shot 'look toward' for a static target.

        ``blocking=True`` (default, unchanged behaviour — every existing caller
        passes a heading only): disable head auto-tilt, drive to the heading and
        wait, then re-enable auto-tilt; returns whether the goal was reached.

        ``blocking=False``: command the rotation and return immediately WITHOUT
        re-enabling auto-tilt (the live-scan / live-approach own the head and keep
        it aimed at people). The caller MUST ``nav.cancel()`` before issuing a
        second non-blocking nav goal — ``nav.go_to(blocking=False)`` spawns a
        *competing* async action thread, it does not preempt the in-flight one
        (walkie_sdk navigation.py). Returns True once the goal was dispatched.
        """
        self.walkie.robot.head.set_auto_tilt(False)
        pose = self.current_pose()
        if blocking:
            reached = self.goto(pose["x"], pose["y"], heading_rad)
            self.walkie.robot.head.set_auto_tilt(True)
            return reached
        try:
            self.walkie.nav.go_to(pose["x"], pose["y"], heading_rad, blocking=False)
        except Exception as exc:
            _log("ctx", f"rotate_to(blocking=False) failed ({exc})")
            return False
        return True

    # --- scoring --------------------------------------------------------

    def score(self, key: str, n: int = 1) -> None:
        """Award *n* units of scoresheet line *key* to the run's ScoreTracker.

        No-op when no scorer is attached. **Observational only** — these are
        *attempted / claimed* points (the robot believes it did the action), not
        referee-awarded, and a bad key or tracker error is logged, never raised,
        so the live tally can never break a task run.
        """
        if self.scorer is None:
            return
        try:
            self.scorer.award(key, n)
        except Exception as exc:
            _log("ctx", f"score({key!r}) failed ({exc})")


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
