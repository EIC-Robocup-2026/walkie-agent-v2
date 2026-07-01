from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents.core.tool_decorators import parallelable_tool, sequential_tool
from agents.core.robot_context import RobotContext
from interfaces.walkie_interface import WalkieInterface

if TYPE_CHECKING:  # annotation only — never import tasks.* at agent import time
    from tasks.base import TaskContext


def make_walkie_main_tools(
    walkie: WalkieInterface,
    walkieAI,
    actuator_agent,
    vision_agent,
    database_agent,
    *,
    agent_name: str = "walkie",
    ctx: "TaskContext | None" = None,
):
    """Tools for the main Walkie agent.

    Sub-agents are wrapped as sequential tools (delegate_to_*). Speak is sequential.
    Long-term spatial memory is reached via ``delegate_to_database`` (the Database
    sub-agent over the walkie_graphs ``SceneStore``).
    """

    def _invoke_subagent(graph, task: str, prefix: str) -> str:
        thread_id = f"{prefix}-{uuid.uuid4()}"
        result = graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        msgs = result.get("messages", [])
        if not msgs:
            return f"{prefix} returned no messages."
        last = msgs[-1]
        return getattr(last, "content", "") or f"{prefix} finished."

    @sequential_tool
    @tool(parse_docstring=True)
    def delegate_to_actuator(task: str) -> str:
        """Delegate a movement or arm task to the Actuator sub-agent.

        Example tasks: "go to x=1.5 y=0.3", "move forward 1 meter",
        "turn left 90 degrees", "wave hello", "pick up the red cup".
        Blocks until the sub-agent finishes.

        Args:
            task: A clear, self-contained instruction for the actuator.

        Returns:
            The actuator's final report.
        """
        print(f"[walkie] -> actuator: {task!r}")
        result = _invoke_subagent(actuator_agent, task, "actuator")
        print(f"[walkie] <- actuator: {result!r}")
        return result

    @sequential_tool
    @tool(parse_docstring=True)
    def delegate_to_vision(task: str) -> str:
        """Delegate a perception question to the Vision sub-agent.

        Example tasks: "what do you see?", "where is the red mug?",
        "is anyone raising a hand?", "describe the room".
        Blocks until the sub-agent finishes.

        Args:
            task: A clear perception question.

        Returns:
            The vision agent's answer.
        """
        print(f"[walkie] -> vision: {task!r}")
        result = _invoke_subagent(vision_agent, task, "vision")
        print(f"[walkie] <- vision: {result!r}")
        return result

    @sequential_tool
    @tool(parse_docstring=True)
    def delegate_to_database(task: str) -> str:
        """Delegate a long-term-memory question to the Walkie Database sub-agent.

        Use for any stored-memory question: "where is the red mug?", "what's
        near the table?", "what did you see in the last minute?", "how many
        chairs do you know about?". Blocks until the sub-agent finishes.

        Args:
            task: A clear, self-contained question about stored spatial memory.

        Returns:
            The database agent's answer (with coordinates when available).
        """
        print(f"[walkie] -> database: {task!r}")
        result = _invoke_subagent(database_agent, task, "database")
        print(f"[walkie] <- database: {result!r}")
        return result

    @sequential_tool
    @tool(parse_docstring=True)
    def speak(text: str) -> str:
        """Speak text aloud to the user through the robot's speaker.

        This is the ONLY way for you to communicate with the user. Plain
        assistant text is internal reasoning and is never heard.

        Args:
            text: What to say. Keep it natural and concise.

        Returns:
            Confirmation that the text was spoken.
        """
        print(f"[walkie] speaking: {text!r}")
        stream = walkieAI.tts.synthesize_stream(text)
        walkie.speaker.play_stream(stream, blocking=True)
        try:
            RobotContext.get().add_speech(agent_name, text)
        except RuntimeError:
            pass
        return f"Spoke: {text!r}"

    @sequential_tool
    @tool(parse_docstring=True)
    def handle_person_request(request: str = "") -> str:
        """Take a person's spoken request, repeat it back, and carry it out.

        Use when a person asks Walkie to do something — e.g. after they raise a hand,
        or after you welcome a guest. If `request` is empty, Walkie listens for the
        person to speak. It repeats the understood command aloud (in the Finals,
        correctly repeating the command scores), then executes it with the full GPSR
        command pipeline (parse → ground → Tier-1 skills, Tier-2 sub-agent fallback).

        Args:
            request: The person's request text; leave empty to listen via the microphone.

        Returns:
            A summary of what was understood and the outcome of each command.
        """
        from tasks.GPSR.dispatch import execute_plan
        from tasks.GPSR.parse import parse_commands
        from tasks.GPSR.plan import render_plan_speech

        utterance = (request or "").strip() or ctx.listen()
        if not utterance:
            return "I did not catch any request."
        world = ctx.world.vocab
        parsed = parse_commands(ctx.model, utterance, world)
        if not parsed:
            return f"I heard {utterance!r} but could not turn it into an action."
        brain = ctx.data.get("brain")
        manip = (
            os.getenv("FINAL_ARM_CALIBRATED") or os.getenv("GPSR_ENABLE_MANIPULATION", "0")
        ).lower() in ("1", "true", "yes")
        outcomes = []
        for src, plan in parsed:
            try:  # repeat the command back (Finals: repeating the command scores)
                ctx.say(render_plan_speech(plan))
            except Exception:  # noqa: BLE001 — never let TTS/render abort execution
                pass
            status = execute_plan(ctx, plan, world, brain, manip_enabled=manip)
            outcomes.append(f"{src!r}: {status}")
        return "Handled request — " + "; ".join(outcomes)

    base = [delegate_to_actuator, delegate_to_vision, delegate_to_database]
    extra = [handle_person_request] if ctx is not None else []
    return [*base, *extra, speak]
