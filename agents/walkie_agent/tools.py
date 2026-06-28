from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents.core.tool_decorators import parallelable_tool, sequential_tool
from agents.core.robot_context import RobotContext
from interfaces.walkie_interface import WalkieInterface


def make_walkie_main_tools(
    walkie: WalkieInterface,
    walkieAI,
    actuator_agent,
    vision_agent,
    database_agent,
    *,
    agent_name: str = "walkie",
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

    return [
        delegate_to_actuator,
        delegate_to_vision,
        delegate_to_database,
        speak,
    ]
