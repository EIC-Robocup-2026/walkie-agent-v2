from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents.core.tool_decorators import parallelable_tool, sequential_tool
from agents.core.robot_context import RobotContext
from agents.core.object_memory import lookup_object_in_memory, query_min_conf, robot_xy
from interfaces.walkie_interface import WalkieInterface


def make_walkie_main_tools(
    walkie: WalkieInterface,
    walkieAI,
    actuator_agent,
    vision_agent,
    database_agent,
    *,
    agent_name: str = "walkie",
    scene_store=None,
):
    """Tools for the main Walkie agent.

    Sub-agents are wrapped as sequential tools (delegate_to_*). Object lookup
    is parallelable. Speak is sequential.

    ``scene_store`` (a :class:`perception.SceneStore`) powers
    ``find_object_from_memory`` and the Walkie Database sub-agent reached via
    ``delegate_to_database``.
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
        return _invoke_subagent(actuator_agent, task, "actuator")

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
        return _invoke_subagent(vision_agent, task, "vision")

    @sequential_tool
    @tool(parse_docstring=True)
    def delegate_to_database(task: str) -> str:
        """Delegate a long-term-memory question to the Walkie Database sub-agent.

        Use for richer database work than a single lookup: "what's near the
        table?", "what did you see in the last minute?", "how many chairs do
        you know about?", or a "where is X?" that may need follow-up reasoning.
        For a plain one-shot "where is X?", prefer `find_object_from_memory`.
        Blocks until the sub-agent finishes.

        Args:
            task: A clear, self-contained question about stored spatial memory.

        Returns:
            The database agent's answer (with coordinates when available).
        """
        print(f"[walkie] -> database: {task!r}")
        return _invoke_subagent(database_agent, task, "database")

    @parallelable_tool
    @tool(parse_docstring=True)
    def find_object_from_memory(
        object_name: str, near_me: bool = False, radius_m: float = 2.0
    ) -> str:
        """Look up where the robot has previously seen an object (long-term DB).

        Fast path: searches stored captions first (text-to-text), then visual
        similarity. Faster than delegating for a simple "where did I see X?".
        Low-confidence positions are filtered out so the result is navigable.

        Set ``near_me=True`` to restrict to the robot's current vicinity (for
        "the X near me / in this room").

        Args:
            object_name: Object name or description to search.
            near_me: Only return matches within ``radius_m`` of the robot now.
            radius_m: Vicinity radius in metres when ``near_me`` is set.

        Returns:
            Top match(es) with map-frame coordinates.
        """
        within = max_dist = None
        if near_me:
            within = robot_xy(walkie)
            if within is None:
                return "Can't search 'near me' — the robot's position is unknown."
            max_dist = float(radius_m)
        return lookup_object_in_memory(
            object_name,
            scene_store=scene_store,
            n_results=5,
            within_radius_of=within,
            max_distance_m=max_dist,
            min_position_conf=query_min_conf(),
        )

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
        find_object_from_memory,
        speak,
    ]
