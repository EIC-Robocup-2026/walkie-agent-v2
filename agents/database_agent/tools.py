from __future__ import annotations

import time

from langchain_core.tools import tool

from agents.core.tool_decorators import parallelable_tool, sequential_tool
from agents.core.robot_context import RobotContext
from interfaces.walkie_interface import WalkieInterface

_NO_MEM = "Long-term spatial memory is not available right now."


def _fmt_age(ts: float) -> str:
    if not ts:
        return "unknown"
    dt = max(0.0, time.time() - ts)
    if dt < 60:
        return f"{int(dt)}s ago"
    if dt < 3600:
        return f"{int(dt / 60)}m ago"
    return f"{int(dt / 3600)}h ago"


def _fmt_node(n) -> str:
    x, y, z = n.centroid
    desc = n.best_caption or n.class_name
    return (
        f"{desc} [{n.class_name}] at ({x:.2f}, {y:.2f}, {z:.2f}) — "
        f"seen {n.n_obs}x, last {_fmt_age(n.last_seen_ts)}"
    )


def make_database_tools(
    walkie: WalkieInterface,
    walkieAI,
    *,
    agent_name: str = "database",
    graphs=None,
):
    """Build the Database sub-agent's tools over the walkie_graphs 3D memory.

    Lookups are read-only (parallelable); speak is sequential. ``graphs`` is a
    :class:`walkie_graphs.WalkieGraphs`; when None the tools report memory is off.
    """

    @parallelable_tool
    @tool(parse_docstring=True)
    def find_object(query: str) -> str:
        """Find where an object has been seen, by description or name.

        Use for "where is the red mug?", "have you seen a backpack?". Searches the
        long-term 3D memory (captions + appearance) and returns stored locations.

        Args:
            query: A description or class name, e.g. "red mug" or "chair".

        Returns:
            Matching objects with their 3D map coordinates, or a not-found note.
        """
        if graphs is None:
            return _NO_MEM
        hits = graphs.query_text(query, k=5)
        if not hits:
            return f"No stored object matches {query!r}."
        return f"Found {len(hits)} match(es) for {query!r}:\n" + "\n".join(
            f"- {_fmt_node(n)}" for n in hits
        )

    @parallelable_tool
    @tool(parse_docstring=True)
    def objects_near(radius_m: float = 1.5) -> str:
        """List stored objects near the robot's current position.

        Use for "what's around me?", "what's nearby?".

        Args:
            radius_m: Search radius in meters around the robot (default 1.5).

        Returns:
            Nearby stored objects with coordinates, nearest first.
        """
        if graphs is None:
            return _NO_MEM
        pose = walkie.status.get_position()
        if not pose:
            return "I don't know my current position yet."
        center = (float(pose["x"]), float(pose["y"]))
        hits = graphs.query_near(center, radius_m)
        if not hits:
            return f"No stored objects within {radius_m:.1f} m of me."
        return f"{len(hits)} object(s) within {radius_m:.1f} m:\n" + "\n".join(
            f"- {_fmt_node(n)}" for n in hits
        )

    @parallelable_tool
    @tool(parse_docstring=True)
    def recently_seen(limit: int = 5) -> str:
        """List the most recently observed objects.

        Use for "what did you just see?", "what have you seen lately?".

        Args:
            limit: Maximum number of objects to return (default 5).

        Returns:
            The most recently seen objects, newest first.
        """
        if graphs is None:
            return _NO_MEM
        hits = graphs.recently_seen(limit)
        if not hits:
            return "I haven't catalogued any objects yet."
        return "Recently seen:\n" + "\n".join(f"- {_fmt_node(n)}" for n in hits)

    @parallelable_tool
    @tool
    def list_known_objects() -> str:
        """Summarize everything in long-term memory, counted by class.

        Use for "what objects do you know about?", "how many chairs have you seen?".
        """
        if graphs is None:
            return _NO_MEM
        objs = graphs.all_objects()
        if not objs:
            return "I haven't catalogued any objects yet."
        counts: dict[str, int] = {}
        for n in objs:
            counts[n.class_name] = counts.get(n.class_name, 0) + 1
        lines = [
            f"- {cls}: {k}"
            for cls, k in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return (
            f"I know {len(objs)} object(s) across {len(counts)} class(es):\n"
            + "\n".join(lines)
        )

    @parallelable_tool
    @tool
    def describe_known_scene() -> str:
        """Describe the whole stored scene: objects and their spatial relations.

        Use for "what's on the table?", "describe what you've mapped", or any
        question needing relations (on / above / inside / near) between objects.
        """
        if graphs is None:
            return _NO_MEM
        return graphs.to_text_description()

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
        find_object,
        objects_near,
        recently_seen,
        list_known_objects,
        describe_known_scene,
        speak,
    ]
