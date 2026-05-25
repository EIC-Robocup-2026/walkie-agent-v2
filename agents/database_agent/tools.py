"""Tools for the Walkie Database sub-agent.

A focused surface over the long-term spatial memory. The CLIP-backed
:class:`perception.SceneStore` is the primary backend (caption text search +
spatial / recency queries); the legacy :class:`db.walkie_db.WalkieVectorDB`
is used as a fallback for whatever it can answer (object lookup, listing).

All lookups are read-only → ``@parallelable_tool``. ``speak`` moves audio →
``@sequential_tool``.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

from langchain_core.tools import tool

from agents.core.object_memory import lookup_object_in_memory
from agents.core.robot_context import RobotContext
from agents.core.tool_decorators import parallelable_tool, sequential_tool
from db.walkie_db import WalkieVectorDB
from interfaces.walkie_interface import WalkieInterface


def _fmt_entries(entries) -> str:
    """One line per SceneEntry: class @ (x, y, z) conf sightings caption."""
    lines = []
    for e in entries:
        x, y, z = e.position
        lines.append(
            f"- {e.class_name} @ ({x:+.2f}, {y:+.2f}, {z:+.2f}) "
            f"conf={e.position_conf:.2f} sightings={e.sightings} "
            f"caption={e.caption!r}"
        )
    return "\n".join(lines)


def make_database_tools(
    walkie: WalkieInterface,
    walkieAI,
    db: WalkieVectorDB,
    *,
    agent_name: str = "database",
    scene_store=None,
):
    """Build the database sub-agent's tool list.

    ``scene_store``: when supplied (a :class:`perception.SceneStore`), the
    spatial / recency / caption tools use it; otherwise they degrade to what
    the legacy ``db`` can answer (object lookup only).
    """

    @parallelable_tool
    @tool(parse_docstring=True)
    def find_object(query: str) -> str:
        """Find where the robot has previously seen an object, by description.

        Searches stored captions first (text-to-text, so "coffee mug" matches
        "a white ceramic coffee mug"), then visual similarity. This is the
        primary "where is X?" lookup.

        Args:
            query: Name or description of the object (e.g. "red backpack").

        Returns:
            Top match(es) with map-frame coordinates, or a not-found message.
        """
        return lookup_object_in_memory(
            query, scene_store=scene_store, db=db, n_results=5
        )

    @parallelable_tool
    @tool(parse_docstring=True)
    def objects_near(x: float, y: float, radius_m: float = 1.5) -> str:
        """List catalogued objects within a radius of a map-frame point.

        Use for "what's around here / near the table" questions.

        Args:
            x: Map-frame X (metres).
            y: Map-frame Y (metres).
            radius_m: Search radius in metres (default 1.5).

        Returns:
            Objects inside the ball, nearest first, with coordinates.
        """
        if scene_store is None:
            return "Spatial search needs the CLIP scene memory, which is off."
        entries = scene_store.spatial_query(
            center=(float(x), float(y), 0.0), radius_m=float(radius_m)
        )
        if not entries:
            return f"No objects within {radius_m:g}m of ({x:+.2f}, {y:+.2f})."
        return (
            f"{len(entries)} object(s) within {radius_m:g}m of "
            f"({x:+.2f}, {y:+.2f}):\n" + _fmt_entries(entries)
        )

    @parallelable_tool
    @tool(parse_docstring=True)
    def recently_seen(within_seconds: float = 60.0) -> str:
        """List objects whose most recent sighting is within a time window.

        Use for "what did you just see?" questions.

        Args:
            within_seconds: Look-back window in seconds (default 60).

        Returns:
            Recently-seen objects, newest first.
        """
        if scene_store is None:
            return "Recency search needs the CLIP scene memory, which is off."
        since = time.time() - float(within_seconds)
        entries = scene_store.recency_query(since_ts=since)
        if not entries:
            return f"Nothing seen in the last {within_seconds:g}s."
        return (
            f"{len(entries)} object(s) seen in the last {within_seconds:g}s:\n"
            + _fmt_entries(entries)
        )

    @parallelable_tool
    @tool
    def list_known_objects() -> str:
        """Summarize the whole long-term database: total + per-class counts.

        Use for "what do you know about?" / "how many chairs?" questions.
        """
        if scene_store is not None:
            entries = scene_store.recency_query(since_ts=0.0)
            total = len(entries)
            if total == 0:
                return "The scene database is empty."
            by_class = Counter(e.class_name for e in entries).most_common()
            breakdown = ", ".join(f"{cls}×{n}" for cls, n in by_class)
            return f"Scene database holds {total} object(s): {breakdown}."
        # Legacy fallback.
        try:
            rows = db.list_all()
        except Exception as e:  # noqa: BLE001
            return f"Could not read the object database: {e}"
        if not rows:
            return "The object database is empty."
        by_class = Counter(r.get("class_name", "?") for r in rows).most_common()
        breakdown = ", ".join(f"{cls}×{n}" for cls, n in by_class)
        return f"Object database holds {len(rows)} object(s): {breakdown}."

    @sequential_tool
    @tool(parse_docstring=True)
    def speak(text: str) -> str:
        """Speak text aloud through the robot's speaker.

        Use sparingly: the parent agent usually speaks the final answer.

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

    return [find_object, objects_near, recently_seen, list_known_objects, speak]
