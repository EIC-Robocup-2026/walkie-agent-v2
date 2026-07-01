from __future__ import annotations

import time
from typing import Optional

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
    world=None,
):
    """Build the Database sub-agent's tools over the walkie_world 3D scene memory.

    Lookups are read-only (parallelable); speak is sequential. ``world`` is a
    :class:`walkie_world.WalkieWorld`; when None the tools report memory is off.
    """

    @parallelable_tool
    @tool(parse_docstring=True)
    def find_object(query: str, room: Optional[str] = None) -> str:
        """Find where an object has been seen, by description or name.

        Use for "where is the red mug?", "have you seen a backpack?", or room-scoped
        "where is the cup in the kitchen?" (pass room="kitchen"). Searches the long-term
        3D memory (captions + appearance); when a room is given, results are limited to
        that room's area.

        Args:
            query: A description or class name, e.g. "red mug" or "chair".
            room: Optional room to limit the search to, e.g. "kitchen".

        Returns:
            Matching objects with their 3D map coordinates, or a not-found note.
        """
        if world is None:
            return _NO_MEM
        hits = world.query_text_in_room(query, room) if room else world.query_text(query, k=5)
        if not hits:
            where = f" in the {room}" if room else ""
            return f"No stored object matches {query!r}{where}."
        where = f" in the {room}" if room else ""
        return f"Found {len(hits)} match(es) for {query!r}{where}:\n" + "\n".join(
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
        if world is None:
            return _NO_MEM
        pose = walkie.status.get_position()
        if not pose:
            return "I don't know my current position yet."
        center = (float(pose["x"]), float(pose["y"]))
        hits = world.query_near(center, radius_m)
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
        if world is None:
            return _NO_MEM
        hits = world.recently_seen(limit)
        if not hits:
            return "I haven't catalogued any objects yet."
        return "Recently seen:\n" + "\n".join(f"- {_fmt_node(n)}" for n in hits)

    @parallelable_tool
    @tool
    def list_known_objects() -> str:
        """Summarize everything in long-term memory, counted by class.

        Use for "what objects do you know about?", "how many chairs have you seen?".
        """
        if world is None:
            return _NO_MEM
        objs = world.all_objects()
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
        if world is None:
            return _NO_MEM
        return world.to_text_description()

    @parallelable_tool
    @tool(parse_docstring=True)
    def get_default_location(item: str) -> str:
        """Where an object or category BELONGS by default (its home placement).

        Use to decide where a misplaced object should be returned — e.g. "where does
        the cola go?" → the cabinet. Resolves object→category→the placement assigned
        that category in the arena map.

        Args:
            item: An object or category name, e.g. "cola", "drinks", "cup".

        Returns:
            The default placement name and its map pose, or a not-found note.
        """
        if world is None:
            return _NO_MEM
        res = world.default_location_for(item)
        if res is None:
            return f"I don't know a default place for {item!r}."
        name, pose = res
        place = name.replace("_", " ")
        if pose is None:
            return f"The {item} belongs at the {place} (pose not surveyed yet)."
        x, y, h = pose
        return (
            f"The {item} belongs at the {place} "
            f"(pose x={x:.2f}, y={y:.2f}, heading={h:.2f} rad)."
        )

    @parallelable_tool
    @tool(parse_docstring=True)
    def objects_in_room(room: str) -> str:
        """List stored objects catalogued in a given room.

        Use for "what have you seen in the kitchen?". Needs the map to define the
        room's boundary; returns objects whose 3D position falls inside it.

        Args:
            room: A room name, e.g. "kitchen", "living room".

        Returns:
            Objects in that room with coordinates, or a not-found note.
        """
        if world is None:
            return _NO_MEM
        hits = world.objects_in_room(room)
        if not hits:
            return f"No stored objects in {room!r} (or the room has no mapped boundary)."
        return f"{len(hits)} object(s) in {room}:\n" + "\n".join(
            f"- {_fmt_node(n)}" for n in hits
        )

    @parallelable_tool
    @tool(parse_docstring=True)
    def recall_person(description: str) -> str:
        """Recall a person the robot has met, by name or appearance description.

        Use for "have we met someone in a red shirt?" / "who did I see?". Searches the
        people memory (faces + attire captions).

        Args:
            description: A name or appearance phrase, e.g. "Alice", "person in a red shirt".

        Returns:
            The best-matching enrolled person, or a not-found note.
        """
        if world is None:
            return _NO_MEM
        try:
            rec = world.find_person_by_caption(description)
        except Exception as exc:  # noqa: BLE001 — people store may be disabled/offline
            return f"People memory lookup failed: {exc}"
        if rec is None:
            return f"I don't recall anyone matching {description!r}."
        bits = [f"name={rec.name}"]
        if getattr(rec, "appearance_caption", None):
            bits.append(f"appearance={rec.appearance_caption!r}")
        if getattr(rec, "last_seen_room", None):
            bits.append(f"last seen in {rec.last_seen_room}")
        return "Recalled person: " + ", ".join(bits) + "."

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
        get_default_location,
        objects_in_room,
        recall_person,
        speak,
    ]
