from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional

from langchain_core.tools import tool

from agents.core.tool_decorators import parallelable_tool, sequential_tool
from agents.core.robot_context import RobotContext
from interfaces.walkie_interface import WalkieInterface

if TYPE_CHECKING:  # annotation only — never import tasks.* at agent import time
    from tasks.base import TaskContext


def _pause_after_walk() -> bool:
    return os.getenv("PAUSE_AFTER_WALK", "0").lower() in ("1", "true", "yes")


def _arm_enabled() -> bool:
    """Arm-motion gate. Off by default → manipulation tools ANNOUNCE instead of moving
    the arm (mirrors LAUNDRY_/RESTAURANT_ARM_CALIBRATED). The Finals task opts in via
    ``FINAL_ARM_CALIBRATED=1``; ``WALKIE_ARM_ENABLED=1`` is the generic override."""
    val = os.getenv("FINAL_ARM_CALIBRATED") or os.getenv("WALKIE_ARM_ENABLED", "0")
    return val.lower() in ("1", "true", "yes")


def make_actuator_tools(
    walkie: WalkieInterface, walkieAI, *, agent_name: str = "actuator",
    ctx: "TaskContext | None" = None,
):
    """Build the actuator sub-agent's tool list bound to a specific robot.

    Returns a list of LangChain tools tagged sequential/parallelable.
    """

    def _get_pose_dict():
        pose = walkie.status.get_position()
        if pose is None:
            return {"x": 0.0, "y": 0.0, "heading": 0.0}
        return pose

    @sequential_tool
    @tool(parse_docstring=True)
    def move_absolute(x: float, y: float, heading: float = 0.0) -> str:
        """Move the robot to a specific (x, y) position on the map, with optional heading.

        Use when the goal is given in map coordinates. Units: x, y in meters; heading
        in degrees (0 = forward/east, 90 = left/north).

        Args:
            x: Target x coordinate in meters (map frame).
            y: Target y coordinate in meters (map frame).
            heading: Target heading in degrees (default: 0).

        Returns:
            Result of the navigation (success status string).
        """
        print(f"[actuator] move_absolute x={x} y={y} heading={heading}")
        heading_rad = math.radians(heading)
        result = walkie.nav.go_to(x=x, y=y, heading=heading_rad, blocking=True)
        if _pause_after_walk():
            input("[actuator] Press Enter to continue...")
        
        return f"Robot moved (status={result})"

    @sequential_tool
    @tool(parse_docstring=True)
    def move_relative(x: float, y: float, heading: float = 0.0) -> str:
        """Move the robot relative to its current pose.

        Use for commands like "go forward N meters" or "turn left 90 degrees".
        In the robot's local frame: +x = forward, +y = left. Units: meters for x, y;
        degrees for heading (positive = counterclockwise).

        Args:
            x: Distance forward in meters (negative = backward).
            y: Distance left in meters (negative = right).
            heading: Change in heading in degrees (positive = turn left).

        Returns:
            Result of the movement (success status string).
        """
        print(f"[actuator] move_relative x={x} y={y} heading={heading}")
        pose = _get_pose_dict()
        x_cur, y_cur, heading_cur_rad = pose["x"], pose["y"], pose["heading"]
        d_heading_rad = math.radians(heading)
        x_global = x_cur + x * math.cos(heading_cur_rad) - y * math.sin(heading_cur_rad)
        y_global = y_cur + x * math.sin(heading_cur_rad) + y * math.cos(heading_cur_rad)
        target_heading = heading_cur_rad + d_heading_rad
        print(
            f"[actuator] -> global x={x_global:.2f} y={y_global:.2f} heading={target_heading:.2f}rad"
        )
        result = walkie.nav.go_to(
            x=x_global, y=y_global, heading=target_heading, blocking=True
        )
        if _pause_after_walk():
            input("[actuator] Press Enter to continue...")
        return f"Robot moved (status={result})"

    @parallelable_tool
    @tool
    def get_current_pose() -> str:
        """Get the robot's current pose (x, y in meters, heading in degrees).

        Use before planning a relative move or to confirm position after a move.
        """
        pose = _get_pose_dict()
        return (
            f"Current pose: x={pose['x']:+6.2f}  "
            f"y={pose['y']:+6.2f}  "
            f"heading={math.degrees(pose['heading']):+6.2f}deg"
        )

    @sequential_tool
    @tool(parse_docstring=True)
    def command_arm(action: str) -> str:
        """Command the robotic arm to perform an action.

        Use for gestures (e.g. wave, point) or manipulation (e.g. pick up, place).
        Be specific: "wave hello", "point left", "pick up the cup".

        Args:
            action: Natural-language description of the desired arm action.

        Returns:
            Confirmation string.
        """
        print(f"[actuator] command_arm action={action!r}")
        # The walkie-sdk arm module exposes high-level actions; fall back gracefully.
        try:
            if hasattr(walkie.arm, "do"):
                walkie.arm.do(action)
            elif hasattr(walkie.arm, "execute"):
                walkie.arm.execute(action)
            elif hasattr(walkie.arm, "command"):
                walkie.arm.command(action)
        except Exception as e:  # noqa: BLE001
            return f"Arm command failed: {e}"
        if _pause_after_walk():
            input("[actuator] Press Enter to continue...")
        return f"Arm command completed: {action}"

    @sequential_tool
    @tool(parse_docstring=True)
    def speak(text: str) -> str:
        """Speak text aloud through the robot's speaker.

        Use sparingly: the main Walkie agent already speaks high-level results.
        Use this only for brief status updates the user genuinely needs to hear.

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

    # --- skill-backed tools (only when a TaskContext is wired) ---------------
    # These reach the robot-tested tasks.skills (nav-through-door, grasp, place) via the
    # shared ctx. Imports are LAZY (inside each tool) so importing the agent stays light
    # and pulls the heavy grasp/Open3D deps only when a manipulation tool actually runs.
    def _resolve_named_pose(name: str):
        """(canonical, pose) for a room/location name via the arena map, or (None, None)."""
        world = ctx.world
        canon = world.location(name) or world.room(name)
        if not canon:
            return None, None
        return canon, world.location_pose(canon)

    @sequential_tool
    @tool(parse_docstring=True)
    def go_to_location(place: str, room: Optional[str] = None) -> str:
        """Drive to a place named or described in natural language.

        Resolves *place* against the arena map AND the robot's scene memory: it
        understands a room-scoped reference ("the table in the kitchen"), and when
        several places match ("the table" with tables in two rooms) it picks the
        NEAREST one and names which it chose. If the map has no surveyed spot it falls
        back to an object the robot has actually seen. Asks a human to open a door only
        if the route is actually blocked. Prefer this over raw coordinates.

        Args:
            place: A room, placement, or described object — e.g. "the kitchen",
                "the table in the kitchen", "the cabinet", "the plant".
            room: Optional room to scope the search to, e.g. "kitchen" (also parsed
                from a "... in the <room>" phrase in *place*).

        Returns:
            Which place it drove to (naming the chosen one) and whether it arrived.
        """
        from tasks.skills import approach_point, go_to_through_door

        near = None
        try:
            p = walkie.status.get_position()
            if p:
                near = (float(p["x"]), float(p["y"]))
        except Exception:  # noqa: BLE001 — no odometry: resolve without a nearest tiebreak
            near = None
        m = ctx.world.resolve_place(place, room=room, near=near)
        if m is None:
            return f"I couldn't find {place!r} on the map or in what I've seen."
        extra = f" (nearest of: {m.label}, {', '.join(m.candidates)})" if m.candidates else ""
        if m.pose is not None:
            ok = go_to_through_door(ctx, *m.pose, ask_even_if_open=ctx.world.is_barrier(m.name))
        elif m.point is not None:
            stop = float(os.getenv("WALKIE_APPROACH_STOP_M", "0.8"))
            ok = approach_point(ctx, m.point[0], m.point[1], stop_distance=stop)
        else:
            return f"I found the {m.label} but it has no known position yet."
        return f"{'Arrived at' if ok else 'Could not reach'} the {m.label}{extra}."

    @sequential_tool
    @tool(parse_docstring=True)
    def go_through_door(name: str) -> str:
        """Drive to a door/threshold and pass through it, opening it autonomously.

        Always asks for the door to be opened (even if it looks open) — use for the
        exit/apartment door when welcoming a guest. For ordinary navigation use
        `go_to_location`, which only asks when the way is blocked.

        Args:
            name: The door or location name to pass through, e.g. "exit", "entrance".

        Returns:
            Whether the robot got through.
        """
        from tasks.skills import go_to_through_door
        canon, pose = _resolve_named_pose(name)
        if pose is None:
            return f"I don't have a surveyed pose for {name!r}."
        ok = go_to_through_door(ctx, *pose, ask_even_if_open=True)
        return f"{'Passed through' if ok else 'Could not get through'} {(canon or name).replace('_', ' ')}."

    @sequential_tool
    @tool(parse_docstring=True)
    def pick_up_object(description: str) -> str:
        """Pick up an object in front of the robot, by description.

        Re-acquires the object visually and runs the full grasp pipeline; remembers what
        it is holding for a later `place_object_down`. The arm only moves when calibrated
        (FINAL_ARM_CALIBRATED=1); otherwise it announces what it would do.

        Args:
            description: What to grasp, e.g. "the red can", "the cup", "trash on the floor".

        Returns:
            Whether the object was grasped.
        """
        if not _arm_enabled():
            return f"(arm not calibrated) I would pick up the {description} now."
        from tasks.skills import pick_object
        ok = pick_object(ctx, prompts=[description], approach_preference="side")
        return f"{'Picked up' if ok else 'Could not pick up'} the {description}."

    @sequential_tool
    @tool(parse_docstring=True)
    def place_object_down(location: Optional[str] = None) -> str:
        """Put the currently-held object down, optionally driving to a named place first.

        Scans for a clear surface and releases the held object there, reconstructing the
        height it was grasped at. Arm-gated like `pick_up_object`.

        Args:
            location: Optional place to drive to first, e.g. "cabinet", "trash bin".

        Returns:
            Whether the object was placed.
        """
        loc_str = f" at {location}" if location else ""
        if not _arm_enabled():
            return f"(arm not calibrated) I would place the held object{loc_str} now."
        from tasks.skills import go_to_through_door, place_object
        if location:
            canon, pose = _resolve_named_pose(location)
            if pose is not None:
                go_to_through_door(ctx, *pose, ask_even_if_open=ctx.world.is_barrier(canon))
        ok = place_object(ctx)
        return f"{'Placed the object' if ok else 'Could not place the object'}{loc_str}."

    skill_tools = (
        [go_to_location, go_through_door, pick_up_object, place_object_down]
        if ctx is not None else []
    )
    return [move_absolute, move_relative, get_current_pose, *skill_tools, command_arm, speak]
