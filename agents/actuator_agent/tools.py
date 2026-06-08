from __future__ import annotations

import math
import os

from langchain_core.tools import tool

from agents.core.tool_decorators import parallelable_tool, sequential_tool
from agents.core.robot_context import RobotContext
from interfaces.walkie_interface import WalkieInterface


def _pause_after_walk() -> bool:
    return os.getenv("PAUSE_AFTER_WALK", "0").lower() in ("1", "true", "yes")


def make_actuator_tools(walkie: WalkieInterface, walkieAI, *, agent_name: str = "actuator"):
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

    return [move_absolute, move_relative, get_current_pose, command_arm, speak]
