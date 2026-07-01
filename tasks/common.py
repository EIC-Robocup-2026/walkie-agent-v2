from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from walkie_sdk import WalkieRobot

from agents.actuator_agent import create_actuator_agent
from agents.database_agent import create_database_agent
from agents.vision_agent import create_vision_agent
from agents.walkie_agent import create_walkie_main_agent
from client import WalkieAIClient
from interfaces.walkie_interface import WalkieInterface
from services.realtime_explore import RealtimeExplore
from walkie_config import load_config  # noqa: F401 — re-exported for task entrypoints


ZENOH_PORT = 7447


def load_task_config(task_dir: str | os.PathLike) -> None:
    """Load a task's local ``config.toml`` (if any), then the global one.

    Both loads use setdefault semantics, so first writer wins among the files:
    shell env > .env > <task_dir>/config.toml > root config.toml > code default.
    Call once at startup, after ``load_dotenv()``, instead of ``load_config()``.
    """
    load_config(Path(task_dir) / "config.toml")
    load_config()


def initialize_robot() -> WalkieRobot:
    ros_protocol = os.getenv("WALKIE_ROS_PROTOCOL", "rosbridge")
    ros_port = int(os.getenv("WALKIE_ROS_PORT", str(ZENOH_PORT if ros_protocol == "zenoh" else 9090)))
    # 127.0.0.1 is correct when running on the robot itself (SSH'd in); set
    # WALKIE_ROBOT_IP to walkie's LAN address when running from a developer PC.
    robot_ip = os.getenv("WALKIE_ROBOT_IP", "127.0.0.1")
    robot = WalkieRobot(
        ip=robot_ip,
        ros_protocol=ros_protocol,
        ros_port=ros_port,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )
    walkie_interface = WalkieInterface(robot)
    return walkie_interface


def initialize_llm_model():
    """OpenRouter via the OpenAI-compatible endpoint."""
    use_local = os.getenv("LLM_USE_LOCAL", "0").lower() in ("1", "true", "yes")
    if use_local:
        print("[main] Using local LLM provider")
        base_url = os.getenv("LOCAL_BASE_URL", "http://10.0.0.210:8000/v1")
        api_key = os.getenv("MODEL_API_KEY", "your api key goes here")
        model = os.getenv("LOCAL_MODEL", "qwen3.5-9b")
    else:
        print("[main] Using OpenRouter LLM provider")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print(
                "[main] WARNING: OPENROUTER_API_KEY not set. Agent calls will fail.",
                file=sys.stderr,
            )
        model = os.getenv("WALKIE_MODEL", "anthropic/claude-sonnet-4.5")
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=float(os.getenv("WALKIE_TEMPERATURE", "0")),
    )

class WalkieBrain:
    """The full Walkie agent stack + perception producer, bound to ONE shared ``ctx``.

    Built from an already-constructed :class:`~tasks.base.TaskContext` so the agents'
    skill-wrapping tools act on the SAME world / people / scorer / blackboard the task
    uses (the agents' tools call the ``ctx``-based skills — nav, grasp, door, people).
    The sub-agents are kept as attributes so a caller can invoke a SINGLE one directly:
    GPSR's scoped Tier-2 fallback (dispatch.py) routes a failed step to just
    ``brain.actuator`` / ``brain.vision``, never the full orchestrator; the Finals task
    and main.py's ready stage drive ``brain.walkie_agent`` as the orchestrator.
    """

    def __init__(self, ctx, *, disable_listening: bool = False):
        self.ctx = ctx
        # Convenience handles (several callers/listen_and_act read these).
        self.walkieAI = ctx.walkieAI
        self.walkie_interface = ctx.walkie
        self.world = ctx.world
        self.disable_listening = disable_listening

        # Perception producer: the background loop that feeds ctx.world and writes the
        # live perception.json the agents read each turn (start it with explore.start()).
        # Point it at RobotContext's perception path when one is initialized so the
        # agents' PerceptionContextMiddleware sees the live snapshot; tasks that never
        # init RobotContext (GPSR/HRI) just get no live snapshot file (as before).
        snapshot_path = None
        try:
            from agents.core.robot_context import RobotContext

            snapshot_path = RobotContext.get().perception_path
        except Exception:  # noqa: BLE001 — RobotContext not initialized: no live file
            pass
        self.explore = RealtimeExplore(
            model=ctx.model, walkieAI=ctx.walkieAI, walkie=ctx.walkie, world=ctx.world,
            snapshot_path=snapshot_path,
        )
        # All four agents share `ctx`, so their new tools reach the ctx-based skills.
        self.actuator = create_actuator_agent(ctx.model, ctx.walkieAI, ctx.walkie, ctx=ctx)
        self.vision = create_vision_agent(ctx.model, ctx.walkieAI, ctx.walkie, ctx=ctx)
        self.database = create_database_agent(
            ctx.model, ctx.walkieAI, ctx.walkie, world=ctx.world, ctx=ctx
        )
        self.walkie_agent = create_walkie_main_agent(
            ctx.model, ctx.walkieAI, ctx.walkie,
            self.actuator, self.vision, self.database, ctx=ctx,
        )

    def listen_and_act(self, retry: bool = True, max_retries: int = 3) -> None:
        # Get instructions
        retry_count = 0
        while True:
            if not self.disable_listening:
                audio = self.walkie_interface.microphone.record_until_silence()
                text = self.walkieAI.stt.transcribe(audio)
            else:
                text = input("Enter your instruction: ")
            
            text = (text or "").strip()
            if not text:
                retry_count += 1
                if retry_count >= max_retries:
                    print("Max retries reached. Exiting.")
                    return
                print("No instruction detected. Please try again.")
                continue
            
            print(f"[user] {text}")
            self.walkie_agent.invoke(
                {"messages": [HumanMessage(content=text)]},
                config={"configurable": {"thread_id": "main"}},
            )