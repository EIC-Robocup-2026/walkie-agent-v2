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
from services.walkie_graphs import WalkieGraphs
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


def initialize_robot() -> WalkieInterface:
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


def initialize_graphs(model, walkie_ai, walkie_interface):
    """Build the walkie_graphs 3D memory for manipulation, starting the observer.

    The GraspNet grasp path reads each object's stored point cloud from this
    store, so the scene must be populated. The observer thread is started unless
    the stub planner is selected (``WALKIE_GRASP_PLANNER=stub`` needs no DB).
    Returns the :class:`~services.walkie_graphs.WalkieGraphs` (call ``.stop()`` on
    teardown), or ``None`` if construction fails (the grasp path then degrades to
    the stub).
    """
    try:
        from services.walkie_graphs import WalkieGraphs

        graphs = WalkieGraphs(model=model, walkieAI=walkie_ai, walkie=walkie_interface)
    except Exception as exc:  # noqa: BLE001
        print(f"[common] WalkieGraphs unavailable ({exc}); grasp falls back to stub")
        return None
    if os.getenv("WALKIE_GRASP_PLANNER", "graspnet").strip().lower() != "stub":
        try:
            graphs.start()  # background perception fills the scene during the run
        except Exception as exc:  # noqa: BLE001
            print(f"[common] graphs.start() failed ({exc})")
    return graphs


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
    def __init__(self, walkieAI: WalkieAIClient, walkie_interface: WalkieInterface, model: ChatOpenAI, disable_listening: bool = False):
        self.walkieAI = walkieAI
        self.walkie_interface = walkie_interface
        self.disable_listening = disable_listening
        
        self.graphs = WalkieGraphs(model=model, walkieAI=walkieAI, walkie=walkie_interface)
        actuator = create_actuator_agent(model, walkieAI, walkie_interface)
        vision = create_vision_agent(model, walkieAI, walkie_interface)
        database = create_database_agent(
            model, walkieAI, walkie_interface, graphs=self.graphs
        )
        self.walkie_agent = create_walkie_main_agent(
            model, walkieAI, walkie_interface, actuator, vision, database
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