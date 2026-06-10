from __future__ import annotations

import os
import sys
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from walkie_sdk import WalkieRobot

from agents.actuator_agent import create_actuator_agent
from agents.database_agent import create_database_agent
from agents.vision_agent import create_vision_agent
from agents.walkie_agent import create_walkie_main_agent
from client import WalkieAIClient
from interfaces.walkie_interface import WalkieInterface
from walkie_graphs import WalkieGraphs


ZENOH_PORT = 7447


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
        base_url = os.getenv("LOCAL_BASE_URL", "http://10.0.0.210:8000/v1")
        api_key = os.getenv("MODEL_API_KEY", "your api key goes here")
        model = os.getenv("LOCAL_MODEL", "qwen3.5-9b")
    else:
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