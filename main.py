from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from walkie_sdk import WalkieRobot

from walkie_config import load_config

from agents.actuator_agent import create_actuator_agent
from agents.core.robot_context import RobotContext
from agents.database_agent import create_database_agent
from agents.vision_agent import create_vision_agent
from agents.walkie_agent import create_walkie_main_agent
from client import WalkieAIClient
from interfaces.walkie_interface import WalkieInterface
from services.walkie_graphs import WalkieGraphs


ZENOH_PORT = 7447

_log = logging.getLogger("main")


def get_robot() -> WalkieRobot:
    # ROS services go over rosbridge, camera over zenoh (the SDK's own default
    # split). bboxes_to_positions -> get_3d_poses uses the custom service type
    # perception/srv/GetObPose; zenoh_ros2_sdk serializes client-side and has no
    # definition for that package (its registry only knows standard ROS repos),
    # so a zenoh service call fails outright. rosbridge_server is a ROS2 node
    # with `perception` sourced, so it resolves the type server-side. Requires a
    # rosbridge_server on the robot at ros_port (default 9090). To force zenoh
    # (only if you've shipped the .srv into zenoh_ros2_sdk/messages), set
    # WALKIE_ROS_PROTOCOL=zenoh.
    ros_protocol = os.getenv("WALKIE_ROS_PROTOCOL", "rosbridge")
    ros_port = int(os.getenv("WALKIE_ROS_PORT", str(ZENOH_PORT if ros_protocol == "zenoh" else 9090)))
    # 127.0.0.1 is correct when running on the robot itself (SSH'd in); set
    # WALKIE_ROBOT_IP to walkie's LAN address when running from a developer PC.
    robot_ip = os.getenv("WALKIE_ROBOT_IP", "127.0.0.1")
    return WalkieRobot(
        ip=robot_ip,
        ros_protocol=ros_protocol,
        ros_port=ros_port,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )


def build_model():
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


def run_ready_stage(walkieAI: WalkieAIClient, walkie: WalkieInterface, model: ChatOpenAI) -> None:
    # walkie_graphs: 3D scene-graph spatial memory the Database sub-agent queries, AND the
    # robot's perception loop. Its background observer thread is the single per-frame
    # pipeline: detect (scoped to the interested classes) → lift/caption/embed/upsert into the
    # graph object records → write the live perception.json snapshot the agents read each turn.
    # Built regardless so the Database agent can read existing memory; WALKIE_GRAPHS_ENABLED
    # gates whether the loop runs (and thus whether new objects + snapshots are produced).
    graphs = WalkieGraphs(
        model=model,
        walkieAI=walkieAI,
        walkie=walkie,
        snapshot_path=RobotContext.get().perception_path,
    )
    graphs_enabled = os.getenv("WALKIE_GRAPHS_ENABLED", "1").lower() in ("1", "true", "yes")
    if graphs_enabled:
        graphs.start()  # background perception loop: detect → ingest → write perception.json

    actuator = create_actuator_agent(model, walkieAI, walkie)
    vision = create_vision_agent(model, walkieAI, walkie)
    database = create_database_agent(
        model, walkieAI, walkie, graphs=graphs
    )
    walkie_agent = create_walkie_main_agent(
        model, walkieAI, walkie, actuator, vision, database
    )

    print("[Ready] Listening — speak to Walkie. Ctrl+C to exit.")
    listening_disabled = os.getenv("DISABLE_LISTENING", "0").lower() in ("1", "true", "yes")
    try:
        while True:
            # One bad turn (STT hiccup, OpenRouter timeout, TTS outage, a sub-agent
            # raising) must NOT take the robot down — log it, say a short apology if
            # we still can, and keep listening. Ctrl+C still propagates out to the
            # outer handler below since it's a BaseException, not Exception.
            try:
                if not listening_disabled:
                    audio = walkie.microphone.record_until_silence()
                    text = walkieAI.stt.transcribe(audio)
                else:
                    text = input("Enter your instruction: ")
                text = (text or "").strip()
                if not text:
                    continue
                print(f"[user] {text}")
                walkie_agent.invoke(
                    {"messages": [HumanMessage(content=text)]},
                    config={"configurable": {"thread_id": "main"}},
                )
            except EOFError:
                # Ctrl+D at the typed-input prompt — treat as a clean exit.
                print("\n[main] EOF — shutting down.")
                break
            except Exception:  # noqa: BLE001 — keep serving across transient failures
                _log.exception("command turn failed; staying up")
                try:
                    stream = walkieAI.tts.synthesize_stream(
                        "Sorry, I ran into a problem with that. Please try again."
                    )
                    walkie.speaker.play_stream(stream, blocking=True)
                except Exception:  # noqa: BLE001 — TTS itself may be the thing that's down
                    pass
    except KeyboardInterrupt:
        print("\n[main] interrupt — shutting down.")
    finally:
        # Stop the background perception loop, then tear down the robot connection.
        # The SDK's rosbridge/zenoh threads are non-daemon: leaving them running
        # hangs the interpreter at exit (threading._shutdown), so the process
        # never dies and the next launch finds port 8500 still held. close()
        # disconnects the robot, which stops those threads.
        graphs.stop()
        walkie.close()
        print("[main] shutdown complete.")


def main() -> None:
    load_dotenv()
    # Tuning knobs (perception/scene/explore/viewer) live in config.toml; .env
    # holds only secrets/endpoints/transport. setdefault means .env + real env
    # still win over config.toml.
    load_config()
    # Keep third-party libs quiet. Perception emits INFO logs — per-tick summaries
    # plus the `scene.dedup action=INSERT/UPDATE ...` lines — but default them to
    # WARNING here so they don't bury the prompt while you're commanding the robot.
    # Set WALKIE_LOG_LEVEL=INFO to watch them. (tools/scene_explore.py defaults to
    # INFO instead, since there the whole point is to watch collection happen.)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("perception").setLevel(
        os.getenv("WALKIE_LOG_LEVEL", "WARNING").upper()
    )
    robot = get_robot()
    walkieAI = WalkieAIClient(
        base_url=os.getenv("WALKIE_AI_BASE_URL", "http://10.0.0.213:5000"),
    )
    walkie = WalkieInterface(robot)
    ctx = RobotContext.init(
        perception_path=os.getenv("PERCEPTION_PATH", "perception.json"),
    )
    model = build_model()

    # No explore stage: the robot is ready immediately. The scene graph builds
    # itself in the background (the walkie_graphs perception loop started inside
    # run_ready_stage) while the agent already takes commands — see, update, and
    # act without any "drive around then press Enter" gate.
    ctx.stage = "ready"
    run_ready_stage(walkieAI, walkie, model)


if __name__ == "__main__":
    main()
