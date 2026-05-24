from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from walkie_sdk import WalkieRobot

from agents.actuator_agent import create_actuator_agent
from agents.core.robot_context import RobotContext
from agents.vision_agent import create_vision_agent
from agents.walkie_agent import create_walkie_main_agent
from client import WalkieAIClient
from db.walkie_db import WalkieVectorDB
from interfaces.walkie_interface import WalkieInterface
from perception import RemoteCLIPEmbedder, RobotPoseLifter, SceneStore
from services import ExploreService, PerceptionService, ScenePerceptionService


ZENOH_PORT = 7447
ROBOT_IP = "127.0.0.1"


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
    return WalkieRobot(
        ip=ROBOT_IP,
        ros_protocol=ros_protocol,
        ros_port=ros_port,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )


def build_model():
    """OpenRouter via the OpenAI-compatible endpoint."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "[main] WARNING: OPENROUTER_API_KEY not set. Agent calls will fail.",
            file=sys.stderr,
        )
    return ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        model=os.getenv("WALKIE_MODEL", "anthropic/claude-sonnet-4.5"),
        temperature=float(os.getenv("WALKIE_TEMPERATURE", "0")),
    )


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


_DISABLED = ("", "0", "off", "none", "false", "no")


def _opt_float(name: str, default: str = "") -> float | None:
    """Parse an env float, treating empty/0/off/none as 'disabled' (None)."""
    raw = os.getenv(name, default).strip()
    if raw.lower() in _DISABLED:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _opt_int(name: str, default: str = "") -> int | None:
    """Parse an env int, treating empty/0/off/none as 'disabled' (None)."""
    raw = os.getenv(name, default).strip()
    if raw.lower() in _DISABLED:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def build_scene_store(walkieAI):
    """Construct the CLIP embedder + SceneStore, probing walkie-ai-server first.

    Returns ``(store, embedder)`` when scene perception is enabled and the
    server's ``/image-embed`` route answers; otherwise ``(None, None)`` so
    the caller falls back to the legacy WalkieVectorDB and skips the loop.
    """
    if not _flag("SCENE_PERCEPTION_ENABLED", "1"):
        print("[scene] disabled via SCENE_PERCEPTION_ENABLED=0")
        return None, None

    embedder = RemoteCLIPEmbedder(walkieAI.image_embed)
    try:
        dim = embedder.dim  # one probe call to /image-embed/embed-image
    except Exception as e:  # noqa: BLE001 — server route may be disabled
        print(
            f"[scene] image-embed unavailable ({e!r}); CLIP scene perception OFF.\n"
            "[scene] Enable the /image-embed blueprint on walkie-ai-server to turn it on.",
            file=sys.stderr,
        )
        return None, None

    store = SceneStore(
        persist_dir=os.getenv("SCENE_CHROMA_DIR", "chroma_db_scene"),
        embedder=embedder,
        frames_dir=os.getenv("SCENE_FRAMES_DIR", "frames"),
        # Refresh the archived thumbnail on every re-sighting so the viewer
        # shows the latest frame, not the first-ever one. Set to 0 to keep the
        # original insert frame for an entry's whole life.
        refresh_frame_on_update=_flag("SCENE_FRAME_REFRESH_ON_UPDATE", "1"),
    )
    print(
        f"[scene] CLIP scene memory ON (dim={dim}, {store.count} existing record(s))"
    )
    return store, embedder


def run_explore_stage(walkieAI, walkie, db) -> None:
    explore = ExploreService(
        walkieAI,
        walkie,
        db,
        interval=float(os.getenv("EXPLORE_INTERVAL_SEC", "1.0")),
        min_sightings=int(os.getenv("EXPLORE_MIN_SIGHTINGS", "5")),
        dedup_radius=float(os.getenv("EXPLORE_DEDUP_RADIUS_M", "1.0")),
        min_conf=float(os.getenv("EXPLORE_MIN_CONF", "0.6")),
    )
    explore.start()
    try:
        print("[Explore] Drive the robot around. Press Enter when done.")
        input()
    finally:
        explore.stop_and_join(timeout=5)
    print(f"[Explore] DB now contains {db.count} confident object(s).")


def run_ready_stage(walkieAI, walkie, db, model) -> None:
    perception = PerceptionService(
        walkieAI,
        walkie,
        RobotContext.get().perception_path,
        interval=float(os.getenv("PERCEPTION_INTERVAL_SEC", "2.0")),
        caption_objects=os.getenv("PERCEPTION_CAPTION_OBJECTS", "0").lower() in ("1", "true", "yes"),
        caption_filter=os.getenv("PERCEPTION_CAPTION_FILTER", "").split(","),
    )
    perception.start()

    # CLIP scene memory (long-term semantic catalogue) — wired to walkie-ai-server.
    # Runs alongside the perception.json live snapshot above.
    scene_store, scene_embedder = build_scene_store(walkieAI)
    scene_service = None
    if scene_store is not None and scene_embedder is not None:
        # Position source for scene entries. "robot" (default) stamps each
        # detection with the robot's own odom pose — coarse but robust when the
        # SDK's get_3d_poses depth-lift is unavailable. "lift" uses the SDK
        # depth+TF lifter (walkie.tools) for true per-object positions.
        pos_source = os.getenv("SCENE_POSITION_SOURCE", "lift").lower()
        scene_lifter = RobotPoseLifter(walkie.status) if pos_source == "robot" else None
        print(f"[scene] position source: {pos_source}")
        # Eviction: without this, removed objects linger in the scene store
        # forever. TTL ages out objects not re-seen for SCENE_PRUNE_TTL_SEC;
        # the radius gates the sweep to the robot's vicinity so objects in
        # rooms it hasn't revisited aren't wrongly deleted while it roams.
        prune_ttl = _opt_float("SCENE_PRUNE_TTL_SEC", "60")
        prune_radius = _opt_float("SCENE_PRUNE_RADIUS_M", "2.5")
        prune_max = _opt_int("SCENE_PRUNE_MAX_RECORDS", "")
        if prune_ttl is not None or prune_max is not None:
            print(
                f"[scene] prune: ttl={prune_ttl}s radius={prune_radius}m "
                f"max_records={prune_max}"
            )
        scene_service = ScenePerceptionService(
            walkieAI,
            walkie,
            scene_store,
            scene_embedder,
            lifter=scene_lifter,
            interval=float(os.getenv("SCENE_PERCEPTION_INTERVAL_SEC", "2.0")),
            min_confidence=float(os.getenv("SCENE_MIN_CONF", "0.0")),
            caption_per_object=_flag("SCENE_CAPTION_PER_OBJECT", "0"),
            prune_ttl_sec=prune_ttl,
            prune_interval_sec=float(os.getenv("SCENE_PRUNE_INTERVAL_SEC", "10")),
            prune_radius_m=prune_radius,
            prune_max_records=prune_max,
        )
        scene_service.start()

    actuator = create_actuator_agent(model, walkieAI, walkie)
    vision = create_vision_agent(
        model, walkieAI, walkie, db, scene_store=scene_store
    )
    walkie_agent = create_walkie_main_agent(
        model, walkieAI, walkie, db, actuator, vision, scene_store=scene_store
    )

    print("[Ready] Listening — speak to Walkie. Ctrl+C to exit.")
    try:
        while True:
            if not os.getenv("DISABLE_LISTENING", "0").lower() in ("1", "true", "yes"):
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
    except KeyboardInterrupt:
        print("\n[main] interrupt — shutting down.")
    finally:
        perception.stop_and_join(timeout=5)
        if scene_service is not None:
            scene_service.stop_and_join(timeout=5)


def main() -> None:
    load_dotenv()
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
        base_url=os.getenv("WALKIE_AI_BASE_URL", "http://localhost:5000"),
    )
    walkie = WalkieInterface(robot)
    db = WalkieVectorDB(
        persist_dir=os.getenv("CHROMA_DIR", "chroma_db"),
        frames_dir=os.getenv("OBJECT_FRAMES_DIR", "object_frames"),
    )
    ctx = RobotContext.init(
        perception_path=os.getenv("PERCEPTION_PATH", "perception.json"),
    )
    model = build_model()

    # ── Stage 1: Explore ──
    ctx.stage = "explore"
    run_explore_stage(walkieAI, walkie, db)

    # ── Stage 2: Ready ──
    ctx.stage = "ready"
    run_ready_stage(walkieAI, walkie, db, model)


if __name__ == "__main__":
    main()
