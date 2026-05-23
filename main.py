from __future__ import annotations

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
from perception import RemoteCLIPEmbedder, SceneStore
from services import ExploreService, PerceptionService, ScenePerceptionService


ZENOH_PORT = 7447
ROBOT_IP = "127.0.0.1"


def get_robot() -> WalkieRobot:
    return WalkieRobot(
        ip=ROBOT_IP,
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
        scene_service = ScenePerceptionService(
            walkieAI,
            walkie,
            scene_store,
            scene_embedder,
            interval=float(os.getenv("SCENE_PERCEPTION_INTERVAL_SEC", "2.0")),
            min_confidence=float(os.getenv("SCENE_MIN_CONF", "0.0")),
            caption_per_object=_flag("SCENE_CAPTION_PER_OBJECT", "0"),
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
