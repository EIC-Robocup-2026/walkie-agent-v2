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
from db.walkie_db import WalkieVectorDB
from interfaces.walkie_interface import WalkieInterface
from perception import RemoteCLIPEmbedder, RobotPoseLifter, SceneStore
from services import ExploreService, PerceptionService, ScenePerceptionService


ZENOH_PORT = 7447
ROBOT_IP = "127.0.0.1"

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
    # The caption text index (powers caption-first find_object) is written on
    # every new sighting, but data collected before it existed has none. Set
    # SCENE_REINDEX_CAPTIONS=1 once to backfill it for the existing records.
    if _flag("SCENE_REINDEX_CAPTIONS", "0") and store.count:
        try:
            n = store.reindex_captions()
            print(f"[scene] reindexed {n} caption(s) for text search")
        except Exception as e:  # noqa: BLE001 — never block startup on a backfill
            print(f"[scene] caption reindex failed: {e!r}", file=sys.stderr)
    return store, embedder


def _lan_ip() -> str | None:
    """Best-effort primary LAN IPv4 of this machine.

    Opens a UDP socket toward a public address and reads back the local end the
    OS picked — no packet is actually sent, it just forces interface selection.
    Returns ``None`` (e.g. fully offline) so callers can fall back gracefully.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def maybe_start_viewer(stores: list[tuple[str, object]]):
    """Start the read-only Chroma web viewer in a daemon thread, in-process.

    Running it inside *this* process — and reusing the robot's own chromadb
    clients (``stores`` is ``(directory, client)`` pairs) — is what makes it both
    live and safe. A second OS process opening the same dir would spin up its own
    HNSW index and corrupt the store under the robot's concurrent writes (that's
    why the standalone ``tools.chroma_viewer`` defaults to a frozen snapshot
    copy). Sharing the one live client means there's a single index, so browsing
    reflects writes instantly and can't desync.

    Returns the started thread, or ``None`` when disabled / unstartable. Call it
    after the stores are built so their clients exist.
    """
    if not _flag("CHROMA_VIEWER_AUTOSTART", "1"):
        return None
    try:
        import threading

        from tools import chroma_viewer as cv
    except Exception as e:  # noqa: BLE001 — never let the viewer block the robot
        print(f"[viewer] not started ({e!r})", file=sys.stderr)
        return None

    host = os.getenv("CHROMA_VIEWER_HOST", "0.0.0.0")
    port = int(os.getenv("CHROMA_VIEWER_PORT", "8500"))
    cv.build_stores_inprocess(stores)
    cv._build_frame_roots(
        [d for d, _ in stores], os.getenv("SCENE_FRAMES_DIR", "frames")
    )
    # Werkzeug logs a line per request at INFO; keep the command prompt clean.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def _serve() -> None:
        try:
            cv.app.run(
                host=host, port=port, debug=False, use_reloader=False, threaded=True
            )
        except Exception:  # noqa: BLE001 — a viewer crash must not take down the robot
            _log.exception("chroma viewer thread crashed")

    t = threading.Thread(target=_serve, daemon=True, name="ChromaViewer")
    t.start()
    print(
        f"[viewer] live DB viewer on http://{host}:{port} "
        f"(in-process — safe to browse while the robot writes)"
    )
    # A wildcard bind (0.0.0.0/::) is reachable across the LAN but prints an
    # unusable host — resolve and log the real address so teammates know the URL.
    if host in ("0.0.0.0", "::", ""):
        lan_ip = _lan_ip()
        if lan_ip:
            print(f"[viewer] LAN: open http://{lan_ip}:{port} from another machine")
        else:
            print("[viewer] LAN IP unavailable (offline?) — use this host's IP")
    return t


def run_restaurant_task_stage(walkieAI, walkie: WalkieRobot, db, model) -> None:
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
            exclude_classes=[
                c.strip()
                for c in os.getenv("SCENE_EXCLUDE_CLASSES", "person").split(",")
                if c.strip()
            ],
            prune_ttl_sec=prune_ttl,
            prune_interval_sec=float(os.getenv("SCENE_PRUNE_INTERVAL_SEC", "10")),
            prune_radius_m=prune_radius,
            prune_max_records=prune_max,
        )
        scene_service.start()

    # Bring up the live DB viewer in-process (see maybe_start_viewer), reusing the
    # clients these stores already hold so it reads the live indexes safely.
    viewer_stores: list[tuple[str, object]] = [
        (os.getenv("CHROMA_DIR", "chroma_db"), db.client)
    ]
    if scene_store is not None:
        viewer_stores.append(
            (os.getenv("SCENE_CHROMA_DIR", "chroma_db_scene"), scene_store.client)
        )
    maybe_start_viewer(viewer_stores)

    actuator = create_actuator_agent(model, walkieAI, walkie)
    vision = create_vision_agent(
        model, walkieAI, walkie, db, scene_store=scene_store
    )
    database = create_database_agent(
        model, walkieAI, walkie, db, scene_store=scene_store
    )
    walkie_agent = create_walkie_main_agent(
        model, walkieAI, walkie, db, actuator, vision, database, scene_store=scene_store
    )

    def invoke_agent(text: str) -> None:
        walkie_agent.invoke(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"thread_id": "main"}},
        )

    print("[Ready] Listening — speak to Walkie. Ctrl+C to exit.")
    try:
        walkie.arm.go_to_home(group_name="left_arm")
        walkie.arm.control_gripper("left_gripper", 0, blocking=True)
        walkie.nav.go_to(-5.5, 0, -1.57)  # pre-position for the task so the first prompt works

        input("Press to continue...")
        invoke_agent("Please say `Hello! My name is walkie! Welcome to our restaurant! Please follow me to your seat.` After that move to x=3, y=-1.5, heading=40")
        # input("Press to continue...")
        # invoke_agent("Please say `Here is your seat! Please feel free to sit down.`")
        input("Press to continue...")
        invoke_agent("Please say `I'll grab some drinks for you. Just a moment please.` Then move to x=-0.15, y=-0.6, heading=-90")
        input("Press to continue...")
        walkie.arm.control_gripper("left_gripper", 0.04, blocking=True)
        walkie.arm.go_to_pose_relative(group_name="left_arm", x=0.1, y=0, z=0.1, roll=0, pitch=-1.57, yaw=0, blocking=True)
        input("Press to continue...")
        walkie.arm.control_gripper("left_gripper", 0, blocking=True)
        walkie.arm.go_to_pose_relative(group_name="left_arm", x=-0.1, y=0, z=0.1, roll=0, pitch=0, yaw=0, blocking=True)
        invoke_agent("A staff has put a bottle on your tray. Please bring it to the table (move to x=-0.15, y=-0.6, heading=-90) Don't say anything")
        input("Press to continue...")
        walkie.arm.go_to_pose_relative(group_name="left_arm", x=0.1, y=0, z=-0.1, roll=0, pitch=0, yaw=0, blocking=True)
        walkie.arm.control_gripper("left_gripper", 0.04, blocking=True)
        invoke_agent("Please say `Please enjoy your drink. Let me know if you need anything else!`")
        input("Press to continue...")
        walkie.arm.control_gripper("left_gripper", 0, blocking=True)
        walkie.arm.go_to_home(group_name="left_arm")
        input("Press to continue...")
    except KeyboardInterrupt:
        print("\n[main] interrupt — shutting down.")
    finally:
        perception.stop_and_join(timeout=5)
        if scene_service is not None:
            scene_service.stop_and_join(timeout=5)


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
    # ctx.stage = "explore"
    # run_explore_stage(walkieAI, walkie, db)

    # ── Stage 2: Ready ──
    ctx.stage = "ready"
    run_restaurant_task_stage(walkieAI, walkie, db, model)


if __name__ == "__main__":
    main()
