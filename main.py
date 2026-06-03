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
from perception import (
    LocalCLIPEmbedder,
    RemoteCLIPEmbedder,
    RobotPoseLifter,
    SceneStore,
)
from services import PerceptionService, ScenePerceptionService


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
    embedder loads (local CLIP) or the server's ``/image-embed`` route answers;
    otherwise ``(None, None)`` so the caller skips the scene loop and the
    memory tools report that scene memory is off.
    """
    if not _flag("SCENE_PERCEPTION_ENABLED", "1"):
        print("[scene] disabled via SCENE_PERCEPTION_ENABLED=0")
        return None, None

    # Embedding backend: "local" runs CLIP in-process (no walkie-ai-server
    # dependency, crash-proof, GPU-accelerated) — needs `uv sync --extra clip`.
    # "remote" calls the server's /image-embed route (the original behavior).
    backend = os.getenv("SCENE_EMBED_BACKEND", "remote").lower()
    if backend == "local":
        fp16_env = os.getenv("SCENE_CLIP_FP16", "auto").strip().lower()
        fp16 = None if fp16_env in ("", "auto") else fp16_env in ("1", "true", "yes")
        embedder = LocalCLIPEmbedder(
            model_name=os.getenv("SCENE_CLIP_MODEL", "openai/clip-vit-base-patch16"),
            device=os.getenv("SCENE_CLIP_DEVICE") or None,
            fp16=fp16,
        )
        try:
            dim = embedder.dim  # loads the model (downloads on first run)
        except Exception as e:  # noqa: BLE001 — torch/transformers missing or load failed
            print(
                f"[scene] local CLIP unavailable ({e!r}); CLIP scene perception OFF.\n"
                "[scene] Install the extra with `uv sync --extra clip`, or set "
                "SCENE_EMBED_BACKEND=remote.",
                file=sys.stderr,
            )
            return None, None
        print(f"[scene] embedding backend: local (model={embedder.model_name})")
    else:
        embedder = RemoteCLIPEmbedder(walkieAI.image_embed)
        try:
            dim = embedder.dim  # one probe call to /image-embed/embed-image
        except Exception as e:  # noqa: BLE001 — server route may be disabled
            print(
                f"[scene] image-embed unavailable ({e!r}); CLIP scene perception OFF.\n"
                "[scene] Enable the /image-embed blueprint on walkie-ai-server, or set "
                "SCENE_EMBED_BACKEND=local (`uv sync --extra clip`).",
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
        # Archive the object crop (bbox region) not the whole camera frame, so
        # the viewer shows the object itself. Set 0 to keep full frames.
        crop_frames_to_bbox=_flag("SCENE_FRAME_CROP", "1"),
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


_viewer_thread = None  # set on first maybe_start_viewer; guards re-starts.


def _register_viewer_stores(stores: list[tuple[str, object]]) -> None:
    """Swap the viewer's STORES list without restarting the Flask thread.

    Routes consult ``cv.STORES`` per-request, so rebuilding it is enough for the
    sidebar / overview to pick up a newly-added directory (e.g. ``chroma_db_scene``
    after run_ready_stage creates the SceneStore). No-op when CHROMA_VIEWER_AUTOSTART
    is off or the viewer module isn't importable.
    """
    if not _flag("CHROMA_VIEWER_AUTOSTART", "1"):
        return
    try:
        from tools import chroma_viewer as cv
    except Exception:  # noqa: BLE001 — viewer optional
        return
    cv.build_stores_inprocess(stores)
    cv._build_frame_roots(
        [d for d, _ in stores], os.getenv("SCENE_FRAMES_DIR", "frames")
    )


def maybe_start_viewer(stores: list[tuple[str, object]]):
    """Start the read-only Chroma web viewer in a daemon thread, in-process.

    Running it inside *this* process — and reusing the robot's own chromadb
    clients (``stores`` is ``(directory, client)`` pairs) — is what makes it both
    live and safe. A second OS process opening the same dir would spin up its own
    HNSW index and corrupt the store under the robot's concurrent writes (that's
    why the standalone ``tools.chroma_viewer`` defaults to a frozen snapshot
    copy). Sharing the one live client means there's a single index, so browsing
    reflects writes instantly and can't desync.

    Idempotent: calling again only swaps the STORES list (so the scene store
    can be added once run_ready_stage builds it) — the Flask thread keeps running.

    Returns the started thread, or ``None`` when disabled / unstartable. Call it
    after the stores are built so their clients exist.
    """
    global _viewer_thread
    if not _flag("CHROMA_VIEWER_AUTOSTART", "1"):
        return None
    if _viewer_thread is not None:
        _register_viewer_stores(stores)
        return _viewer_thread
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
    _viewer_thread = t
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


def run_ready_stage(walkieAI, walkie, model) -> None:
    perception = PerceptionService(
        walkieAI,
        walkie,
        RobotContext.get().perception_path,
        interval=float(os.getenv("PERCEPTION_INTERVAL_SEC", "2.0")),
        caption_objects=os.getenv("PERCEPTION_CAPTION_OBJECTS", "0").lower() in ("1", "true", "yes"),
        # Empty = caption every object; a non-empty comma list restricts to those
        # classes. Strip blanks so the default "" parses to [] (caption all), not
        # [""] (which matched nothing — captions came out empty).
        caption_filter=[
            c.strip()
            for c in os.getenv("PERCEPTION_CAPTION_FILTER", "").split(",")
            if c.strip()
        ],
        # Depth-lift (walkie.tools.bboxes_to_positions) timeout. The SDK logs
        # "[Tools] Service call timed out after Ns" when the ROS-3D node is
        # slower than this; 2s was too tight, so default to 5s.
        position_timeout=float(os.getenv("PERCEPTION_POSITION_TIMEOUT_SEC", "5.0")),
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
        # When the depth lift fails for a detection, stamping the robot's own
        # pose stores "where the robot stood", not where the object is — sending
        # the robot back there navigates to nothing. Default off: drop the
        # detection so only objects with a real 3D lift enter the catalogue.
        pos_fallback = _flag("SCENE_POSITION_FALLBACK_POSE", "0")
        print(
            f"[scene] position source: {pos_source} "
            f"(lift-fail fallback to robot pose: {'on' if pos_fallback else 'off — drop'})"
        )
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
        # Sanity gate on the SDK depth lift: if a lifted position is farther
        # than this from the robot, treat it as a sensor outlier (bbox center
        # sampling a far wall behind the object, shiny/transparent surface, etc.)
        # and stamp the robot's pose instead. Set to '' to disable. Tune to the
        # camera's reliable depth range (ZED2 mini: ~3-5m, ZED2/2i: ~7-10m).
        max_lift = _opt_float("SCENE_MAX_LIFT_DISTANCE_M", "")
        if max_lift is not None:
            print(f"[scene] max lift distance: {max_lift}m (outliers → robot pose)")
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
            position_fallback_to_pose=pos_fallback,
            # Depth-lift timeout for the scene catalogue (see note above on the
            # live-snapshot service). Raised from 2s → 5s so a slow ROS-3D node
            # doesn't time out and drop the whole frame each tick.
            position_timeout=float(os.getenv("SCENE_POSITION_TIMEOUT_SEC", "5.0")),
            prune_ttl_sec=prune_ttl,
            prune_interval_sec=float(os.getenv("SCENE_PRUNE_INTERVAL_SEC", "10")),
            prune_radius_m=prune_radius,
            prune_max_records=prune_max,
            max_lift_distance_m=max_lift,
        )
        scene_service.start()

    # Bring up the live DB viewer in-process (see maybe_start_viewer), reusing the
    # client the scene store already holds so it reads the live index safely.
    viewer_stores: list[tuple[str, object]] = []
    if scene_store is not None:
        viewer_stores.append(
            (os.getenv("SCENE_CHROMA_DIR", "chroma_db_scene"), scene_store.client)
        )
    maybe_start_viewer(viewer_stores)

    actuator = create_actuator_agent(model, walkieAI, walkie)
    vision = create_vision_agent(model, walkieAI, walkie, scene_store=scene_store)
    database = create_database_agent(
        model, walkieAI, walkie, scene_store=scene_store
    )
    walkie_agent = create_walkie_main_agent(
        model, walkieAI, walkie, actuator, vision, database, scene_store=scene_store
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
        # Stop the background services, then tear down the robot connection.
        # The SDK's rosbridge/zenoh threads are non-daemon: leaving them running
        # hangs the interpreter at exit (threading._shutdown), so the process
        # never dies and the next launch finds port 8500 still held. close()
        # disconnects the robot, which stops those threads.
        perception.stop_and_join(timeout=5)
        if scene_service is not None:
            scene_service.stop_and_join(timeout=5)
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

    # Start the in-process Chroma viewer NOW so it's up from the first second;
    # run_ready_stage re-registers it with the SceneStore client a moment later
    # (idempotent).
    maybe_start_viewer([])

    # No explore stage: the robot is ready immediately. The scene DB builds
    # itself in the background (ScenePerceptionService inside run_ready_stage)
    # while the agent already takes commands — see, update, and act without any
    # "drive around then press Enter" gate.
    ctx.stage = "ready"
    run_ready_stage(walkieAI, walkie, model)


if __name__ == "__main__":
    main()
