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
from services import PerceptionService


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


    actuator = create_actuator_agent(model, walkieAI, walkie)
    vision = create_vision_agent(model, walkieAI, walkie)
    database = create_database_agent(
        model, walkieAI, walkie
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
