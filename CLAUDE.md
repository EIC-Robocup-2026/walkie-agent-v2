# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`walkie-agent-v2` is the on-robot brain for **Walkie** — a 4th-gen omnidirectional robot from Chulalongkorn University's EIC team. It orchestrates a LangChain/LangGraph multi-agent system over a real robot body (movement, arm, camera, mic, speaker) plus a separate HTTP server (`walkie-ai-server`) that hosts the heavy AI models (STT, TTS, object detection, image captioning, pose estimation).

This repo is *not* the AI inference server. It's a thin local process that calls into:
- **`walkie-sdk`** (git dep, `EIC-Robocup-2026/walkie-sdk`) → hardware: nav, arm, camera over Zenoh
- **`walkie-ai-server`** at `WALKIE_AI_BASE_URL` (default `http://localhost:5000`) → model inference via `client/`

## Commands

```bash
# Setup (uv-based, Python 3.12 required)
uv sync                                # install deps (resolves walkie-sdk from git)
cp .env.example .env                   # then fill OPENROUTER_API_KEY

# Run the robot
uv run python main.py                  # full pipeline (Ready stage by default)

# Run a competition task directly (each builds its own WalkieBrain + scorecard)
uv run python -m tasks.GPSR.run        # GPSR (rulebook 5.3)
uv run python -m tasks.Final.run       # RoboCup@Home Finals (chapter 6); DISABLE_LISTENING=1 to type

# Standalone client/robot demos (need walkie-ai-server running; some need the robot or a webcam).
# Run as modules so repo root is on sys.path for `from client import ...`.
uv run python -m manual_tests.test_robot_object_detection
uv run python -m manual_tests.test_captioning
uv run python -m manual_tests.test_pose_estimation
uv run python -m manual_tests.test_object_segmentation
uv run python -m manual_tests.test_graphs_live   # live realtime_explore ingest + Rerun viz

# The real pytest suite is under tests/ (pyproject testpaths=["tests"]).
# The interactive demo scripts live in manual_tests/ (webcam/robot/live server,
# guarded by __main__) — deliberately OUTSIDE testpaths so pytest never collects them.

# Wipe the walkie_world scene store for a clean slate (run with the robot stopped).
# Removes the scene store (graph_scene/) and the snapshot ring buffer (graph_buffer/).
uv run python -m services.realtime_explore.tools.reset      # asks for confirmation
uv run python -m services.realtime_explore.tools.reset -y   # no confirmation
#   (or: ./run.sh reset  /  ./run.sh fresh)

# Build a scene OFFLINE from a recorded snapshot buffer (deterministic; no robot):
uv run python -m services.realtime_explore.tools.replay graph_buffer [--pose-mode auto --tsdf --store graph_scene]

# Check Open3D GPU support (for pose_mode=auto / TSDF):
uv run python -m services.realtime_explore.tools.check_gpu
```

To run without the microphone (typing prompts at a TTY), set `DISABLE_LISTENING=1`.

## Architecture

### Lifecycle: ready-immediately

`main.py` sets `RobotContext.stage = "ready"` and runs `run_ready_stage` directly — there is no separate explore/catalogue-building stage and no operator "drive around then press Enter" gate.

In the **`ready`** stage a single background thread — the `realtime_explore` perception loop (`services.realtime_explore`, started by `explore.start()`) — runs alongside the agent. Each tick (every `WALKIE_EXPLORE_INTERVAL_SEC`) it captures an RGB-D frame, runs one masked open-vocabulary detection scoped to `WALKIE_EXPLORE_INTERESTED_CLASSES`, lifts each mask to a 3D world point, fuses/captions/embeds it into the scene-graph object records, and writes the latest live snapshot to `perception.json`. The agent stack listens to mic input via STT, runs the Walkie agent on each utterance, and speaks back via TTS. So the scene graph fills itself in the background while the robot already takes commands. (Pose/people detection is **not** part of this loop — live pose lookups live only in the Vision agent's tools.)

The legacy explore stage (`ExploreService`) and its object store (`WalkieVectorDB`/`chroma_db`) were removed; the `SceneStore` is the only long-term memory backend. `tools/reset_db --object` / `db_doctor --object` still operate on a leftover `chroma_db/` dir by path for cleanup, but nothing writes it anymore.

### Four-agent topology

All four agents are built by the same factory: `agents/core/agent.py::create_walkie_agent`, which wraps `langchain.agents.create_agent` with a fixed middleware stack.

The sub-agents' tool factories take an optional `ctx` (a `tasks.base.TaskContext`). When a `ctx` is wired (via `WalkieBrain(ctx)` — used by `main.py`'s ready stage, GPSR, and the Final task), each agent gains **skill-backed tools** that call the robot-tested `tasks/` skills against the shared world/blackboard; without a `ctx` only the primitive tools are present. `TaskContext` is imported under `TYPE_CHECKING` and every `tasks.*` import lives **inside** the tool body (lazy), so `import agents…` stays light and pulls the heavy grasp/Open3D stack only when a manipulation tool actually runs.

- **Walkie main** (`agents/walkie_agent/`) — user-facing orchestrator. Owns the conversation thread (`thread_id="main"`). Delegates with `delegate_to_actuator` / `delegate_to_vision` / `delegate_to_database` (sequential tools that invoke the sub-agent graphs synchronously), `speak`, and — with `ctx` — `handle_person_request` (runs the GPSR parse→repeat→`execute_plan` pipeline for a person's spoken request). All long-term spatial-memory lookups go through `delegate_to_database`.
- **Actuator** (`agents/actuator_agent/`) — primitives `move_absolute`, `move_relative`, `get_current_pose`, `command_arm`, `speak`; with `ctx` also `go_to_location` (map name → `go_to_through_door`), `go_through_door` (open + pass, for the exit door), `pick_up_object` / `place_object_down` (the shared grasp/place skills). Arm tools are gated by `FINAL_ARM_CALIBRATED` (announce-only when off). `move_relative` does the local→global frame conversion in-process before calling `walkie.nav.go_to`.
- **Vision** (`agents/vision_agent/`) — **live camera only**: `detect_objects_from_view`, `look_for_object`, `image_caption`, `detect_people_poses`, `find_person_raising_hand` (hand-raise / call-for-help cue), `find_person`, `get_camera_view_description`, `speak`. (Long-term memory lookups were moved out to the Database agent.)
- **Database** (`agents/database_agent/`) — long-term spatial-memory specialist over the `SceneStore` + the map + people: `find_object` (caption-first), `objects_near`, `recently_seen`, `list_known_objects`, `describe_known_scene`, `get_default_location` (object→category→placement, for returning a misplaced object), `objects_in_room`, `recall_person`, `speak`. Use for "where have I seen X / where does X belong / what's near here / who have I met".

Division of labour: "what's in front of me now" → Vision; "where have I seen it / where does it belong / what's stored" → Database; "someone is asking me to do something" → Walkie main's `handle_person_request`.

Sub-agents are invoked as plain tools — there's no streaming or interleaving; the parent blocks until the sub-agent returns its last AIMessage content.

### `WalkieBrain` — the agents↔tasks bridge

`tasks/common.py::WalkieBrain(ctx)` builds the whole agent stack (the four agents above + the `realtime_explore` producer) bound to ONE shared `TaskContext`, so the agents' skill-backed tools act on the same world/people/scorer/blackboard the task uses. It is GPSR's **Tier-2 scoped fallback** (`tasks/GPSR/dispatch.py` routes a failed step to a single sub-agent), and the orchestrator for `main.py`'s ready stage and the Final task. Build `ctx` first, then `brain = WalkieBrain(ctx)`, then `ctx.data["brain"] = brain` (so `handle_person_request` / dispatch can reach it).

### Final task (RoboCup@Home 2026 Finals) — `tasks/Final/`

The integration task (`uv run python -m tasks.Final.run`): the robot autonomously *finds problems and solves them* under a 10-min cap. **Hybrid scaffold + agent** (`subtasks.py`): deterministic handlers (`skills.py`) score the fixed, position-known, high-value problems — welcome a guest through the exit door (`go_to_through_door`), move the laundry basket to the washing machine, close the dishwasher — then `PatrolAndSolve` drives each room and hands it to `brain.walkie_agent` to find + solve one open-ended problem (trash → bin, misplaced object → `get_default_location`, hand-raise → `handle_person_request`). Manipulation is gated by `FINAL_ARM_CALIBRATED`; knobs live in `tasks/Final/config.toml`; the scoresheet is `tasks/Final/scoring.py::FINAL_SHEET`.

### "No plain text output" contract

By design, an AIMessage with no tool calls **ends the agent loop without saying anything to the user**. The only way to talk is to call `speak`, which streams TTS audio and appends to `RobotContext.speech_log`. Every system prompt enforces this — when editing prompts or adding agents, keep the rule explicit, otherwise the agent will silently "respond" in text the user never hears.

### Middleware stack (applied to every agent)

In order, defined in `agents/core/agent.py`:

1. `SummarizationMiddleware` — compresses history past `WALKIE_SUMMARIZE_AT_TOKENS` (default 6000), keeping `WALKIE_SUMMARIZE_KEEP_MSGS` (default 12).
2. `TodoListMiddleware` — gives agents a `write_todos` task tracker.
3. `PerceptionContextMiddleware` — on every model call, reads `perception.json` and appends a `## Current perception` block to the system message. No-op when `RobotContext.stage != "ready"` or the snapshot is older than `PERCEPTION_STALE_SEC`.
4. `RobotContextMiddleware` — appends `## Stage` and `## Recently spoken (any agent)` so sub-agents can see what their siblings just said and avoid repeating.
5. `ToolGroupingMiddleware` — see below.

### Tool parallelism

`agents/core/tool_decorators.py` exposes `@parallelable_tool` and `@sequential_tool`. The grouping middleware (`agents/core/middleware/tool_grouping.py`) inspects the AIMessage's `tool_calls` list and partitions it into runs: consecutive parallelable tools execute concurrently via `asyncio.gather`; a sequential tool runs alone, blocking the next group. **This only works in the async path** — sync `wrap_tool_call` is a no-op because `ToolNode` already serializes in a single thread.

Convention: read-only inspection / DB lookup → parallelable. Anything that moves the robot, plays audio, or delegates to a sub-agent → sequential. When adding a tool, pick the decorator deliberately; the default if you forget is sequential (safer).

### Cross-agent state: `RobotContext`

`agents/core/robot_context.py` is a thread-safe process-wide singleton (`RobotContext.init(...)` in `main.py`, then `RobotContext.get()` everywhere else). It holds:

- `perception_path` — where the `realtime_explore` perception loop writes the live snapshot.
- `stage` — currently always `"ready"` (set in `main.py`); the field is kept because perception middleware gates on it.
- `speech_log` — bounded deque of `(agent_name, text, ts)` appended whenever any agent's `speak` tool fires. Read by `RobotContextMiddleware` to inject into prompts.

There's no other shared state mechanism — graph checkpointing is `InMemorySaver` per-agent (lost on restart). If you need durable conversation state, that's where to extend.

### `WalkieInterface` and `WalkieAIClient` — two different things

- `WalkieInterface` (`interfaces/walkie_interface.py`) — composes the **hardware** sub-clients: `walkie.nav`, `walkie.arm`, `walkie.status`, `walkie.tools` come from the `walkie-sdk` `WalkieRobot`; `walkie.camera`, `walkie.microphone`, `walkie.speaker` are local wrappers in `interfaces/devices/`. The camera defaults to robot mode but can be instantiated with `device=0` for a local webcam.
- `WalkieAIClient` (`client/`) — HTTP client to the **remote AI server**. Each sub-client (`stt`, `tts`, `object_detection`, `pose_estimation`, `image_caption`, `image_embed`, `face_recognition`, `appearance`) holds its own `requests.Session`. Responses are unwrapped from `{"success": true, "data": ...}`; failures raise `WalkieAPIError`.

The two are passed side-by-side everywhere (typically as `walkie, walkieAI`).

### LLM

`build_model()` in `main.py` uses `ChatOpenAI` pointed at OpenRouter (`OPENROUTER_BASE_URL`, defaults to `anthropic/claude-sonnet-4.5`). Switching providers means swapping the `ChatOpenAI` construction; the agent code is provider-agnostic as long as the model supports tool calls.

### Configuration: `config.toml` + module configs + `.env`

Tuning knobs live in **`config.toml`** (version-controlled) plus **module-local `services/*/config.toml`** files (e.g. `services/realtime_explore/config.toml` holds every `WALKIE_EXPLORE_*` knob); secrets/endpoints/transport stay in **`.env`** (gitignored). `walkie_config.py::load_config()` reads the root `config.toml` first, then every `services/*/config.toml`, and `os.environ.setdefault`s every leaf — so the code still reads everything via `os.getenv(NAME, default)` unchanged, and precedence is **shell env > `.env` > root `config.toml` > module `config.toml` > code default** (first-set wins, so the root can override a module knob). Every entrypoint calls `load_dotenv()` then `load_config()`. The TOML keys *are* the exact env-var names; tables are just for grouping. When you add a new tunable, give it a sensible `os.getenv` default in code AND an entry in the owning module's `config.toml` (root `config.toml` for cross-cutting knobs) — don't put it back in `.env`.

### World model (`walkie_world`) + perception producer (`services/realtime_explore`)

> **Refactor note (walkie_world):** the old `services/walkie_graphs` package was split into
> two. **`walkie_world`** is the import-light DOMAIN MODEL + query engine reached as
> **`ctx.world`** — it owns the static map (rooms/locations/doors/object shapes + the
> grounding vocabulary, `walkie_world/map/`), the numpy object scene graph
> (`walkie_world/scene/`: `store.py` = the old `scene.py`, `relations.py`, `ingest.py`),
> and PEOPLE memory (`walkie_world/people/`: face + appearance re-ID, the old
> `perception/people_store.py`). **`services/realtime_explore`** is the PERCEPTION PRODUCER
> (the old capture/build half: `service.py` = renamed `RealtimeExplore`, `buffer/builder/
> associate/poses/tsdf/viz`) that feeds the model via `world.observe_objects(...)`. Exactly
> ONE `WalkieWorld` is built per process (run.py / main.py) and injected into the producer,
> the agents, and `TaskContext.world`. The map editor's object shapes (XY bbox + Z height)
> seed `source="map"` placeholder nodes that perception promotes to point clouds; rooms get
> boundary polygons (`room_at(x,y)`, `is_near_door(x,y)`).

The long-term spatial memory is the **batch-snapshot** pipeline. `walkie_world/__init__.py` is a lazy (PEP 562) facade exposing `WalkieWorld` with the query contract consumers depend on (`query_text/query_near/recently_seen/all_objects/get/relations_of/to_text_description` + `observe_objects`; map: `room/location/obj/.../is_near_door/room_at`; people: `enroll_person/recognize_person*/find_person_by_caption`) and `ObjectNode` fields (`centroid/best_caption/class_name/n_obs/last_seen_ts/captions/id/source/footprint_polygon`). Importing `walkie_world` is import-light (no eager ChromaDB/Open3D/camera; chromadb loads only on first people use). Full pipeline + every knob: [`docs/WALKIE_WORLD.md`](docs/WALKIE_WORLD.md) (still under the old name — being updated). The load-bearing facts:

- **Two decoupled loops, not a real-time fold.** A cheap **capture thread** (`service.py`) grabs 1 RGB-D frame + 1 detect/caption/embed round-trip per `INTERVAL_SEC`, writes the live `perception.json` straight from the detections, and appends a compact `Snapshot` to an on-disk ring buffer (`graph_buffer/`) — no ICP, no fusion, no maintenance. An occasional single-flight **batch build worker** (every `REBUILD_EVERY_N` snapshots) refines poses → lifts every mask with its optimized pose (`geometry.deproject_mask`) → **batch constrained-agglomerative association** (`associate.py`) → **merges into the persisted `SceneStore`, never shrinking** → derives relations → atomically installs the new scene. Queries read the last installed scene.
- **No ChromaDB for the scene.** The store (`walkie_world/scene/store.py`, `graph_scene/`) is a numpy L2-normalized `(N,D)` embedding matrix + `nodes.json` + `edges.json` (+ `map.npz` when TSDF is on). `query_text` is one brute-force cosine matmul over ≤`PRUNE_MAX_RECORDS` (default 500) objects, with a keyword fallback when the embed server is down; a single `RLock` and an immutable-pointer `install()` mean rebuilds never block queries. Survives restart and accretes (builds merge, never shrink). (ChromaDB is still a dep — `walkie_world/people/store.py` uses it for faces + appearance + caption re-ID.)
- **Association is where the precision lives** (`associate.py`): a hard centroid cap kills twin fusion; **mutual-min** cloud overlap kills flat-object→table absorption; same-class CLIP + a stricter **cross-class** CLIP gate (`ASSOC_CROSS_CLASS_CLIP_MIN`) recovers detector label flip-flop (cup↔mug) without fusing distinct objects; complete-linkage + a per-class AABB-extent veto prevents chaining a row of chairs into one blob. `n_obs` is the cluster member count, so the confirmation gate (`MIN_OBS_CONFIRM`, default 2) clears in one build — no multi-sighting lag.
- **Embeddings/detection/captions are remote.** All from walkie-ai-server (`walkieAI.image.process` for masks+caption+embed in one round-trip; `image.embed_text` for queries). Detection is scoped to `WALKIE_EXPLORE_INTERESTED_CLASSES`; `EXCLUDE_CLASSES` (default `person`) never become nodes; only `CAPTION_CLASSES` get captioned. The searchable document is the caption (the detector class is frequently wrong).
- **Pose & volumetric map are staged.** `POSE_MODE=baseline` (trust nav pose, no Open3D) + `TSDF=0` is the default object-recall path. `POSE_MODE=auto` (Open3D RGB-D odometry + pose-graph, seeded/sanity-bounded by nav so it can't do worse) + `TSDF=1` (VoxelBlockGrid volumetric map) is **off until validated on a replayed buffer** — a pose graph can make poses worse than settled nav. Both Open3D paths are import-guarded and degrade to baseline/None.
- **Offline replay is the dev loop.** Record one run on the robot, then iterate deterministically with no robot: `uv run python -m services.realtime_explore.tools.replay graph_buffer [--pose-mode auto --tsdf --store graph_scene]`. Wipe with `uv run python -m services.realtime_explore.tools.reset` (clears `graph_scene/` + `graph_buffer/`). Knobs live in `services/realtime_explore/config.toml` (~46, grouped). **When changing this package, run the bare-numpy tests: `pytest tests/graphs/`.**

## Conventions worth knowing

- **Adding a tool to an agent**: write it in that agent's `tools.py`, decorate with `@parallelable_tool` or `@sequential_tool` *outside* the `@tool` decorator (the wrapper sets the `_walkie_parallelable` attribute on the `BaseTool` instance), and document via Google-style docstring with `parse_docstring=True` if the tool takes args.
- **Adding a new sub-agent**: copy the shape of `agents/vision_agent/` (a `__init__.py` exporting a `create_*_agent` factory, a `prompts.py`, a `tools.py`). Wire it into `main.py:run_ready_stage` and add a `delegate_to_*` tool in `agents/walkie_agent/tools.py`.
- **Atomic perception writes**: `services.realtime_explore.snapshot.write_atomic` writes `perception.json.tmp` then `os.replace` — readers in `PerceptionContextMiddleware` are tolerant of read-during-write but never read a half-written file. Preserve this if you add new on-disk shared state.
- **Bbox conventions**: `walkie-ai-server` returns object bboxes in `xyxy` (used directly in the snapshot JSON and as crop/heading bounds). The `realtime_explore` path lifts each detection's *mask* to 3D via depth deprojection (`interfaces/perception/geometry.py`), not the legacy `bboxes_to_positions` ROS lift, so no `xyxy`→`cxcywh` conversion is involved.
