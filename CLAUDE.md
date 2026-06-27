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

# Standalone client/robot demos (need walkie-ai-server running; some need the robot or a webcam).
# Run as modules so repo root is on sys.path for `from client import ...`.
uv run python -m manual_tests.test_robot_object_detection
uv run python -m manual_tests.test_captioning
uv run python -m manual_tests.test_pose_estimation
uv run python -m manual_tests.test_object_segmentation
uv run python -m manual_tests.test_graphs_live   # live walkie_graphs ingest + Rerun viz

# The real pytest suite is under tests/ (pyproject testpaths=["tests"]).
# The interactive demo scripts live in manual_tests/ (webcam/robot/live server,
# guarded by __main__) — deliberately OUTSIDE testpaths so pytest never collects them.

# Wipe the walkie_graphs store for a clean slate (run with the robot stopped —
# ChromaDB's persistent client is single-process). Removes the node Chroma DB,
# point-cloud sidecars, capture segments, background cloud, thumbnails, and edges.
uv run python -m services.walkie_graphs.tools.reset      # asks for confirmation
uv run python -m services.walkie_graphs.tools.reset -y   # no confirmation
#   (or: ./run.sh reset  /  ./run.sh fresh)

# Check Open3D GPU/ICP support:
uv run python -m services.walkie_graphs.tools.check_gpu
```

To run without the microphone (typing prompts at a TTY), set `DISABLE_LISTENING=1`.

## Architecture

### Lifecycle: ready-immediately

`main.py` sets `RobotContext.stage = "ready"` and runs `run_ready_stage` directly — there is no separate explore/catalogue-building stage and no operator "drive around then press Enter" gate.

In the **`ready`** stage a single background thread — the `walkie_graphs` perception loop (`services.walkie_graphs`, started by `graphs.start()`) — runs alongside the agent. Each tick (every `WALKIE_GRAPHS_INTERVAL_SEC`) it captures an RGB-D frame, runs one masked open-vocabulary detection scoped to `WALKIE_GRAPHS_INTERESTED_CLASSES`, lifts each mask to a 3D world point, fuses/captions/embeds it into the scene-graph object records, and writes the latest live snapshot to `perception.json`. The agent stack listens to mic input via STT, runs the Walkie agent on each utterance, and speaks back via TTS. So the scene graph fills itself in the background while the robot already takes commands. (Pose/people detection is **not** part of this loop — live pose lookups live only in the Vision agent's tools.)

The legacy explore stage (`ExploreService`) and its object store (`WalkieVectorDB`/`chroma_db`) were removed; the `SceneStore` is the only long-term memory backend. `tools/reset_db --object` / `db_doctor --object` still operate on a leftover `chroma_db/` dir by path for cleanup, but nothing writes it anymore.

### Four-agent topology

All four agents are built by the same factory: `agents/core/agent.py::create_walkie_agent`, which wraps `langchain.agents.create_agent` with a fixed middleware stack.

- **Walkie main** (`agents/walkie_agent/`) — user-facing orchestrator. Owns the conversation thread (`thread_id="main"`). Delegates with `delegate_to_actuator` / `delegate_to_vision` / `delegate_to_database` (sequential tools that invoke the sub-agent graphs synchronously), plus a fast-path `find_object_from_memory` and `speak`.
- **Actuator** (`agents/actuator_agent/`) — `move_absolute`, `move_relative`, `get_current_pose`, `command_arm`, `speak`. `move_relative` does the local→global frame conversion in-process before calling `walkie.nav.go_to`.
- **Vision** (`agents/vision_agent/`) — **live camera only**: `detect_objects_from_view`, `image_caption`, `detect_people_poses`, `get_camera_view_description`, `speak`. (Long-term memory lookups were moved out to the Database agent.)
- **Database** (`agents/database_agent/`) — long-term spatial-memory specialist over the `SceneStore`: `find_object` (caption-first), `objects_near`, `recently_seen`, `list_known_objects`, `speak`. Use for "where have I seen X / what's near here / what did I just see".

Division of labour: "what's in front of me now" → Vision; "where have I seen it / what's stored" → Database.

Sub-agents are invoked as plain tools — there's no streaming or interleaving; the parent blocks until the sub-agent returns its last AIMessage content.

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

- `perception_path` — where the `walkie_graphs` perception loop writes the live snapshot.
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

Tuning knobs live in **`config.toml`** (version-controlled) plus **module-local `services/*/config.toml`** files (e.g. `services/walkie_graphs/config.toml` holds every `WALKIE_GRAPHS_*` knob); secrets/endpoints/transport stay in **`.env`** (gitignored). `walkie_config.py::load_config()` reads the root `config.toml` first, then every `services/*/config.toml`, and `os.environ.setdefault`s every leaf — so the code still reads everything via `os.getenv(NAME, default)` unchanged, and precedence is **shell env > `.env` > root `config.toml` > module `config.toml` > code default** (first-set wins, so the root can override a module knob). Every entrypoint calls `load_dotenv()` then `load_config()`. The TOML keys *are* the exact env-var names; tables are just for grouping. When you add a new tunable, give it a sensible `os.getenv` default in code AND an entry in the owning module's `config.toml` (root `config.toml` for cross-cutting knobs) — don't put it back in `.env`.

### Scene memory specifics (`services/walkie_graphs/`)

**Two backends behind one facade — `WALKIE_GRAPHS_BACKEND` (default `v1`).** `services/walkie_graphs/__init__.py` is a lazy (PEP 562) facade that routes `WalkieGraphs` to either v1 or v2; both expose the identical query contract (`query_text/query_near/recently_seen/all_objects/get/relations_of/to_text_description` + `start/stop/observe`) and the same `ObjectNode` fields (`centroid/best_caption/class_name/n_obs/last_seen_ts/captions/id`). Importing the package — or any submodule — is import-light (no eager ChromaDB/Open3D/camera).

- **`v2` (`WALKIE_GRAPHS_BACKEND=v2`) — the batch-snapshot redesign** (`buffer/scene/associate/relations/builder/service_v2/poses/tsdf.py`). Cheap **capture thread** (1 frame + 1 detect/caption/embed round-trip → live `perception.json` + a compact `Snapshot` in an on-disk ring buffer `graph_buffer/`; no ICP/fusion/maintenance) + an occasional **batch build worker** (every `REBUILD_EVERY_N` snapshots → refine poses → lift masks → **batch constrained-agglomerative association** → **merge into the persisted numpy `SceneStore` `graph_scene/`, never shrinking** → derive relations → atomic install). No ChromaDB for the scene (`query_text` is a brute-force cosine matmul over an L2-normalized `(N,D)` matrix). `POSE_MODE=baseline`+`TSDF=0` is Stage 1 (object recall, no Open3D on the build path); `POSE_MODE=auto`+`TSDF=1` is the Stage-2 volumetric map (Open3D pose-graph + VoxelBlockGrid, seeded/bounded by nav — **validate on a replayed buffer first**). Offline tuning: `uv run python -m services.walkie_graphs.tools.replay graph_buffer [--pose-mode auto --tsdf --store graph_scene]`. v2 knobs live under `[graphs.v2]` in the module config. **When changing v2, run the bare-numpy tests: `pytest tests/graphs_v2/`.** See [`docs/WALKIE_GRAPHS.md`](docs/WALKIE_GRAPHS.md) §"Two backends".

- **`v1` (default) — `GraphMemory`** (`services/walkie_graphs/memory.py`), a ConceptGraphs-style real-time 3D scene graph. The full pipeline + every tuning knob is documented in [`docs/WALKIE_GRAPHS.md`](docs/WALKIE_GRAPHS.md); the load-bearing facts:

- **One Chroma collection + point-cloud sidecars.** Object nodes live in a single Chroma collection (`objects`, dir `WALKIE_GRAPHS_CHROMA_DIR=chroma_db_graph`) holding each node's CLIP image embedding + metadata. The 3D point clouds, capture segments, classless background cloud, thumbnails, and relation edges live in `.npz`/JSON sidecars (`graph_pcds/`, `graph_captures/`, `graph_background.npz`, `graph_thumbs/`, `graph_edges.json`). `reset` and `prune` keep them in lock-step.
- **Capture-centric ingest.** Each tick lifts every detection's *mask* to a world-frame cloud via depth deprojection (`geometry.deproject_mask`), assembles the segments + a background cloud into one `Capture`, applies **one rigid ICP correction per frame** (anchored by the background), then upserts each detection. Pose error lives on the capture, not per-object.
- **Two-stage association (insert vs. merge).** `upsert` first tries a geometric+semantic match (`fusion.nn_ratio` cloud overlap + CLIP cosine, gated by `WALKIE_GRAPHS_SIM_THRESHOLD`; cross-class gated harder by `WALKIE_GRAPHS_CROSS_CLASS_SIM_THRESHOLD`), then a visual-K fallback. Spatial candidates come from within `WALKIE_GRAPHS_DEDUP_RADIUS_M`; matched → merge clouds + update node, unmatched → new node.
- **Embeddings are remote.** CLIP image/text embeds come from walkie-ai-server's `/image-embed` (`walkieAI.image_embed`, injected as `GraphMemory.embed_text`). There is **no** local-CLIP / `SCENE_EMBED_BACKEND` path in v2 — detection, captioning, STT, TTS, and embeds all come from the server. If the text-embed call fails, caption lookups degrade gracefully.
- **Caption-led documents, scoped classes.** Detection is scoped to `WALKIE_GRAPHS_INTERESTED_CLASSES`; `WALKIE_GRAPHS_EXCLUDE_CLASSES` (default `person`) are dropped *before* lift (moving things can't be position-deduped, so they'd duplicate endlessly); only `WALKIE_GRAPHS_CAPTION_CLASSES` get captioned. The stored searchable document is the caption — the detector class is frequently wrong, so it's kept in metadata only.
- **Periodic maintenance.** Denoise / merge-overlapping-nodes / per-object refine / free-space carve / prune / relation-derivation run on **staggered tick cadences** (the `WALKIE_GRAPHS_*_EVERY_N` knobs) so no two collide on one tick; prune caps the store at `WALKIE_GRAPHS_PRUNE_MAX_RECORDS` (default 500).
- **Single-process only.** ChromaDB's `PersistentClient` is not safe for concurrent multi-process access, and the `walkie_graphs` loop writes its store **continuously** during the `ready` stage — opening the *same* directory from a second process corrupts the HNSW index (`InternalError: Error finding id`, mismatched query results). To wipe & rebuild: `uv run python -m services.walkie_graphs.tools.reset` (or `./run.sh reset`), then let the `ready` stage repopulate. (The v1 `chroma_viewer` / `db_doctor` snapshot tools were removed with `SceneStore`.)

## Conventions worth knowing

- **Adding a tool to an agent**: write it in that agent's `tools.py`, decorate with `@parallelable_tool` or `@sequential_tool` *outside* the `@tool` decorator (the wrapper sets the `_walkie_parallelable` attribute on the `BaseTool` instance), and document via Google-style docstring with `parse_docstring=True` if the tool takes args.
- **Adding a new sub-agent**: copy the shape of `agents/vision_agent/` (a `__init__.py` exporting a `create_*_agent` factory, a `prompts.py`, a `tools.py`). Wire it into `main.py:run_ready_stage` and add a `delegate_to_*` tool in `agents/walkie_agent/tools.py`.
- **Atomic perception writes**: `services.walkie_graphs.snapshot.write_atomic` writes `perception.json.tmp` then `os.replace` — readers in `PerceptionContextMiddleware` are tolerant of read-during-write but never read a half-written file. Preserve this if you add new on-disk shared state.
- **Bbox conventions**: `walkie-ai-server` returns object bboxes in `xyxy` (used directly in the snapshot JSON and as crop/heading bounds). The `walkie_graphs` path lifts each detection's *mask* to 3D via depth deprojection (`interfaces/perception/geometry.py`), not the legacy `bboxes_to_positions` ROS lift, so no `xyxy`→`cxcywh` conversion is involved.
