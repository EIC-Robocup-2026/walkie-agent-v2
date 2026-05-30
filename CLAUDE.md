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

# Standalone client demos (need walkie-ai-server running + local webcam).
# Run as modules so repo root is on sys.path for `from client import ...`.
uv run python -m manual_tests.test_object_detection
uv run python -m manual_tests.test_captioning
uv run python -m manual_tests.test_pose_estimation

# The real pytest suite is under tests/ (pyproject testpaths=["tests"]).
# The interactive demo scripts live in manual_tests/ (webcam/robot/live server,
# guarded by __main__) — deliberately OUTSIDE testpaths so pytest never collects them.

# Wipe a vector DB for a clean slate (run with robot/viewer stopped):
uv run python -m tools.reset_db --object   # legacy object DB (chroma_db) + object_frames
uv run python -m tools.reset_db --scene    # CLIP scene memory (chroma_db_scene) + frames
uv run python -m tools.reset_db --all -y   # both, no confirmation

# Diagnose a corrupt / desynced store, read-only (snapshot copy, never the live files):
uv run python -m tools.db_doctor --scene   # dangling vectors + caption↔entries desync
```

To run without the microphone (typing prompts at a TTY), set `DISABLE_LISTENING=1`.

## Architecture

### Two-stage lifecycle

`main.py` drives the robot through two stages tracked on the `RobotContext` singleton:

1. **`explore`** — `ExploreService` (background thread) detects objects, lifts bboxes to 3D map-frame coordinates via `walkie.tools.bboxes_to_positions`, tracks them across frames, and promotes confident multi-sighting tracks into the `WalkieVectorDB` (ChromaDB, persisted at `CHROMA_DIR`). The operator drives the robot around and presses Enter to end.
2. **`ready`** — `PerceptionService` (background thread) writes the latest scene snapshot to `perception.json` every `PERCEPTION_INTERVAL_SEC`. The agent stack listens to mic input via STT, runs the Walkie agent on each utterance, and speaks back via TTS.

Stage 1 is currently commented out in `main.py:122-125` — re-enable when you want to rebuild the world catalogue. Stage 2 expects a populated `chroma_db/` from a prior explore run for `find_object_from_memory` to be useful.

### Four-agent topology

All four agents are built by the same factory: `agents/core/agent.py::create_walkie_agent`, which wraps `langchain.agents.create_agent` with a fixed middleware stack.

- **Walkie main** (`agents/walkie_agent/`) — user-facing orchestrator. Owns the conversation thread (`thread_id="main"`). Delegates with `delegate_to_actuator` / `delegate_to_vision` / `delegate_to_database` (sequential tools that invoke the sub-agent graphs synchronously), plus a fast-path `find_object_from_memory` and `speak`.
- **Actuator** (`agents/actuator_agent/`) — `move_absolute`, `move_relative`, `get_current_pose`, `command_arm`, `speak`. `move_relative` does the local→global frame conversion in-process before calling `walkie.nav.go_to`.
- **Vision** (`agents/vision_agent/`) — **live camera only**: `detect_objects_from_view`, `image_caption`, `detect_people_poses`, `get_camera_view_description`, `speak`. (Long-term memory lookups were moved out to the Database agent.)
- **Database** (`agents/database_agent/`) — long-term spatial-memory specialist over the `SceneStore` (legacy `WalkieVectorDB` fallback): `find_object` (caption-first), `objects_near`, `recently_seen`, `list_known_objects`, `speak`. Use for "where have I seen X / what's near here / what did I just see".

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

- `perception_path` — where `PerceptionService` writes.
- `stage` — `"explore"` or `"ready"`; toggled in `main.py`.
- `speech_log` — bounded deque of `(agent_name, text, ts)` appended whenever any agent's `speak` tool fires. Read by `RobotContextMiddleware` to inject into prompts.

There's no other shared state mechanism — graph checkpointing is `InMemorySaver` per-agent (lost on restart). If you need durable conversation state, that's where to extend.

### `WalkieInterface` and `WalkieAIClient` — two different things

- `WalkieInterface` (`interfaces/walkie_interface.py`) — composes the **hardware** sub-clients: `walkie.nav`, `walkie.arm`, `walkie.status`, `walkie.tools` come from the `walkie-sdk` `WalkieRobot`; `walkie.camera`, `walkie.microphone`, `walkie.speaker` are local wrappers in `interfaces/devices/`. The camera defaults to robot mode but can be instantiated with `device=0` for a local webcam.
- `WalkieAIClient` (`client/`) — HTTP client to the **remote AI server**. Each sub-client (`stt`, `tts`, `object_detection`, `pose_estimation`, `image_caption`) holds its own `requests.Session`. Responses are unwrapped from `{"success": true, "data": ...}`; failures raise `WalkieAPIError`.

The two are passed side-by-side everywhere (typically as `walkie, walkieAI`).

### LLM

`build_model()` in `main.py` uses `ChatOpenAI` pointed at OpenRouter (`OPENROUTER_BASE_URL`, defaults to `anthropic/claude-sonnet-4.5`). Switching providers means swapping the `ChatOpenAI` construction; the agent code is provider-agnostic as long as the model supports tool calls.

### Configuration: `config.toml` + `.env`

Tuning knobs live in **`config.toml`** (version-controlled), secrets/endpoints/transport in **`.env`** (gitignored). `walkie_config.py::load_config()` reads `config.toml` and `os.environ.setdefault`s every leaf — so the code still reads everything via `os.getenv(NAME, default)` unchanged, and precedence is **shell env > `.env` > `config.toml` > code default**. Every entrypoint calls `load_dotenv()` then `load_config()` (main.py, tools/chroma_viewer.py, tools/scene_explore.py, tools/reset_db.py). The TOML keys *are* the exact env-var names; tables are just for grouping. When you add a new tunable, give it a sensible `os.getenv` default in code AND an entry in `config.toml` — don't put it back in `.env`.

### Scene memory specifics (`perception/store.py`)

- **Two collections, one id space.** `scene_entries` holds the CLIP *image* embedding (used for dedup + `visual_query`/`semantic_query`). `scene_captions` holds the CLIP *text* embedding of each caption; `text_query` searches it (text→text) so "where is the mug?" matches the caption words. `find_object_from_memory` / the Database agent query captions first, then fall back to image search. The caption index is written on insert and on caption-changing updates only; `reindex_captions()` (gated by `SCENE_REINDEX_CAPTIONS=1`) backfills pre-existing data. `prune`/`clear` keep both collections in lock-step.
- **Caption-led documents.** `_format_document` returns the caption alone (class name only as a fallback when the caption is empty) — the detector's `class_name` is frequently wrong (it may label a bottle a "mug") while the captioner is reliable, so keeping the wrong class out of the searchable text stops it polluting "find X". The class still lives in metadata for dedup/`where` filters.
- **Embedding backend: local (default) or remote.** `build_scene_store` picks via `SCENE_EMBED_BACKEND`. `local` (`perception.LocalCLIPEmbedder`) loads the same checkpoint (`openai/clip-vit-base-patch16`) in-process via `transformers` (CUDA+fp16 auto), so embeddings need no walkie-ai-server — crash-proof. Needs the optional extra `uv sync --extra clip` (torch+transformers, imported lazily; the rest of the app and the tests run without them). `remote` (`RemoteCLIPEmbedder`) calls `/image-embed`. Both record the same `model_name`, so the two are interchangeable over one store. Image detection/caption/STT/TTS still come from the server regardless.
- **Embed outage is non-fatal for lookups.** `text_query`/`semantic_query` need a CLIP text embed (local tower or the server). If it errors, `lookup_object_in_memory` (shared by `find_object` and `find_object_from_memory`) catches it and degrades to `SceneStore.keyword_query` — a local word-overlap scan over stored captions that needs no embedder — labelling the answer "(semantic search unavailable)". `find_object`/`find_object_from_memory` also take `near_me=True` (restrict to the robot's pose vicinity) and drop matches below `SCENE_QUERY_MIN_CONF`.
- **Thumbnails are object crops.** `_archive_frame` crops the source frame to the detection's bbox (+`SCENE_FRAME_CROP_MARGIN` padding) before saving, so the viewer shows the object, not the whole scene. Disable with `crop_frames_to_bbox=False` / `SCENE_FRAME_CROP=0`.
- **Position fallback.** `process_frame(..., fallback_position=)` (fed the robot pose via the loop's `pose_provider`) stamps detections whose 3D depth-lift returns `None` — small/distant objects, or the whole batch on timeout — with the robot's own pose instead of dropping them. Without a fallback the old drop-on-`None` behavior holds.
- **Dedup candidate sourcing.** `upsert` decides insert-vs-update by feeding `classify` (`perception/dedup.py`) a candidate set of same-class records. That set is **spatial** (`find_nearby`, within `SCENE_DEDUP_RADIUS_M`) plus, when `SCENE_DEDUP_VISUAL_K > 0`, the top-K **visual** neighbours by image embedding *regardless of position* (`_visual_candidates`). The visual path exists because the spatial radius is tight, so a confident re-sighting whose 3D position drifted (lift jitter, or the robot-pose fallback) would otherwise duplicate. Watch the band: `SCENE_DEDUP_RADIUS_M`/`TIGHT_M` must stay **above** lift jitter (~5–15 cm) or even stationary objects re-insert. Visual dedup defaults **off in code** (`os.getenv(...,"0")` — what the unit tests assert) and **on via config.toml** — the standard code-default-plus-config split. Tradeoff: distance-independent visual merge can fuse two genuinely-distinct identical-looking objects; raise `SCENE_EMB_SIM_HIGH` to be stricter.
- **Moving classes don't belong in a spatial catalogue.** `SCENE_EXCLUDE_CLASSES` (default `person`) is filtered in `process_frame` *before* lift/caption/embed, so excluded classes cost nothing and never reach the store — people move every tick and can't be position-deduped, so they'd duplicate endlessly.
- **Single-process only.** ChromaDB's `PersistentClient` is not safe for concurrent multi-process access. During the `ready` stage `ScenePerceptionService` writes `chroma_db_scene` continuously (insert/update + prune), so opening the *same* directory from a second process corrupts the HNSW index — surfacing as `InternalError: Error finding id` and vector queries whose results don't match the stored records. Therefore: `tools/chroma_viewer` opens a **snapshot copy** of each dir by default (`--live` / `CHROMA_VIEWER_LIVE=1` opts back into the live files, only safe with the robot stopped); `tools/db_doctor` does the same; and `text_query`'s id-join (`SceneStore._safe_get_by_ids`) degrades to a per-id fetch so one dangling id can't crash or skew the whole lookup. To diagnose an already-corrupt store use `tools/db_doctor`; to recover, `reindex_captions` fixes caption desync in place, but dangling vectors require a rebuild (`reset_db --scene` + re-explore).

## Conventions worth knowing

- **Adding a tool to an agent**: write it in that agent's `tools.py`, decorate with `@parallelable_tool` or `@sequential_tool` *outside* the `@tool` decorator (the wrapper sets the `_walkie_parallelable` attribute on the `BaseTool` instance), and document via Google-style docstring with `parse_docstring=True` if the tool takes args.
- **Adding a new sub-agent**: copy the shape of `agents/vision_agent/` (a `__init__.py` exporting a `create_*_agent` factory, a `prompts.py`, a `tools.py`). Wire it into `main.py:run_ready_stage` and add a `delegate_to_*` tool in `agents/walkie_agent/tools.py`.
- **Atomic perception writes**: `PerceptionService._write_atomic` writes `perception.json.tmp` then `os.replace` — readers in `PerceptionContextMiddleware` are tolerant of read-during-write but never read a half-written file. Preserve this if you add new on-disk shared state.
- **Bbox conventions**: `walkie-ai-server` returns object bboxes in `xyxy` (used directly in JSON), but `walkie.tools.bboxes_to_positions` expects `cxcywh`. Conversion lives in `_xyxy_to_cxcywh` (duplicated in `services/explore.py` and `services/perception.py`).
