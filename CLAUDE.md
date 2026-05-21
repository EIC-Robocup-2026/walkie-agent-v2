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

# Standalone client tests (need walkie-ai-server running + local webcam)
uv run python test_object_detection.py
uv run python test_captioning.py
uv run python test_pose_estimation.py

# pyproject declares pytest under tool.pytest.ini_options testpaths=["tests"],
# but no tests/ directory exists yet. The test_*.py files at the repo root are
# manual demo scripts, NOT a pytest suite.
```

To run without the microphone (typing prompts at a TTY), set `DISABLE_LISTENING=1`.

## Architecture

### Two-stage lifecycle

`main.py` drives the robot through two stages tracked on the `RobotContext` singleton:

1. **`explore`** — `ExploreService` (background thread) detects objects, lifts bboxes to 3D map-frame coordinates via `walkie.tools.bboxes_to_positions`, tracks them across frames, and promotes confident multi-sighting tracks into the `WalkieVectorDB` (ChromaDB, persisted at `CHROMA_DIR`). The operator drives the robot around and presses Enter to end.
2. **`ready`** — `PerceptionService` (background thread) writes the latest scene snapshot to `perception.json` every `PERCEPTION_INTERVAL_SEC`. The agent stack listens to mic input via STT, runs the Walkie agent on each utterance, and speaks back via TTS.

Stage 1 is currently commented out in `main.py:122-125` — re-enable when you want to rebuild the world catalogue. Stage 2 expects a populated `chroma_db/` from a prior explore run for `find_object_from_memory` to be useful.

### Three-agent topology

All three agents are built by the same factory: `agents/core/agent.py::create_walkie_agent`, which wraps `langchain.agents.create_agent` with a fixed middleware stack.

- **Walkie main** (`agents/walkie_agent/`) — user-facing orchestrator. Owns the conversation thread (`thread_id="main"`). Delegates with `delegate_to_actuator` / `delegate_to_vision` (sequential tools that invoke the sub-agent graphs synchronously), plus a fast-path `find_object_from_memory` and `speak`.
- **Actuator** (`agents/actuator_agent/`) — `move_absolute`, `move_relative`, `get_current_pose`, `command_arm`, `speak`. `move_relative` does the local→global frame conversion in-process before calling `walkie.nav.go_to`.
- **Vision** (`agents/vision_agent/`) — `detect_objects_from_view`, `image_caption`, `detect_people_poses`, `find_object_from_memory`, `get_camera_view_description`, `speak`.

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

## Conventions worth knowing

- **Adding a tool to an agent**: write it in that agent's `tools.py`, decorate with `@parallelable_tool` or `@sequential_tool` *outside* the `@tool` decorator (the wrapper sets the `_walkie_parallelable` attribute on the `BaseTool` instance), and document via Google-style docstring with `parse_docstring=True` if the tool takes args.
- **Adding a new sub-agent**: copy the shape of `agents/vision_agent/` (a `__init__.py` exporting a `create_*_agent` factory, a `prompts.py`, a `tools.py`). Wire it into `main.py:run_ready_stage` and add a `delegate_to_*` tool in `agents/walkie_agent/tools.py`.
- **Atomic perception writes**: `PerceptionService._write_atomic` writes `perception.json.tmp` then `os.replace` — readers in `PerceptionContextMiddleware` are tolerant of read-during-write but never read a half-written file. Preserve this if you add new on-disk shared state.
- **Bbox conventions**: `walkie-ai-server` returns object bboxes in `xyxy` (used directly in JSON), but `walkie.tools.bboxes_to_positions` expects `cxcywh`. Conversion lives in `_xyxy_to_cxcywh` (duplicated in `services/explore.py` and `services/perception.py`).
