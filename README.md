# walkie-agent-v2

The on-robot brain for **Walkie** — a 4th-gen omnidirectional robot from Chulalongkorn University's EIC team.

This repo orchestrates a LangChain/LangGraph **multi-agent system** over a real robot body (movement, arm, camera, mic, speaker). It is a thin local process: it does **not** run AI models itself. Instead it talks to two things:

- **`walkie-sdk`** (git dependency) → hardware: navigation, arm, camera over Zenoh.
- **`walkie-ai-server`** (separate HTTP service at `WALKIE_AI_BASE_URL`) → heavy model inference (STT, TTS, object detection, image captioning, pose estimation).

```
┌──────────────────┐      Zenoh        ┌──────────────┐
│  walkie-agent-v2 │ ───────────────►  │ robot body   │  (nav, arm, camera)
│  (this repo)     │                   └──────────────┘
│                  │      HTTP         ┌──────────────────┐
│  LangGraph       │ ───────────────►  │ walkie-ai-server │  (STT/TTS/vision)
│  agent stack     │                   └──────────────────┘
│                  │      HTTP         ┌──────────────┐
│                  │ ───────────────►  │ OpenRouter   │  (the LLM brain)
└──────────────────┘                   └──────────────┘
```

---

## TL;DR — the commands you'll actually run

```bash
uv sync                                  # one-time: install deps (incl. walkie-sdk)
cp .env.example .env                      # one-time: then set OPENROUTER_API_KEY + WALKIE_AI_BASE_URL
```

Day to day it's three commands (often three terminals): **collect → inspect → command.**

```bash
# 1. Build/refresh the scene catalogue: wipes it, then drive the robot to fill it.
uv run python -m tools.scene_explore -y      # press Enter when you're done driving

# 2. Inspect what got stored — read-only web UI at http://localhost:8500
uv run python -m tools.chroma_viewer

# 3. Command the robot (press Enter once to skip the legacy explore prompt).
DISABLE_LISTENING=1 uv run python main.py    # type instructions; drop the prefix to use the mic
```

Steps 1–2 need **`walkie-ai-server`** (with its `/image-embed` route) up at `WALKIE_AI_BASE_URL`. Each command is explained in full below.

---

## How it works (30-second tour)

`main.py` runs the robot through two stages tracked on a process-wide `RobotContext`:

1. **`explore`** — drives around detecting objects, lifts them to 3D map coordinates, and stores confident multi-sighting objects into the legacy vector DB (ChromaDB at `CHROMA_DIR`). `main.py` runs this **first** — press Enter to finish it. To build the newer CLIP scene memory instead, use [`tools/scene_explore`](#building--rebuilding-it-toolsscene_explore).
2. **`ready`** — the default. A background service writes a perception snapshot to `perception.json`; the agent listens to the mic (STT), runs the **Walkie agent** on each utterance, and speaks replies back (TTS).

The agent stack is three agents built from one factory:

- **Walkie main** — user-facing orchestrator; delegates to the two sub-agents.
- **Actuator** — movement + arm (`move_absolute`, `move_relative`, `command_arm`, …).
- **Vision** — `detect_objects_from_view`, `image_caption`, `detect_people_poses`, `find_object_from_memory`, …

> **Note:** the agent only "talks" by calling the `speak` tool. A plain-text model reply with no tool call ends the turn silently — by design.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12** | Pinned in `.python-version`. |
| **[uv](https://docs.astral.sh/uv/)** | Package/venv manager used by this repo. |
| **Git access to `EIC-Robocup-2026/walkie-sdk`** | Resolved automatically by `uv sync`. |
| **OpenRouter API key** | The LLM brain. Get one at [openrouter.ai](https://openrouter.ai). |
| **`walkie-ai-server` running** | Needed for STT/TTS/vision. Default `http://localhost:5000`. |
| **A robot (or local webcam)** | Full nav/arm needs the robot; the camera can fall back to `device=0`. |
| **PortAudio + a mic/speaker** | For `pyaudio`/`sounddevice`. On Debian/Ubuntu: `sudo apt install portaudio19-dev`. |

---

## Step-by-step: getting started

### 1. Install dependencies

```bash
uv sync          # creates .venv and resolves all deps, including walkie-sdk from git
```

### 2. Configure your environment

```bash
cp .env.example .env
```

Open `.env` and at minimum set your LLM key:

```dotenv
OPENROUTER_API_KEY=sk-or-...
```

Key variables (full list in `.env.example`):

| Variable | Default | What it does |
|---|---|---|
| `OPENROUTER_API_KEY` | *(empty)* | **Required.** Agent calls fail without it. |
| `WALKIE_MODEL` | `anthropic/claude-sonnet-4.5` | LLM used by every agent. |
| `WALKIE_AI_BASE_URL` | `http://localhost:5000` | Where `walkie-ai-server` lives. |
| `CHROMA_DIR` | `chroma_db` | Vector DB location (object memory). |
| `PERCEPTION_PATH` | `perception.json` | Where the ready-stage snapshot is written. |
| `PERCEPTION_INTERVAL_SEC` | `10.0` | How often the perception snapshot refreshes. |
| `SCENE_PERCEPTION_ENABLED` | `1` | CLIP scene memory (see below). Set `0` to disable. |
| `SCENE_CHROMA_DIR` | `chroma_db_scene` | Where the CLIP scene memory is persisted. |
| `DISABLE_LISTENING` | *(unset)* | Set `1` to type prompts instead of using the mic. |

### 3. Start the dependencies

- Make sure **`walkie-ai-server`** is up and reachable at `WALKIE_AI_BASE_URL`.
- Make sure the **robot** is powered and reachable over Zenoh (or plan to use a local webcam — see below).

### 4. Run the robot

```bash
uv run python main.py
```

By default this launches the **ready stage**. You'll see:

```
[Ready] Listening — speak to Walkie. Ctrl+C to exit.
```

Now **speak to Walkie** through the mic. Each utterance is transcribed, handed to the agent, and answered with spoken audio. Press **Ctrl+C** to shut down.

### 5. Run without a microphone (typing mode)

Handy for development without audio hardware:

```bash
DISABLE_LISTENING=1 uv run python main.py
```

You'll get an `Enter your instruction:` prompt — type and press Enter to drive the agent.

---

## The legacy explore stage (object memory in `chroma_db`)

`main.py` currently runs the legacy explore stage **first** — driving the detection loop into the older `WalkieVectorDB` (`chroma_db/`). On startup you'll see:

```
[Explore] Drive the robot around. Press Enter when done.
```

Drive around to populate it, or just **press Enter** to skip straight to the ready (commanding) stage. To stop running it at all, comment out the Stage 1 block in `main.py`:

```python
# ── Stage 1: Explore ──
# ctx.stage = "explore"
# run_explore_stage(walkieAI, walkie, db)
```

> For semantic "where is the X?" lookups, prefer the **CLIP scene memory** below (built with `tools/scene_explore`) — it's what `find_object_from_memory` uses whenever `/image-embed` is available.

---

## CLIP scene memory (long-term semantic search)

During the **ready** stage the app also runs an always-on **scene-perception loop** that builds a semantic, spatial memory of what the robot sees. It is wired to walkie-ai-server's CLIP service:

```
camera → object_detection → bboxes_to_positions (3D) → image_caption + CLIP embed → ChromaDB (chroma_db_scene/)
                                                                          (image_embed on walkie-ai-server)
```

Once it's populated, `find_object_from_memory` answers "where is the X?" via CLIP semantic search (`SceneStore.semantic_query`) and returns map-frame `(x, y, z)` coordinates the actuator can navigate to. It runs **alongside** the `perception.json` live snapshot — they serve different purposes (long-term catalogue vs. current view).

**This depends on the server's `/image-embed` route.** On startup the app probes it once:

- Route available → `[scene] CLIP scene memory ON (dim=…, N existing record(s))`, loop starts.
- Route unavailable (it's commented out on walkie-ai-server by default) → `[scene] image-embed unavailable …; CLIP scene perception OFF`, the loop is skipped and `find_object_from_memory` falls back to the legacy `chroma_db/` from the explore stage. **The rest of the app runs normally either way.**

To turn the CLIP route on, the server team uncomments `app.register_blueprint(image_embed.bp)` in `walkie-ai-server` and redeploys. No changes are needed on this side.

Tuning knobs (all optional, see `.env.example`): `SCENE_PERCEPTION_ENABLED`, `SCENE_CHROMA_DIR`, `SCENE_FRAMES_DIR`, `SCENE_PERCEPTION_INTERVAL_SEC`, `SCENE_MIN_CONF`, `SCENE_CAPTION_PER_OBJECT`, `SCENE_FRAME_REFRESH_ON_UPDATE`.

### Building / rebuilding it: `tools/scene_explore`

The ready stage fills the scene memory passively while you talk to the robot. To build it **deliberately** — wipe it clean and drive around just to collect — use the standalone helper:

```bash
uv run python -m tools.scene_explore         # asks before wiping, then collects
uv run python -m tools.scene_explore -y      # skip the wipe confirmation (fast iteration)
uv run python -m tools.scene_explore --keep   # don't wipe; add to what's already there
uv run python -m tools.scene_explore --reset-only  # just wipe and exit
```

It deletes `SCENE_CHROMA_DIR` + `SCENE_FRAMES_DIR`, runs the same detect → lift → embed → store loop (no agent, no mic), and stops when you **press Enter**. Pruning stays off here — an explore run keeps everything it sees. Needs `/image-embed` up, same as the ready-stage loop. Its logs are at INFO so you can watch records land (`main.py` keeps them quiet — see [Logs / verbosity](#logs--verbosity)).

### Eviction (keeping it fresh)

In the **ready** stage the loop periodically prunes objects it no longer sees, so things you physically move away stop lingering in the store (and the viewer). It's spatially gated to the robot's current vicinity, so objects in rooms it hasn't revisited aren't wrongly deleted while it roams; thumbnails also refresh on each re-sighting. Tune with `SCENE_PRUNE_TTL_SEC`, `SCENE_PRUNE_RADIUS_M`, `SCENE_PRUNE_INTERVAL_SEC`, `SCENE_PRUNE_MAX_RECORDS` (see `.env.example`).

---

## Inspecting the vector DBs (Chroma viewer)

A **read-only** web UI to browse everything the robot has stored — the explore-stage `objects` (and older `people` / `scenes`) collections in `chroma_db/`, plus the CLIP `scene_entries` memory in `chroma_db_scene/`:

```bash
uv run python -m tools.chroma_viewer            # http://localhost:8500
uv run python -m tools.chroma_viewer --dirs chroma_db,chroma_db_scene --port 8500
```

It enumerates every collection in each directory and renders rows from whatever metadata they carry (so it works for any collection, not just the ones above). The UI gives you:

- A persistent **sidebar** of stores → collections (with live count badges) and a **light/dark theme** toggle.
- **Sortable columns**, **class-filter chips**, **colored class badges**, and inline **confidence/distance bars**.
- A top-down **position map** (SVG scatter of each record's `x`/`y` in the map frame, colored by class, with the robot origin marked) — click a point to open that record.
- **Frame thumbnails** with click-to-zoom **lightbox**; per record, the full metadata, document, archived JPEG, and an **embedding sparkline** + stats (dim / L2 norm).
- **Search**: substring by default; switch the dropdown to **semantic** for a CLIP/vector query (best-effort — falls back to substring with a warning if `walkie-ai-server` is down).

**Live updates:** the header has an **auto-refresh** dropdown (off / 2s / 5s / 10s / 30s with a countdown, remembered per-browser; initial value is `CHROMA_VIEWER_REFRESH_SEC`, default 5s). It refreshes by swapping just the content area — so your scroll position, theme, and search focus are preserved (no jarring full reload) — and pauses while you're typing. Browse tables, counts, and substring search reflect the robot's latest writes, so you can watch the DB fill in real time. One caveat: **semantic (vector) search** results are loaded into the viewer's memory at startup and only refresh when you **restart** the viewer; browse and substring search are always live.

It only ever reads, so it's safe to run while the robot is writing. Config: `CHROMA_VIEWER_DIRS`, `CHROMA_VIEWER_PORT`, `CHROMA_VIEWER_REFRESH_SEC`, `SCENE_FRAMES_DIR` (see `.env.example`).

---

## Standalone client tests (manual demos)

These open a **local webcam** and require `walkie-ai-server` running. They are visual smoke tests, not pytest tests:

```bash
uv run python test_object_detection.py   # boxes + labels live from webcam
uv run python test_captioning.py         # image captioning
uv run python test_pose_estimation.py    # human pose keypoints
```

Press `q` in the OpenCV window to quit.

---

## Running the test suite

A real pytest suite lives under `tests/` (mostly the perception subsystem):

```bash
uv run pytest                 # all tests
uv run pytest tests/perception -v
```

> The `test_*.py` files at the repo **root** are the manual webcam demos above — not part of the pytest run (`testpaths` is `["tests"]`).

---

## Project layout

```
main.py                  Entry point: builds clients, picks the stage, runs the loop.
agents/
  core/                  Shared agent factory, middleware stack, RobotContext, tool decorators.
  walkie_agent/          Main orchestrator agent (thread_id="main").
  actuator_agent/        Movement + arm tools.
  vision_agent/          Detection / captioning / pose / memory-lookup tools.
client/                  HTTP client to walkie-ai-server (stt, tts, detection, pose, caption, embed).
interfaces/
  walkie_interface.py    Composes hardware sub-clients (nav/arm/status/tools + camera/mic/speaker).
  devices/               Local camera, microphone, speaker wrappers.
services/
  explore.py             Explore-stage background service (legacy object memory).
  perception.py          Ready-stage snapshot writer.
  scene_perception.py    Ready-stage CLIP scene-perception loop (thread adapter).
perception/              Scene store, dedup, prune, async loop, embedders, pipeline.
db/walkie_db.py          WalkieVectorDB (ChromaDB wrapper) for object memory.
tools/chroma_viewer.py   Read-only web UI to inspect the ChromaDB stores.
tools/scene_explore.py   Reset + collect into the CLIP scene store (no agent/mic).
docs/                    Scene perception design docs (EN + TH).
tests/                   pytest suite (perception).
```

---

## Logs / verbosity

Perception emits a line per tick plus a `scene.dedup` line per detection — handy when watching collection, but they bury your prompt when commanding the robot. So the default differs per entrypoint:

| Command | Perception logs | Why |
|---|---|---|
| `main.py` | **WARNING** (quiet) | you're typing or speaking commands |
| `tools.scene_explore` | **INFO** (verbose) | you're watching it collect |

Override with `WALKIE_LOG_LEVEL` — uncomment it in `.env` to force one level everywhere, or set it inline for a single run: `WALKIE_LOG_LEVEL=INFO uv run python main.py`.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `WARNING: OPENROUTER_API_KEY not set` and agent errors | Fill `OPENROUTER_API_KEY` in `.env`. |
| Perception logs flooding the prompt | Expected at INFO. `main.py` defaults to WARNING; if it's noisy, comment out / unset `WALKIE_LOG_LEVEL` in `.env` (see [Logs / verbosity](#logs--verbosity)). |
| Connection errors to vision/STT/TTS | `walkie-ai-server` not running or wrong `WALKIE_AI_BASE_URL`. |
| `PortAudio`/`pyaudio` build or device errors | Install PortAudio (`sudo apt install portaudio19-dev`), or run with `DISABLE_LISTENING=1`. |
| Robot/Zenoh connection fails | Check the robot is up and `ROBOT_IP`/`ZENOH_PORT` in `main.py` are correct. |
| `find_object_from_memory` returns nothing | The DB is empty — build it first with `uv run python -m tools.scene_explore` (CLIP memory) or the legacy explore stage. |
| Walkie "responds" but says nothing aloud | Expected unless the agent calls `speak` — the no-plain-text contract. |

---

## More context

See [`CLAUDE.md`](./CLAUDE.md) for the deeper architecture notes (middleware stack, tool parallelism, cross-agent state, bbox conventions) and [`docs/`](./docs) for the scene-perception design.
