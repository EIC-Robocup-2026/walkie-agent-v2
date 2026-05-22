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

## How it works (30-second tour)

`main.py` runs the robot through two stages tracked on a process-wide `RobotContext`:

1. **`explore`** — drives around detecting objects, lifts them to 3D map coordinates, and stores confident multi-sighting objects into a vector DB (ChromaDB at `CHROMA_DIR`). *(Currently commented out in `main.py` — re-enable to rebuild the world catalogue.)*
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

## Re-running the explore stage (rebuild object memory)

The `ready` stage's `find_object_from_memory` is only useful once the vector DB is populated. To (re)build it, edit `main.py` and **uncomment the Stage 1 block**:

```python
# ── Stage 1: Explore ──
ctx.stage = "explore"
run_explore_stage(walkieAI, walkie, db)
```

Then `uv run python main.py`, drive the robot around, and **press Enter** when done. It prints how many confident objects were stored. Re-comment the block afterward to go back to the ready stage.

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
  explore.py             Explore-stage background service.
  perception.py          Ready-stage snapshot writer.
perception/              Scene store, dedup, async loop, embedders, pipeline.
db/walkie_db.py          WalkieVectorDB (ChromaDB wrapper) for object memory.
docs/                    Scene perception design docs (EN + TH).
tests/                   pytest suite (perception).
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `WARNING: OPENROUTER_API_KEY not set` and agent errors | Fill `OPENROUTER_API_KEY` in `.env`. |
| Connection errors to vision/STT/TTS | `walkie-ai-server` not running or wrong `WALKIE_AI_BASE_URL`. |
| `PortAudio`/`pyaudio` build or device errors | Install PortAudio (`sudo apt install portaudio19-dev`), or run with `DISABLE_LISTENING=1`. |
| Robot/Zenoh connection fails | Check the robot is up and `ROBOT_IP`/`ZENOH_PORT` in `main.py` are correct. |
| `find_object_from_memory` returns nothing | The vector DB is empty — run the explore stage first. |
| Walkie "responds" but says nothing aloud | Expected unless the agent calls `speak` — the no-plain-text contract. |

---

## More context

See [`CLAUDE.md`](./CLAUDE.md) for the deeper architecture notes (middleware stack, tool parallelism, cross-agent state, bbox conventions) and [`docs/`](./docs) for the scene-perception design.
