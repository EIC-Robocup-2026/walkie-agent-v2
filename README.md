# walkie-agent-v2

The on-robot brain for **Walkie** — a 4th-gen omnidirectional robot from Chulalongkorn University's EIC team.

This repo orchestrates a LangChain/LangGraph **multi-agent system** over a real robot body (movement, arm, camera, mic, speaker). It is a thin local process — the heavy perception/speech models live in a separate **`walkie-ai-server`**. It talks to:

- **`walkie-sdk`** (git dependency) → hardware: navigation, arm, camera over Zenoh.
- **`walkie-ai-server`** (HTTP, at `WALKIE_AI_BASE_URL`) → model inference: STT, TTS, object detection, image captioning, pose estimation, image/text embeddings, face & appearance re-ID.
- **OpenRouter** (HTTP) → the LLM that drives the agents.

```
┌──────────────────┐      Zenoh        ┌──────────────────┐
│  walkie-agent-v2 │ ───────────────►  │ robot body       │  (nav, arm, camera, mic, speaker)
│  (this repo)     │                   └──────────────────┘
│                  │      HTTP         ┌──────────────────┐
│  LangGraph       │ ───────────────►  │ walkie-ai-server │  (STT/TTS/vision/embeddings)
│  agent stack     │                   └──────────────────┘
│  + walkie_graphs │      HTTP         ┌──────────────────┐
│  perception loop │ ───────────────►  │ OpenRouter       │  (the LLM brain)
└──────────────────┘                   └──────────────────┘
```

---

## Quick start

```bash
uv sync                                   # one-time: deps (resolves walkie-sdk from git, Python 3.12)
cp .env.example .env                      # one-time: then set OPENROUTER_API_KEY
uv run python main.py                      # run the robot — ready for commands immediately
```

- `.env` holds **only** secrets/endpoints (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`). All tuning lives in version-controlled TOML — no edit needed to start.
- Needs **`walkie-ai-server`** reachable at `WALKIE_AI_BASE_URL` (default in `config.toml`) for STT/TTS/detection/caption/embeddings.
- No microphone? Type prompts instead: `DISABLE_LISTENING=1 uv run python main.py`.
- Optional extras: `uv sync --extra graphs` adds [Rerun](https://rerun.io) for live 3D scene-graph visualization (imported lazily; everything else runs without it).

The unified launcher wraps the common operations:

```bash
./run.sh                 # start the agent (same as: uv run python main.py)
./run.sh reset           # wipe the walkie_graphs store for a clean slate
./run.sh fresh           # reset, then start
./run.sh help            # usage
```

---

## How it works (30-second tour)

`main.py` brings the robot up **ready to take commands immediately** — there's no explore stage and nothing to press Enter for. From the first second:

- A background **`walkie_graphs` perception loop** continuously builds a 3D scene graph of what the camera sees — capture RGB-D → masked open-vocabulary detection → lift each mask to a world-frame 3D point → register (ICP) → caption + embed → fuse into the spatial-memory store. It also writes a live snapshot to `perception.json` each tick for the agents to read.
- The agent listens to the mic (STT), runs the **Walkie agent** on each utterance, and speaks replies back (TTS).

The agent stack is **four agents** built from one factory (`agents/core/agent.py`):

- **Walkie main** — user-facing orchestrator; delegates to the sub-agents.
- **Actuator** — movement + arm.
- **Vision** — *live camera only*: detection, captioning, pose.
- **Database** — long-term spatial memory over the `walkie_graphs` scene graph ("where have I seen X / what's near here?").

> The agent only "talks" by calling the `speak` tool. A plain-text model reply with no tool call ends the turn **silently** — by design.

See **[`CLAUDE.md`](./CLAUDE.md)** for the authoritative architecture (middleware stack, tool parallelism, cross-agent state, the full tool list per agent) and **[`docs/WALKIE_GRAPHS.md`](./docs/WALKIE_GRAPHS.md)** for the perception-pipeline deep-dive.

---

## RoboCup challenge tasks (`tasks/`)

Scripted challenge runs (their own state machine over the robot, separate from the conversational agent) live under `tasks/`:

```bash
uv run python -m tasks.HRI.run            # 5.1 HRI / Receptionist  (reference implementation)
uv run python -m tasks.PickAndPlace.run   # 5.2 Pick and Place      (scaffold)
uv run python -m tasks.GPSR.run           # 5.3 GPSR                (scaffold — delegates to the agent stack)
uv run python -m tasks.Laundry.run        # 5.4 Doing Laundry       (scaffold)
uv run python -m tasks.Restaurant.run     # 5.5 Restaurant          (scaffold)
DISABLE_LISTENING=1 uv run python -m tasks.HRI.run   # type instead of speak (any task)
```

A task is an ordered list of `SubTask`s over a shared `TaskContext` (`tasks/base.py`); see `tasks/HRI/` for the reference implementation (greet & learn guests, offer a seat, introduce guests) and `tasks/HRI/config.toml` for its tuning (waypoints, host name, seat classes, …).

The other four challenges (`tasks/PickAndPlace`, `tasks/GPSR`, `tasks/Laundry`, `tasks/Restaurant`) are **placeholder scaffolds** copying the HRI shape: each lays out its rulebook flow as named `SubTask` steps with honest TODO stubs for the perception/manipulation that doesn't exist yet (degrading like `HRI.FollowHostAndDropBag`, never crashing), plus a `prompts.py`, `config.toml`, and (where useful) a `skills.py`. The per-challenge rulebook excerpts they implement live in [`docs/`](docs/) — one PDF per challenge (`docs/HRI.pdf`, `docs/PickAndPlace.pdf`, `docs/GPSR.pdf`, `docs/Laundry.pdf`, `docs/Restaurant.pdf`), cut from the RoboCup@Home 2026 rulebook chapter 5.

---

## Configuration

Tuning knobs live in **TOML** (version-controlled); secrets/endpoints in **`.env`** (gitignored). The TOML keys *are* the exact env-var names the code reads.

| File | Scope |
|------|-------|
| `config.toml` | App-wide: LLM (`WALKIE_MODEL`), transport, `WALKIE_AI_BASE_URL`, runtime toggles. |
| `services/walkie_graphs/config.toml` | The perception pipeline: detection classes, ICP/fusion thresholds, maintenance cadences, storage paths, Rerun viz. |
| `tasks/HRI/config.toml` | The HRI task: waypoints, host, seat detection, camera FOV. Loaded **before** the app config so task values win. |

**Precedence:** shell env **>** `.env` **>** task `config.toml` **>** app `config.toml` **>** module `config.toml` **>** code default. `walkie_config.py::load_config()` `setdefault`s every key, so the code keeps reading everything via `os.getenv(NAME, default)`.

---

## Maintenance tools

```bash
# Wipe the walkie_graphs store (chroma + point clouds + captures + background + thumbs).
# Run with the robot stopped — ChromaDB's persistent client is single-process.
uv run python -m services.walkie_graphs.tools.reset       # asks for confirmation
uv run python -m services.walkie_graphs.tools.reset -y     # no confirmation   (or: ./run.sh reset)

# Check Open3D GPU / ICP support.
uv run python -m services.walkie_graphs.tools.check_gpu
```

---

## Tests

```bash
uv run pytest                 # the real suite (pyproject testpaths = ["tests"])
```

Interactive demos that need live hardware / the AI server live in `manual_tests/` (guarded by `__main__`, deliberately **outside** `testpaths` so pytest never collects them):

```bash
uv run python -m manual_tests.test_robot_object_detection   # robot camera + detection
uv run python -m manual_tests.test_captioning               # image captioning
uv run python -m manual_tests.test_pose_estimation          # human pose keypoints
uv run python -m manual_tests.test_graphs_live              # live walkie_graphs ingest + Rerun viz
```

---

## Project layout

```
main.py                  Entry point: builds clients, runs the ready loop.
run.sh                   Unified launcher (start / reset / fresh).
walkie_config.py         Loads config.toml layers via os.environ.setdefault.
config.toml              App-wide tuning knobs.
agents/
  core/                  Shared agent factory, middleware stack, RobotContext, tool decorators.
  walkie_agent/          Main orchestrator agent (thread_id="main").
  actuator_agent/        Movement + arm tools.
  vision_agent/          Live-camera tools: detection / captioning / pose.
  database_agent/        Long-term spatial-memory tools over the scene graph.
client/                  HTTP client to walkie-ai-server (stt, tts, detection, pose, caption, embed, face, appearance).
interfaces/
  walkie_interface.py    Composes hardware sub-clients (nav/arm/status/tools + camera/mic/speaker).
  devices/               Local camera, microphone, speaker wrappers.
services/
  walkie_graphs/         The 3D scene-graph perception pipeline + store + config.toml + tools/.
perception/              Stores layer: shared ChromaDB plumbing (vector_db) + PeopleStore (face/attire re-ID).
tasks/                   Scripted RoboCup challenges (base/common + HRI/).
tests/                   pytest suite (mostly walkie_graphs).
manual_tests/            Interactive hardware/server demos (run via python -m manual_tests.*).
docs/                    Long-form docs (WALKIE_GRAPHS.md pipeline deep-dive, RERUN.md, HRI.pdf).
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `OPENROUTER_API_KEY not set` and agent errors | Fill `OPENROUTER_API_KEY` in `.env`. |
| Connection errors to vision/STT/TTS | `walkie-ai-server` not running, or wrong `WALKIE_AI_BASE_URL`. |
| `PortAudio`/`pyaudio` device errors | Install PortAudio (`sudo apt install portaudio19-dev`), or run with `DISABLE_LISTENING=1`. |
| Walkie "responds" but says nothing aloud | Expected unless the agent calls `speak` — the no-plain-text contract. |
| `walkie_graphs` Rerun viewer times out from another computer | The robot's host firewall is dropping the ports. Open both: `sudo ufw allow <WALKIE_GRAPHS_RERUN_WEB_PORT>/tcp && sudo ufw allow <WALKIE_GRAPHS_RERUN_GRPC_PORT>/tcp`. Launch with `WALKIE_GRAPHS_VIZ=rerun WALKIE_GRAPHS_RERUN_SERVE=1` (without it you get a local-only native window). A *connection refused* (instant, not a timeout) means it isn't serving — check the startup log. |

For deeper architecture and conventions, see **[`CLAUDE.md`](./CLAUDE.md)**.
