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
- [Rerun](https://rerun.io) (live 3D scene-graph visualization) ships as a core dependency; it's imported lazily, so everything else runs even if it's unavailable.

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

See **[`CLAUDE.md`](./CLAUDE.md)** for the authoritative architecture (middleware stack, tool parallelism, cross-agent state, the full tool list per agent) and **[`docs/WALKIE_WORLD.md`](./docs/WALKIE_WORLD.md)** for the perception-pipeline deep-dive.

---

## RoboCup challenge tasks (`tasks/`)

Scripted challenge runs (their own state machine over the robot, separate from the conversational agent) live under `tasks/`:

```bash
uv run python -m tasks.HRI.run            # 5.1 HRI / Receptionist
uv run python -m tasks.PickAndPlace.run   # 5.2 Pick and Place
uv run python -m tasks.GPSR.run           # 5.3 GPSR
uv run python -m tasks.Laundry.run        # 5.4 Doing Laundry
uv run python -m tasks.Restaurant.run     # 5.5 Restaurant
DISABLE_LISTENING=1 uv run python -m tasks.HRI.run   # type instead of speak (any task)
```

A task is an ordered list of `SubTask`s over a shared `TaskContext` (`tasks/base.py`); every non-critical step **degrades rather than crashes** (partial scoring is allowed, so a failed step logs and the run moves on).

**The arm is being brought up as a separate skill**, so the manipulation tasks gate every grasp/place behind an `*_ARM_CALIBRATED` flag (default off) and run + score their **non-arm budget** first — navigation, perception, and *communicating perception* to the referee (the rulebook scores recognizing an object / indicating a placement, no grasp required). Flip the flag once the arm lands and the manipulation budget unlocks with no flow rewrite. Most tasks also expose a step-by-step `*_SLICE` runner (e.g. `PNP_SLICE=perceive`, `RESTAURANT_SLICE=phase0`, `HRI_SLICE=greet`) for validating one phase at a time on the robot — no commenting steps in and out.

| Task | Status | Notes |
|------|--------|-------|
| **GPSR** (5.3) | mature | STT → parse → **speak a plan** → typed-plan dispatch, with an agent fallback; manipulation gated. Parser corpus-tested. |
| **Restaurant** (5.5) | Phase-0 serve | detect a waving customer → approach → take + confirm the order → relay to the barman; pick/serve gated (`RESTAURANT_ARM_CALIBRATED`), serial + batched loops, `RESTAURANT_SLICE` runner. |
| **Pick and Place** (5.2) | non-arm pipeline | navigate, recognize each object, indicate the correct placement (scores ~195 of 3515 with the arm gated); pick/place gated (`PNP_ARM_CALIBRATED`), `PNP_SLICE` runner. |
| **HRI** (5.1) | reference | face/appearance re-ID, seat detection, guest introductions, follow-host (the most-built flow); run via `HRI_SLICE` (`seats`/`greet`/`follow_host`/`full`, default `full`). Bag handover gated (`HRI_ENABLE_BAG`). |
| **Laundry** (5.4) | scaffold | almost-pure manipulation: the only non-arm line on the scoresheet is navigating to the laundry area. |

Each task carries a `prompts.py`, `config.toml`, and (where useful) a `skills.py`; the shared grasp/perception primitives live in `tasks/manipulation.py`. The per-challenge rulebook excerpts live in [`docs/`](docs/) — one PDF each (`docs/{HRI,PickAndPlace,GPSR,Laundry,Restaurant}.pdf`), cut from the RoboCup@Home 2026 rulebook chapter 5. Expected points per challenge are tracked in [`docs/SCORING.md`](docs/SCORING.md) — see [Scoring](#scoring).

---

## Scoring

[`docs/SCORING.md`](docs/SCORING.md) is a living worksheet estimating Walkie's expected points per challenge under partial scoring (low / expected / high capture %). The per-line points are **code-backed**: each challenge's rulebook scoresheet is encoded as a `ScoreSheet` in `tasks/<Challenge>/scoring.py`, the framework is `tasks/scoring.py`, and `tests/test_scoring.py` **reconciles every sheet against its official rulebook total**. The same `ScoreSheet` feeds both the planning estimate and a live runtime tally (`ScoreTracker`) — read the tally as *attempted/claimed* points, **not** referee-awarded. This makes the **non-arm budget** each task can score with the arm gated explicit:

| Challenge | Rulebook total | Non-arm budget (arm gated) |
|-----------|--------------:|---------------------------:|
| GPSR | 1490 | 740 fixed + draw-dependent solve |
| Restaurant | 2360 | 960 (Phase-0 serve) |
| Pick and Place | 3515 | 195 (recognize + indicate) |
| HRI | 1450 | 950 (gaze / seat / intro / follow) |
| Laundry | 4415 | 15 (navigate only) |

---

## Configuration

Tuning knobs live in **TOML** (version-controlled); secrets/endpoints in **`.env`** (gitignored). The TOML keys *are* the exact env-var names the code reads.

| File | Scope |
|------|-------|
| `config.toml` | App-wide: LLM (`WALKIE_MODEL`), transport, `WALKIE_AI_BASE_URL`, runtime toggles. |
| `services/realtime_explore/config.toml` | The perception pipeline: detection classes, ICP/fusion thresholds, maintenance cadences, storage paths, Rerun viz. |
| `tasks/<Challenge>/config.toml` | Per-task tuning: waypoints, gating flags (`*_ARM_CALIBRATED`), detector classes, slice/runner knobs. Loaded **before** the app config so task values win. |

**Precedence:** shell env **>** `.env` **>** task `config.toml` **>** app `config.toml` **>** module `config.toml` **>** code default. `walkie_config.py::load_config()` `setdefault`s every key, so the code keeps reading everything via `os.getenv(NAME, default)`.

---

## Maintenance tools

```bash
# Wipe the walkie_graphs store (chroma + point clouds + captures + background + thumbs).
# Run with the robot stopped — ChromaDB's persistent client is single-process.
uv run python -m services.realtime_explore.tools.reset       # asks for confirmation
uv run python -m services.realtime_explore.tools.reset -y     # no confirmation   (or: ./run.sh reset)

# Check Open3D GPU / ICP support.
uv run python -m services.realtime_explore.tools.check_gpu
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
uv run python -m manual_tests.test_object_segmentation      # open-vocab detect + segment masks
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
tasks/                   Scripted RoboCup challenges: base/common, manipulation, scoring + GPSR/HRI/PickAndPlace/Laundry/Restaurant (each with run/subtasks/skills/prompts/config/scoring).
tests/                   pytest suite (walkie_graphs, gpsr, hri, restaurant, manipulation, scoring, skills, …).
manual_tests/            Interactive hardware/server demos (run via python -m manual_tests.*).
docs/                    Long-form docs: WALKIE_WORLD.md, SCORING.md, GPSR_DESIGN.md, RESTAURANT_DESIGN.md, RERUN.md + per-challenge rulebook PDFs.
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `OPENROUTER_API_KEY not set` and agent errors | Fill `OPENROUTER_API_KEY` in `.env`. |
| Connection errors to vision/STT/TTS | `walkie-ai-server` not running, or wrong `WALKIE_AI_BASE_URL`. |
| `PortAudio`/`pyaudio` device errors | Install PortAudio (`sudo apt install portaudio19-dev`), or run with `DISABLE_LISTENING=1`. |
| Walkie "responds" but says nothing aloud | Expected unless the agent calls `speak` — the no-plain-text contract. |
| `walkie_graphs` Rerun viewer times out from another computer | The robot's host firewall is dropping the ports. Open both: `sudo ufw allow <WALKIE_EXPLORE_RERUN_WEB_PORT>/tcp && sudo ufw allow <WALKIE_EXPLORE_RERUN_GRPC_PORT>/tcp`. Launch with `WALKIE_EXPLORE_VIZ=rerun WALKIE_EXPLORE_RERUN_SERVE=1` (without it you get a local-only native window). A *connection refused* (instant, not a timeout) means it isn't serving — check the startup log. |

For deeper architecture and conventions, see **[`CLAUDE.md`](./CLAUDE.md)**.
