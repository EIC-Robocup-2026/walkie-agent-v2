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

## TL;DR — the three commands you'll actually run

```bash
uv sync                                  # one-time: install deps (incl. walkie-sdk)
cp .env.example .env                      # one-time: then set OPENROUTER_API_KEY + WALKIE_AI_BASE_URL
```

`.env` holds only secrets/endpoints; all tuning lives in **`config.toml`** (no edit needed to start). See [Configure your environment](#2-configure-your-environment).

Then, usually two things:

```bash
# Run the robot + agent — ready for commands immediately; the scene DB fills
# itself in the background. This also auto-starts the live DB viewer at :8500.
uv run python main.py                        # add DISABLE_LISTENING=1 to type instead of speaking
#    → open http://<ip-of-this-box>:8500 to watch the database fill in real time

# Optional: deliberately rebuild the catalogue offline (wipe, then just drive).
uv run python -m tools.scene_explore -y      # press Enter when you're done driving
```

`main.py` now brings the **DB viewer up in its own process**, so you don't run a second script — open `http://<ip-of-this-box>:8500` (it binds `0.0.0.0`, so teammates on the LAN can open it too). Because it shares the robot's live ChromaDB client, browsing is fully live *and* can't corrupt the store. Disable it with `CHROMA_VIEWER_AUTOSTART=0`; run it standalone (robot stopped) with `uv run python -m tools.chroma_viewer`. `main.py` and `scene_explore` need **`walkie-ai-server`** up at `WALKIE_AI_BASE_URL` (collection also uses its `/image-embed` route). Each command is explained in full below.

> Need a clean slate? `uv run python -m tools.reset_db --all` wipes both vector DBs.

---

## How it works (30-second tour)

`main.py` brings the robot up **ready to take commands immediately** — there's no explore stage and nothing to press Enter for. From the first second:

- A background **scene-perception loop** continuously builds and updates the CLIP scene memory (`chroma_db_scene`) from whatever the camera sees — see, remember, and re-see without any "drive around first" phase.
- A background service writes a live perception snapshot to `perception.json`.
- The agent listens to the mic (STT), runs the **Walkie agent** on each utterance, and speaks replies back (TTS) — so you can look at, update, and command the robot all at once.

(To *deliberately* rebuild the catalogue offline — wipe and just drive to collect — use the standalone [`tools/scene_explore`](#building--rebuilding-it-toolsscene_explore); it's optional now that the ready stage fills the DB on its own.)

The agent stack is four agents built from one factory:

- **Walkie main** — user-facing orchestrator; delegates to the sub-agents.
- **Actuator** — movement + arm (`move_absolute`, `move_relative`, `command_arm`, …).
- **Vision** — *live camera only*: `detect_objects_from_view`, `image_caption`, `detect_people_poses`, …
- **Database** — long-term spatial memory: `find_object` (caption-first), `objects_near`, `recently_seen`, `list_known_objects`. "Where have I seen X / what's near here?" → Database; "what's in front of me now?" → Vision.

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

Config is split in two:

- **`.env`** (gitignored) — secrets, endpoints, per-machine/runtime toggles only.
- **`config.toml`** (version-controlled) — all tuning knobs (perception/scene/explore/viewer/model). Edit here for behavior changes.

Precedence: a shell env var **>** `.env` **>** `config.toml` **>** the code default — so you can still override any tunable from `.env` or the shell for a one-off run.

```bash
cp .env.example .env
```

Open `.env` and at minimum set your LLM key:

```dotenv
OPENROUTER_API_KEY=sk-or-...
```

`.env` variables (the full set):

| Variable | Default | What it does |
|---|---|---|
| `OPENROUTER_API_KEY` | *(empty)* | **Required.** Agent calls fail without it. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | LLM endpoint. |
| `WALKIE_AI_BASE_URL` | `http://localhost:5000` | Where `walkie-ai-server` lives. |
| `WALKIE_ROS_PROTOCOL` / `WALKIE_ROS_PORT` | `rosbridge` / `9090` | Robot transport. |
| `DISABLE_LISTENING` | `0` | Set `1` to type prompts instead of using the mic. |

Tuning lives in **`config.toml`** — e.g. `[llm] WALKIE_MODEL`, `[scene] SCENE_PERCEPTION_INTERVAL_SEC`, `[scene.dedup] SCENE_DEDUP_RADIUS_M`, `[viewer] CHROMA_VIEWER_PORT`. Each key is the exact env-var name the code reads, grouped into tables for readability.

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

## The legacy object memory (`chroma_db`)

`main.py` **no longer runs an explore stage** — the robot is ready to take commands the instant it starts, and the CLIP scene memory (below) fills itself in the background. The older `WalkieVectorDB` (`chroma_db/`) is kept only as a *fallback* that `find_object_from_memory` reads when the CLIP `/image-embed` route is unavailable; nothing populates it automatically anymore.

To collect into it deliberately you can still drive the legacy `ExploreService` yourself (it lives in `services/explore.py`), but for "where is the X?" lookups prefer the **CLIP scene memory** below — it's what `find_object_from_memory` uses whenever `/image-embed` is available.

---

## CLIP scene memory (long-term semantic search)

During the **ready** stage the app also runs an always-on **scene-perception loop** that builds a semantic, spatial memory of what the robot sees. It is wired to walkie-ai-server's CLIP service:

```
camera → object_detection → bboxes_to_positions (3D) → image_caption + CLIP embed → ChromaDB (chroma_db_scene/)
                                                          (CLIP runs in-process by default; see Embedding backend)
```

Once it's populated, `find_object_from_memory` (and the **Database agent**) answer "where is the X?" by searching the stored **captions** first (`SceneStore.text_query`, text→text — "coffee mug" matches a record captioned "a white ceramic coffee mug"), falling back to CLIP image search (`semantic_query`) if the caption index has no hit. Either way it returns map-frame `(x, y, z)` coordinates the actuator can navigate to. It runs **alongside** the `perception.json` live snapshot — they serve different purposes (long-term catalogue vs. current view).

Internally the store keeps two ChromaDB collections under one id space: `scene_entries` (CLIP image embeddings, used for dedup) and `scene_captions` (CLIP text embeddings of the captions, used by `text_query`). The caption index is written automatically on new sightings; for data collected with an older build, set `SCENE_REINDEX_CAPTIONS=1` once (in `.env` or `config.toml`) to backfill it.

Objects whose 3D depth-lift fails — small/distant ones, or a whole crowded frame on timeout — are **dropped** by default, so only objects with a real per-object position enter the catalogue (stamping the robot's own pose instead would store *where the robot stood*, not where the object is, and navigating back there finds nothing). Set `SCENE_POSITION_FALLBACK_POSE=1` to re-enable the old robot-pose fallback once `get_3d_poses` is trustworthy. A separate sanity gate, `SCENE_MAX_LIFT_DISTANCE_M`, rejects lifted positions farther than N metres from the robot as sensor outliers. On a confident re-sighting that merges across a large distance (visual dedup), the position is **not** averaged into the empty space between the two observations — the higher-confidence one is kept.

### Embedding backend: local (default) or remote

CLIP embeddings can be produced **in-process** (`SCENE_EMBED_BACKEND=local`, the default) or by the **server** (`remote`):

- **local** — loads the same checkpoint (`openai/clip-vit-base-patch16`) in this process via `transformers`, GPU-accelerated (CUDA + fp16 auto). No dependency on walkie-ai-server's `/image-embed`, so a server hiccup can't break perception or lookups. Needs the optional extra: `uv sync --extra clip`. On an **RTX 5090** (Blackwell/sm_120) install a CUDA 12.8 torch build, e.g. `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`. First run downloads the model (~600 MB) and caches it.
  - Startup → `[scene] embedding backend: local (model=clip-vit-base-patch16)` then `[scene] CLIP scene memory ON (dim=…, N record(s))`.
  - Missing extra / load failure → `[scene] local CLIP unavailable …; CLIP scene perception OFF` and the app continues on the legacy fallback.
- **remote** — calls the server's `/image-embed` route (must be enabled there: uncomment `app.register_blueprint(image_embed.bp)` in `walkie-ai-server`). Unavailable → `[scene] image-embed unavailable …; CLIP scene perception OFF`.

Either backend records the same `model_name`, so they're interchangeable over one store. Even so, if the embed path is unavailable mid-run, "where is X?" lookups still degrade to a local keyword search (see below).

Tuning knobs (in `config.toml`, `[scene]` / `[scene.dedup]` / `[scene.position]` / `[scene.query]`): `SCENE_PERCEPTION_ENABLED`, `SCENE_EMBED_BACKEND`, `SCENE_CLIP_MODEL`, `SCENE_CLIP_DEVICE`, `SCENE_CLIP_FP16`, `SCENE_CHROMA_DIR`, `SCENE_FRAMES_DIR`, `SCENE_PERCEPTION_INTERVAL_SEC`, `SCENE_MIN_CONF`, `SCENE_CAPTION_PER_OBJECT`, `SCENE_FRAME_REFRESH_ON_UPDATE`, `SCENE_FRAME_CROP`, `SCENE_FRAME_CROP_MARGIN`, `SCENE_REINDEX_CAPTIONS`, `SCENE_DEDUP_RADIUS_M`, `SCENE_DEDUP_VISUAL_K`, `SCENE_POSITION_SOURCE`, `SCENE_POSITION_FALLBACK_POSE`, `SCENE_MAX_LIFT_DISTANCE_M`, `SCENE_QUERY_MIN_CONF`.

The stored document is **caption-led** (the detector class is often wrong, so it's kept out of the search text and only used as a fallback when there's no caption), and archived thumbnails are the **object crop** (bbox + `SCENE_FRAME_CROP_MARGIN` padding), not the whole frame. For "where is X?" the Database agent and `find_object_from_memory` accept `near_me=True` to restrict the search to the robot's current vicinity, and drop matches below `SCENE_QUERY_MIN_CONF` so a returned coordinate is always one the robot can actually be sent to. If the `/image-embed` server is down, lookups **fall back to a local keyword (word-overlap) search** over the stored captions so "find X" keeps working offline.

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

In the **ready** stage the loop periodically prunes objects it no longer sees, so things you physically move away stop lingering in the store (and the viewer). It's spatially gated to the robot's current vicinity, so objects in rooms it hasn't revisited aren't wrongly deleted while it roams; thumbnails also refresh on each re-sighting. Tune with `SCENE_PRUNE_TTL_SEC`, `SCENE_PRUNE_RADIUS_M`, `SCENE_PRUNE_INTERVAL_SEC`, `SCENE_PRUNE_MAX_RECORDS` (in `config.toml`, `[scene.prune]`).

---

## Inspecting the vector DBs (Chroma viewer)

A **read-only** web UI to browse everything the robot has stored — the legacy `objects` (and older `people` / `scenes`) collections in `chroma_db/`, plus the CLIP `scene_entries` memory in `chroma_db_scene/`.

**The usual way: it starts itself.** `python main.py` launches the viewer in-process (a daemon thread reusing the robot's own ChromaDB clients) and prints its URL. No second script, and — crucially — no risk to the store: it reads the *same* live index the robot writes, so there's only one HNSW index (a separate process opening the same dir would spin up a rival index and corrupt it). Turn it off with `CHROMA_VIEWER_AUTOSTART=0`; change where it binds with `CHROMA_VIEWER_HOST` / `CHROMA_VIEWER_PORT`.

**Running it standalone** (a separate process — only when the robot is **not** running):

```bash
uv run python -m tools.chroma_viewer            # http://localhost:8500
uv run python -m tools.chroma_viewer --dirs chroma_db,chroma_db_scene --port 8500
```

A standalone process can't share the robot's index, so it defaults to opening a **read-only snapshot copy** of each dir (taken once at startup) — safe to run anytime, but its data is **frozen at launch**; restart it to pick up new writes. `--live` (or `CHROMA_VIEWER_LIVE=1`) reads the real dirs in place instead, which is live but **only safe with the robot stopped** (concurrent writes from two processes corrupt the DB). For watching the DB grow during a run, use the in-process auto-start above.

**Sharing it on the LAN:** the viewer binds `0.0.0.0` by default, so anyone on the same network opens `http://<ip-of-the-box-running-it>:8500` — it runs on the machine holding the Chroma dirs (the one running `main.py`). To pin a different port use `--port` (or `CHROMA_VIEWER_PORT`).

It enumerates every collection in each directory and renders rows from whatever metadata they carry (so it works for any collection, not just the ones above). The UI gives you:

- A persistent **sidebar** of stores → collections (with live count badges) and a **light/dark theme** toggle.
- **Sortable columns**, **class-filter chips**, **colored class badges**, and inline **confidence/distance bars**.
- A top-down **position map** (SVG scatter of each record's `x`/`y` in the map frame, colored by class, with the robot origin marked) — click a point to open that record.
- **Frame thumbnails** with click-to-zoom **lightbox**; per record, the full metadata, document, archived JPEG, and an **embedding sparkline** + stats (dim / L2 norm). Frames are **downscaled server-side** (cached, keyed by `?w=`) so full-resolution camera captures render as small thumbnails in tables/galleries instead of failing to load — the lightbox still shows a crisp larger version.
- **Search**: substring by default; switch the dropdown to **semantic** for a CLIP/vector query (best-effort — falls back to substring with a warning if `walkie-ai-server` is down).

**Live updates:** the header has an **auto-refresh** dropdown (off / 2s / 5s / 10s / 30s with a countdown, remembered per-browser; initial value is `CHROMA_VIEWER_REFRESH_SEC`, default 5s). It refreshes by swapping just the content area — so your scroll position, theme, and search focus are preserved (no jarring full reload) — and pauses while you're typing. **When auto-started by `main.py`** (in-process), browse tables, counts, and substring search reflect the robot's latest writes, so you can watch the DB fill in real time; only **semantic (vector) search** lags — it's loaded into memory at startup and refreshes on viewer restart. **In standalone snapshot mode**, the auto-refresh re-renders but the underlying data is frozen at launch, so restart to see new rows.

Config (in `config.toml`, `[viewer]`): `CHROMA_VIEWER_AUTOSTART`, `CHROMA_VIEWER_HOST`, `CHROMA_VIEWER_DIRS`, `CHROMA_VIEWER_PORT`, `CHROMA_VIEWER_REFRESH_SEC`, `CHROMA_VIEWER_THUMB_CACHE`, plus `SCENE_FRAMES_DIR`.

---

## Standalone client tests (manual demos)

These open a **local webcam** and require `walkie-ai-server` running. They are visual smoke tests, not pytest tests. They live in `manual_tests/` — run them as modules so `from client import …` resolves:

```bash
uv run python -m manual_tests.test_object_detection   # boxes + labels live from webcam
uv run python -m manual_tests.test_captioning         # image captioning
uv run python -m manual_tests.test_pose_estimation    # human pose keypoints
```

Press `q` in the OpenCV window to quit.

---

## Running the test suite

A real pytest suite lives under `tests/` (mostly the perception subsystem):

```bash
uv run pytest                 # all tests
uv run pytest tests/perception -v
```

> The webcam demos live in `manual_tests/` (not `tests/`), so they're never collected by the pytest run (`testpaths` is `["tests"]`).

### Wiping a vector DB

The DBs are generated at runtime (and gitignored). To start fresh — run with the robot/viewer stopped:

```bash
uv run python -m tools.reset_db --object   # legacy object DB (chroma_db) + object_frames
uv run python -m tools.reset_db --scene    # CLIP scene memory (chroma_db_scene) + frames
uv run python -m tools.reset_db --all -y   # both, skip the confirmation
```

---

## Project layout

```
main.py                  Entry point: builds clients, runs the ready loop (no explore stage).
agents/
  core/                  Shared agent factory, middleware stack, RobotContext, tool decorators.
  walkie_agent/          Main orchestrator agent (thread_id="main").
  actuator_agent/        Movement + arm tools.
  vision_agent/          Live-camera tools: detection / captioning / pose.
  database_agent/        Long-term spatial-memory tools (find_object, objects_near, …).
client/                  HTTP client to walkie-ai-server (stt, tts, detection, pose, caption, embed).
config.toml              Tuning knobs (perception/scene/explore/viewer/model); loaded by walkie_config.py.
interfaces/
  walkie_interface.py    Composes hardware sub-clients (nav/arm/status/tools + camera/mic/speaker).
  devices/               Local camera, microphone, speaker wrappers.
services/
  explore.py             Legacy explore background service (no longer run by main.py; manual use only).
  perception.py          Ready-stage snapshot writer.
  scene_perception.py    Ready-stage CLIP scene-perception loop (thread adapter).
perception/              Scene store, dedup, prune, async loop, embedders, pipeline.
db/walkie_db.py          WalkieVectorDB (ChromaDB wrapper) for object memory.
tools/chroma_viewer.py   Read-only web UI to inspect the ChromaDB stores.
tools/scene_explore.py   Reset + collect into the CLIP scene store (no agent/mic).
tools/reset_db.py        Wipe the object and/or CLIP scene DBs for a clean slate.
docs/                    Scene perception design docs (EN + TH).
tests/                   pytest suite (perception).
manual_tests/            Interactive webcam/robot demos (run via `python -m manual_tests.*`).
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
| `find_object_from_memory` returns nothing | The DB hasn't filled yet — let the robot look around (the ready-stage loop builds it in the background), or collect deliberately with `uv run python -m tools.scene_explore`. Also check matches aren't all being dropped by `SCENE_QUERY_MIN_CONF`. |
| Walkie "responds" but says nothing aloud | Expected unless the agent calls `speak` — the no-plain-text contract. |

---

## More context

See [`CLAUDE.md`](./CLAUDE.md) for the deeper architecture notes (middleware stack, tool parallelism, cross-agent state, bbox conventions) and [`docs/`](./docs) for the scene-perception design.
