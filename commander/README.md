# 🤖 Walkie Task Commander

A small LAN web UI for **launching, monitoring, and scoring the RoboCup@Home
challenges** in `walkie-agent-v2` — Run/Stop each challenge from any device on the
network, watch its live log, type commands into its stdin, and read the scorecard
it leaves behind.

It's the task-running counterpart to
[`walkie-commander`](https://github.com/EIC-Robocup-2026/walkie-commander) (generic
shell commands): same NiceGUI + asyncio process-management idea, but it already
knows the five challenges and how each is launched and scored.

> **Self-contained by design.** This lives *inside* the repo but is its own
> sub-project with its own venv and lockfile (depends only on NiceGUI). It does
> **not** import the agent code and does **not** touch the root
> `pyproject.toml` / `uv.lock` / `main.py` / any task file — so `main` can keep
> moving underneath it. It just shells out to the challenge entrypoints.

## Run

```bash
cd commander
uv run python main.py            # → http://localhost:8083  (and the LAN URL)
```

Start it from a shell where you'd normally run `./run.sh` — challenge subprocesses
inherit this process's environment (on the robot: a ROS-sourced shell).

## What it does

- **Discovers** every `tasks/*/run.py` (GPSR, Restaurant, Pick&Place, Laundry, HRI)
  — the same rule `run.sh` uses, so a new challenge shows up automatically.
- **Launches** `<repo>/.venv/bin/python -m tasks.<NAME>.run` from the repo root,
  in its own process group, with live stdout/stderr streamed to the browser.
  - Uses the repo's venv interpreter directly (not bare `uv run`) so a launch
    never re-resolves and dirties `uv.lock`. Falls back to `uv run --no-sync` only
    if `.venv` is missing.
- **DISABLE_LISTENING** toggle (on by default): the task reads typed prompts from
  stdin instead of the mic — type them into the log's input box. Turn it off to
  drive a run from the robot microphone.
- **Slices**: Restaurant (`RESTAURANT_SLICE`) and Pick&Place (`PNP_SLICE`) expose
  their bring-up stages (e.g. `pick`, `perceive`) as a dropdown. Plus a free-form
  `KEY=VALUE` env field for anything else.
- **Scorecard panel**: shows the JSON each task writes (forced into
  `walkie-runner-data/` so it never dirties the repo root).

## Known limitations (by design — keeps the agent code untouched)

- **Stop is graceful-first** — SIGINT, so the task's `finally:` runs (releases
  nav/arm/camera over zenoh and writes the scorecard), then escalates to
  SIGTERM → SIGKILL. A hard kill (SIGKILL) would leave the robot's zenoh handles
  unreleased and the next launch might fail to acquire the camera; the SIGINT-first
  path avoids that in the normal case.
- **The scorecard is the *last completed* run, not a live tally.** Tasks call
  `ScoreTracker.write()` only at the end (in `run.py`'s `finally`), so the panel
  updates when a run finishes or is gracefully stopped — not continuously.
- A full challenge only really runs on the **robot + walkie-ai-server** box; on a
  dev box without them the task will fail fast in its bring-up checks (still useful
  to confirm a challenge launches and streams output).

## Layout

```
commander/
  main.py                      # entry: uv run python main.py  → 0.0.0.0:8083
  pyproject.toml               # deps: nicegui only (own venv/lock)
  walkie_runner/
    registry.py                # discover tasks/*/run.py + metadata + launch resolution
    process_manager.py         # subprocess start/stream/stop (SIGINT→TERM→KILL), stdin
    pages.py                   # the NiceGUI page (one card per challenge)
  walkie-runner-data/          # scorecards the tasks write (gitignored)
```
