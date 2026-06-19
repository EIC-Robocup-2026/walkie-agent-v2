# GPSR — implementation checklist (rulebook 5.3)

Status legend: `[x]` done / real · `[~]` real logic but **needs on-robot validation
or real poses** · `[ ]` stub / Tier-2 fallback / not implemented.

> Architecture: two-tier hybrid (`docs/GPSR_DESIGN.md`). Each command is parsed into
> a typed `Plan`, the plan is **spoken** (scores the 300), then each step dispatches
> to a deterministic Tier-1 skill (`skills.py`), falling back to the agent stack
> (Tier-2) for ungrounded / gated / missing primitives.

## Scored actions (§5.3 score sheet — total 1490)

- [x] **Understand the spoken command** (3×80 = 240) — LLM extract → typed plan
      (`parse.py`). Offline coverage gate `tests/test_gpsr_coverage.py` = **100%**
      (39/39) on the real LLM.
- [x] **Demonstrate a plan has been generated** (3×100 = 300) — deterministic
      `plan.render_plan_speech`, spoken by `subtasks.ReceiveAndPlanCommands`.
- Solve the three commands (3×250 = 750) — per primitive:
  - [x] `say` / answer — LLM grounded with config identity + live clock (`say`).
  - [x] `find_person` — match by **gesture/pose** (keypoints), **clothing**
        (caption + LLM pick), **name** (best-effort, no enrollment). Shared by `greet`.
  - [~] `navigate` — world-model pose + `ctx.goto`; **needs real poses** (`world.toml`
        is all `[0,0,0]`) + nav stack.
  - [~] `find_object` — `walkie_graphs` memory + live re-detect; needs perception/robot.
  - [~] `follow` — **reuses HRI `follow_person` + `select_largest_person`** (landed via
        the main merge). Tracks whoever is in front (GPSR enrolls nobody). **No
        destination-arrival stopper yet** → "follow me to X" is bounded by
        `HRI_FOLLOW_TIMEOUT_SEC`, not by reaching X; arrival line is best-effort.
  - [~] `count` — navigate + detect/pose + `len()`; needs perception/robot.
  - [~] `greet` — `find_person` + spoken greeting; needs perception/robot.
  - [~] `get_person_info` — pose/gesture keypoints, clothing caption, name-by-ask.
  - [~] `get_object_property` — world-model category, else caption/measure.
  - [~] `guide` — lead a person to a destination (drive to `from` → confirm/face
        the person → lead to `to` → arrival announce). **No active follow-back
        tracking** yet (doesn't pause if they fall behind); needs perception/robot.
  - [ ] `pick` / `place` / `deliver` — **gated off** (`GPSR_ENABLE_MANIPULATION=0`)
        until the arm is calibrated; promote Restaurant's grasp (`tasks/manipulation.py`).
        Falls through to Tier-2.
- [ ] **Interleaved Task Bonus** (200) — serial MVP only; the all-3-at-once scheduler
      is the last phase (`docs/GPSR_DESIGN.md` §5.5).

## Avoiding penalties

- [x] **Minimize rephrasings + recovery** (rephrasing 6×−30, custom op 3×−20) — re-ask
      only on an empty parse (§5.2), bounded by `GPSR_MAX_REPHRASINGS`; then **request a
      custom operator** (`GPSR_USE_CUSTOM_OPERATOR`) before giving up, always leaving the
      command list set so the robot keeps "attending"
      (`ReceiveAndPlanCommands._receive_commands`, `tests/test_gpsr_recovery.py`).
- [x] **No bypassing STT** (3×−50) — commands come through the mic/STT path.
- [x] **Attending** — the fixed envelope always returns to the instruction point.

## Implementation status (branch `feat/GPSR`)

- [x] Fixed envelope: `GoToInstructionPoint → ReceiveAndPlanCommands → ExecuteCommands
      → ReturnToInstructionPoint` (`subtasks.py`).
- [x] Parser + grounding for all 13 primitives (`parse.py`, world-vocab injected).
- [x] World model loader, alias/plural-tolerant (`world.py` / `world.toml`).
- [x] Pure keypoint gesture heuristics (`gestures.py`, unit-tested).
- [x] Two-tier dispatcher + partial-scoring status aggregation (`dispatch.py` / `plan.py`).
- [x] Ported to the unified image API (`walkieAI.image.*`) after the main merge.
- [x] Shared helpers imported from the **global `tasks.skills` package** (geometry /
      lift / navigation / people), per the skills-refactor policy — not via
      `tasks.HRI.skills`.
- [x] Offline test suite: 66 GPSR tests (`tests/test_gpsr_*`) + coverage gate.

## TODO (next, roughly in priority)

- [ ] **`follow` destination stopper** — end the loop on arrival at `to`, not on timeout
      (a pose-watching stopper for `follow_person`); then on-robot validate.
- [ ] **`guide` follow-back tracking** — pause/re-acquire if the guided person falls
      behind (periodic pose check between nav segments); currently leads open-loop.
- [ ] **Real arena poses** in `world.toml` (announced ~2 h before the test).
- [ ] **Manipulation** (`pick`/`place`/`deliver`) once the arm is calibrated.
- [ ] **Interleave scheduler** (bonus 200), last.

## How to run / verify

```bash
# Pure offline tests (no robot, no server):
uv run pytest tests/ -k "gpsr and not coverage"

# Parser coverage gate (needs OPENROUTER_API_KEY; ~2.5 min, real LLM):
uv run pytest tests/test_gpsr_coverage.py -s

# No-robot parser dry run (type a command, read back the spoken plan):
uv run python -m tasks.GPSR.parse

# On the robot (needs walkie-ai-server; set real poses first):
DISABLE_LISTENING=1 uv run python -m tasks.GPSR.run
```
