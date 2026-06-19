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
  - [~] `follow` — reuses HRI `follow_person` + `select_largest_person` (tracks
        whoever is in front; GPSR enrolls nobody). "follow me to X" now ends the
        moment the robot reaches X via a `tracking.ArrivalStopper` (returns
        'stopped'), not on `HRI_FOLLOW_TIMEOUT_SEC`. Needs perception/robot.
  - [~] `count` — navigate + detect/pose + `len()`; needs perception/robot.
  - [~] `greet` — `find_person` + spoken greeting; needs perception/robot.
  - [~] `get_person_info` — pose/gesture keypoints, clothing caption, name-by-ask.
  - [~] `get_object_property` — world-model category, else caption/measure.
  - [~] `guide` — lead a person to a destination (drive to `from` → confirm/face
        the person → lead to `to` → announce arrival). Confirming the person arrived
        needs **mid-route re-acquire** (the robot leads with its back to them, so a
        forward arrival frame can't see a trailing follower) — still open; needs robot.
  - [ ] `pick` / `place` / `deliver` — **gated off** (`GPSR_ENABLE_MANIPULATION=0`)
        until the arm is calibrated; promote Restaurant's grasp (`tasks/manipulation.py`).
        Falls through to Tier-2.
- [~] **Interleaved Task Bonus** (200) — `GPSR_INTERLEAVE` merges all planned commands
      into one **room-batched** order (`schedule.py`), executed with a shared nav cache
      so each room is visited once (the "reduce movements" the bonus rewards). Serial is
      the default + the fallback. Deterministic (not an LLM scheduler) → reliable +
      offline-tested; needs real poses + on-robot validation.

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
- [x] Shared follow/guide tracking (`tracking.py`): `ArrivalStopper` (follow ends on
      arrival — wired) + `companion_present` (building block for guide's mid-route
      re-acquire — not yet wired). Pure bits offline-tested; poll thread is robot-side.
- [x] Interleave scheduler (`schedule.py`) + per-command-isolated interleaved executor.
- [x] Pose-survey tool (`tools/teach_poses.py`): drive-and-capture poses into
      `world.toml`; the in-place TOML writer is pure + offline-tested.
- [x] Offline test suite: 88 GPSR tests (`tests/test_gpsr_*`) + coverage/split LLM gates.

## TODO (next, roughly in priority)

- [ ] **`guide` mid-route re-acquire** — pause/re-acquire if the guided person falls
      behind *during* the lead (needs interruptible/segmented nav; today it confirms
      only at arrival via `companion_present`).
- [ ] **Survey the real arena poses** (announced ~2 h before the test) — drive the
      robot to each place and capture with `python -m tasks.GPSR.tools.teach_poses`
      (writes `world.toml` in place, preserving fields), then paste the printed
      `GPSR_INSTRUCTION_POINT_POSE` into `config.toml`.
- [ ] **Manipulation** (`pick`/`place`/`deliver`) once the arm is calibrated.
- [ ] **Interleave on-robot tuning** — the room-batching lands the bonus's
      movement-reduction; with real poses, consider distance-aware ordering within a
      room and verify the wall-clock saving on the robot.

## How to run / verify

```bash
# Pure offline tests (no robot, no server):
uv run pytest tests/ -k "gpsr and not coverage"

# Parser coverage gate (needs OPENROUTER_API_KEY; ~2.5 min, real LLM):
uv run pytest tests/test_gpsr_coverage.py -s

# No-robot parser dry run (type a command, read back the spoken plan):
uv run python -m tasks.GPSR.parse

# Survey the arena (~2 h before the test): drive the robot to each place, Enter to capture:
uv run python -m tasks.GPSR.tools.teach_poses        # only un-surveyed places (--all for every)

# On the robot (needs walkie-ai-server; set real poses first):
DISABLE_LISTENING=1 uv run python -m tasks.GPSR.run
```
