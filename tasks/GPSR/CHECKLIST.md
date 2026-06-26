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
      (39/39) on the real LLM. Grounding has a **fuzzy fallback** (`world._lookup`,
      `GPSR_GROUNDING_FUZZY_CUTOFF`) so a mis-heard noun ("kitchen tabel") still
      grounds instead of forfeiting the command — consulted only on an exact miss.
- [x] **Demonstrate a plan has been generated** (3×100 = 300) — deterministic
      `plan.render_plan_speech`, spoken by `subtasks.ReceiveAndPlanCommands`. Both
      coverage gates now **also assert the render is clean** for every complete
      parse (non-empty, no leaked raw token) so a parse-then-degenerate-render can't
      lose the 300 silently.
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
  - [~] `count` — persons: full-turn scan + spatial dedup. Objects: now re-shoots
        the placement `GPSR_COUNT_OBJ_FRAMES` times and reports the **per-frame
        median** (`_count_objects_stable`) so one flickery detector frame can't
        in/deflate the count. Pure median offline-tested; needs perception/robot.
  - [~] `greet` — `find_person` + spoken greeting; needs perception/robot.
  - [~] `get_person_info` — pose/gesture keypoints, clothing caption, name-by-ask.
  - [~] `get_object_property` — world-model category (deterministic), else caption.
        A **superlative size** query ("the biggest/smallest object on the X") now
        detects over the object vocabulary and picks the winner by image-bbox area
        (`_superlative_dir`/`_pick_by_size`, direction read from the raw clause) and
        names it, instead of describing an arbitrary box. Pure pick offline-tested.
  - [~] `guide` — lead a person to a destination (drive to `from` → confirm/face
        the person → lead to `to` → announce arrival). **Mid-route re-acquire** is
        now implemented (`GPSR_GUIDE_REACQUIRE`, default OFF): leads in capped hops
        (`tracking.segment_route`) and turns back between them to re-acquire a
        trailing follower (`tracking.companion_present`), prompting + waiting then
        leading on best-effort. Pure bits offline-tested; the turn/camera/wait loop
        needs on-robot validation before the flag is flipped on.
  - [~] `pick` / `place` — Tier-1 skills wired to the shared grasp system
        (`tasks/skills.pick_object` / `place_object`, the layer Restaurant drives),
        **gated** behind `GPSR_ENABLE_MANIPULATION` (0 default → Tier-2 fallback;
        1 → deterministic Tier-1). Needs on-robot validation before the flag flips.
  - [ ] `deliver` — stays Tier-2 even with manipulation on: a robot→human handover
        isn't a grasp-system primitive, so the agent stack drives the handoff (a
        prior `pick` step still grabs the object Tier-1).
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
      arrival — wired), `companion_present` + `segment_route`/`heading_between` (now
      wired into guide's mid-route re-acquire, gated by `GPSR_GUIDE_REACQUIRE`). Pure
      bits offline-tested; poll/turn/wait loops are robot-side.
- [x] Interleave scheduler (`schedule.py`) + per-command-isolated interleaved executor.
- [x] Pose-survey tool (`tools/teach_poses.py`): drive-and-capture poses into
      `world.toml`; the in-place TOML writer is pure + offline-tested.
- [x] Closed-door handling (shared `tasks.skills` door feature): arena entry's
      `request_open_door` auto-detects open/closed via depth; named navigation
      (`go_to_named`) routes through `go_to_through_door` so a closed door blocking a
      goal triggers an ask + retry (gated `GPSR_NAV_DOOR_RETRY`, default ON; depth
      thresholds + the false-ask risk are on-robot tuning).
- [x] Offline test suite: 144 GPSR tests (`tests/test_gpsr_*`) + coverage/generator
      LLM gates (the gates also assert render quality now).

## TODO (next, roughly in priority)

- [~] **`guide` mid-route re-acquire** — *implemented* (segmented lead + look-back,
      `GPSR_GUIDE_REACQUIRE`, default OFF); pure bits offline-tested. **On-robot:**
      validate the turn-back/companion-check loop and tune `GPSR_GUIDE_SEGMENT_M` vs.
      the 7-minute clock before flipping the flag on.
- [ ] **Survey the real arena poses** (announced ~2 h before the test) — drive the
      robot to each place and capture with `python -m tasks.GPSR.tools.teach_poses`
      (writes `world.toml` in place, preserving fields), then paste the printed
      `GPSR_INSTRUCTION_POINT_POSE` into `config.toml`.
- [ ] **Manipulation** — `pick`/`place` are wired (shared grasp system); flip
      `GPSR_ENABLE_MANIPULATION=1` once the arm is calibrated. `deliver` stays Tier-2.
- [ ] **Interleave on-robot tuning** — the room-batching lands the bonus's
      movement-reduction; with real poses, consider distance-aware ordering within a
      room and verify the wall-clock saving on the robot.

## How to run / verify

> On-robot validation (arena survey → nav-class → follow/guide → interleave →
> clock budget): [`docs/GPSR_ONROBOT_RUNBOOK.md`](../../docs/GPSR_ONROBOT_RUNBOOK.md).

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
