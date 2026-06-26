# Pick and Place — implementation checklist (rulebook 5.2)

Status legend: `[x]` done / real · `[~]` real motion but **needs on-robot calibration**
· `[G]` **arm-gated** (built, runs only when `PNP_ARM_CALIBRATED=1`) · `[ ]` stub /
ask-referee / not implemented.

> **Arm-excluded scope (current focus).** The arm is being brought up as a
> *separate skill*, so PickAndPlace gates every grasp/place behind
> **`PNP_ARM_CALIBRATED`** (default 0), mirroring Restaurant's
> `RESTAURANT_ARM_CALIBRATED`. With the gate off the flow runs end-to-end and
> earns its **non-arm budget**: navigate, *recognize each object*, and *indicate
> the correct placement* to the referee (rulebook remark 16 — pointing /
> announcing / visualizing one object at a time scores with no grasp). Flip the
> gate to 1 once the arm skill lands and the ~3300-pt manipulation budget unlocks
> with **no flow rewrite**. See [`docs/SCORING.md`](../../docs/SCORING.md#pick-and-place--rulebook-52).
>
> The deliberately-stubbed boundary inside the arm path (when gated on): the
> **grasp planner** (`manipulation.plan_grasp`) is a heuristic that returns the
> hand pose from the object's 3D centroid + config; the robot then really commands
> the arm and gripper. Autonomous appliance manipulation and pouring stay as spoken
> referee requests (the rulebook permits asking, no human-assist penalty for those).

## Main goals

- [x] Navigate to the kitchen / dining table (`GoToKitchen`, `PerceiveDiningTable`).
- [x] Perceive + recognise objects on a surface, open-vocab (`skills.perceive_surface`).
- [x] **Recognize + announce each object** (`announce_object`) — scoresheet 12×10. *(non-arm)*
- [x] **Indicate the correct placement per object** (`indicate_placement`) — remark 16. *(non-arm)*
- [x] **Perceive the cabinet shelf + indicate grouping** (`perceive_and_indicate_shelf`) — 2×30. *(non-arm)*
- [x] Sort each table object → dishwasher / trash / cabinet (LLM, `skills.sort_object`).
- [G] Tidy the dining table — pick + place each object (`TidyDiningTable`, arm pass).
- [G] Place dirty tableware/cutlery in the dishwasher (destination poses to calibrate).
- [G] Place designated trash in the trash bin.
- [G] Store other objects in the cabinet, grouped by similarity (`cabinet_group` from LLM).
- [G] Serve breakfast — fetch bowl+spoon + milk+cereal, arrange on the table
      (`ServeBreakfast`; spoon next to bowl, cereal next to milk via slot poses).
- [G] Tidy the extra surface → cabinet (`TidyExtraSurface`).

## Optional goals

- [ ] Pick up trash from the floor (not yet wired — needs a floor-scan + low grasp pose).
- [ ] Open / close the dishwasher door (`OpenDishwasher` / `CloseDishwasher` — ask referee;
      gated by `PNP_ENABLE_DISHWASHER`; close only after ≥1 item loaded).
- [ ] Pull / push the dishwasher rack (ask referee).
- [ ] Place a dishwasher tab in the slot (not implemented).
- [ ] Pour milk + cereal into the bowl (`PourBreakfast` — ask referee; gated by `PNP_ENABLE_POUR`).

## Implementation status (this repo)

- [x] **Arm gate** `PNP_ARM_CALIBRATED` (default 0) — flow runs end-to-end with no
      arm motion; pick/place self-gate in `skills.py` (`arm_enabled()`).
- [x] **Non-arm scoring path** — `PerceiveDiningTable` announces each recognized
      object; `TidyDiningTable` / `TidyExtraSurface` run a Pass-1 (always: sort +
      indicate placement + shelf indication) then a Pass-2 (arm-gated pick/place).
- [x] **Step-by-step runner** `PNP_SLICE` (`nav|perceive|sort|breakfast|full`) +
      fail-fast `preflight()` in `run.py` (mirrors the Restaurant runner).
- [x] **Offline flow tests** `tests/test_pickandplace_flow.py` — arm-off path
      touches no arm yet indicates every placement; arm-on path picks/places.
- [x] **Scoring worksheet** — PickAndPlace section in `docs/SCORING.md`.
- [x] Task scaffold + flow ordering, dishwasher open-before / close-after (`subtasks.py`).
- [x] Perception 3D lift — full map-frame centroid `world_xyz` (`manipulation.perceive_surface`).
- [x] Map → base_footprint frame transform (`manipulation.world_to_base`, unit-tested).
- [x] Grasp planner stub — top-down (default) / front (`manipulation.plan_grasp`, unit-tested).
- [x] Place pose stub — per-destination (`skills.place_object`) + breakfast slots (`place_at`).
- [x] Real pick motion — open → pre-grasp → grasp → close (`arm.grasp`) → lift → carry.
- [x] Real place motion — pre-place → place → open (release) → carry → close.
- [x] LLM destination sorting + cabinet grouping (`sort_object` → `prompts.ObjectSort`).
- [x] Spoken narration for perception / pick / place / help requests (`prompts.py`).
- [x] Config knobs — robot-wide grasp/arm in root `config.toml` `[manipulation]` (`WALKIE_*`);
      PickAndPlace waypoints + place poses in `tasks/PickAndPlace/config.toml` (`PNP_*`).
- [x] Pure-geometry unit tests (`tests/test_manipulation_geometry.py`).

> The real pick/place now route to the shared **grasp system** (`tasks/skills`:
> GraspNet pick + depth-lifted vision placement, the same layer Restaurant drives);
> `skills.py` is the PickAndPlace-specific facade (class detection, LLM sort,
> destination nav, the gate). Camera-only perception lift (`perceive_surface` ->
> `DetectedObject`) still comes from `tasks/manipulation.py` and feeds the non-arm
> "recognize + indicate placement" budget.

## Calibration TODO (before a real run) — the `[~]` items above depend on this

- [ ] Map the arena and set every waypoint in `config.toml`:
      `PNP_KITCHEN_POSE`, `PNP_DINING_TABLE_POSE`, `PNP_DISHWASHER_POSE`,
      `PNP_CABINET_POSE`, `PNP_TRASH_BIN_POSE`, `PNP_BREAKFAST_SURFACE_POSE`,
      `PNP_EXTRA_SURFACE_POSE` (`"x,y,heading_rad"`).
- [ ] Tune the grasp orientations `PNP_GRASP_RPY_TOPDOWN` / `PNP_GRASP_RPY_FRONT` and
      `PNP_GRASP_Z_OFFSET_M` / `PNP_PREGRASP_OFFSET_M` / `PNP_LIFT_HEIGHT_M` on the arm
      (defaults are borrowed from HRI reach poses — not verified for grasping).
- [ ] Set the destination drop poses `PNP_PLACE_POSE_{DISHWASHER,CABINET,TRASH}` and the
      breakfast slot poses `PNP_BREAKFAST_{BOWL,SPOON,MILK,CEREAL}_POSE`
      (`"x,y,z,roll,pitch,yaw"` in `PNP_ARM_FRAME`).
- [ ] Set `PNP_TRASH_CATEGORY` to the category announced during Setup Days.
- [ ] Decide `PNP_ARM` (left/right) and `PNP_ARM_FRAME` (`base_footprint` vs `map`).

## Stubs / not autonomous yet

- Grasp quality is heuristic (centroid + fixed orientation), not a learned grasp network.
- Dishwasher door/rack and dishwasher-tab placement: ask the referee.
- Milk-container opening and pouring: ask the referee.
- Floor-trash optional goal: not wired into the flow.

## How to run / verify

```bash
# Offline tests (no robot, no server):
uv run pytest tests/test_pickandplace_flow.py tests/test_manipulation_geometry.py

# Build/import smoke test:
uv run python -c "from tasks.PickAndPlace.subtasks import build_pick_and_place_task; print('ok')"

# On the robot, step-by-step bring-up (arm stays gated; needs walkie-ai-server).
# Validate each slice before the next:
DISABLE_LISTENING=1 PNP_SLICE=nav       uv run python -m tasks.PickAndPlace.run  # waypoint tour
DISABLE_LISTENING=1 PNP_SLICE=perceive  uv run python -m tasks.PickAndPlace.run  # recognize each object
DISABLE_LISTENING=1 PNP_SLICE=sort      uv run python -m tasks.PickAndPlace.run  # recognize + indicate placement
DISABLE_LISTENING=1 PNP_SLICE=full      uv run python -m tasks.PickAndPlace.run  # whole flow

# Once the arm skill lands + is calibrated, unlock manipulation:
PNP_ARM_CALIBRATED=1 PNP_SLICE=full     uv run python -m tasks.PickAndPlace.run
```
