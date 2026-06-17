# Pick and Place — implementation checklist (rulebook 5.2)

Status legend: `[x]` done / real · `[~]` real motion but **needs on-robot calibration**
· `[ ]` stub / ask-referee / not implemented.

> The deliberately-stubbed boundary (agreed with the team): the **grasp planner**
> (`skills.plan_grasp` / `plan_place`) is a heuristic that returns the hand pose from
> the object's 3D centroid + config; the robot then **really** commands the arm and
> gripper. Autonomous appliance manipulation and pouring stay as spoken referee
> requests (the rulebook permits asking, with no human-assist penalty for those).

## Main goals

- [x] Navigate to the kitchen / dining table (`GoToKitchen`, `PerceiveDiningTable`).
- [x] Perceive + recognise objects on a surface, open-vocab (`skills.perceive_surface`).
- [x] Communicate perception to the referee (spoken announce + arm reach on every pick).
- [x] Sort each table object → dishwasher / trash / cabinet (LLM, `skills.sort_object`).
- [~] Tidy the dining table — pick + place each object (`TidyDiningTable`).
- [~] Place dirty tableware/cutlery in the dishwasher (destination poses to calibrate).
- [~] Place designated trash in the trash bin.
- [~] Store other objects in the cabinet, grouped by similarity (`cabinet_group` from LLM).
- [~] Serve breakfast — fetch bowl+spoon + milk+cereal, arrange on the table
      (`ServeBreakfast`; spoon next to bowl, cereal next to milk via slot poses).
- [~] Tidy the extra surface → cabinet (`TidyExtraSurface`).

## Optional goals

- [ ] Pick up trash from the floor (not yet wired — needs a floor-scan + low grasp pose).
- [ ] Open / close the dishwasher door (`OpenDishwasher` / `CloseDishwasher` — ask referee;
      gated by `PNP_ENABLE_DISHWASHER`; close only after ≥1 item loaded).
- [ ] Pull / push the dishwasher rack (ask referee).
- [ ] Place a dishwasher tab in the slot (not implemented).
- [ ] Pour milk + cereal into the bowl (`PourBreakfast` — ask referee; gated by `PNP_ENABLE_POUR`).

## Implementation status (this repo)

- [x] Task scaffold + flow ordering, dishwasher open-before / close-after (`subtasks.py`).
- [x] Perception 3D lift — full map-frame centroid `world_xyz` (`perceive_surface`).
- [x] Map → base_footprint frame transform (`skills.world_to_base`, unit-tested).
- [x] Grasp planner stub — top-down (default) / front (`skills.plan_grasp`, unit-tested).
- [x] Place pose stub — per-destination + breakfast slots (`skills.plan_place` / `place_at`).
- [x] Real pick motion — open → pre-grasp → grasp → close (`arm.grasp`) → lift → carry.
- [x] Real place motion — pre-place → place → open (release) → carry → close.
- [x] LLM destination sorting + cabinet grouping (`sort_object` → `prompts.ObjectSort`).
- [x] Spoken narration for perception / pick / place / help requests (`prompts.py`).
- [x] Config knobs for arm, frame, grasp approach/orientation, offsets, poses (`config.toml`).
- [x] Pure-geometry unit tests (`tests/test_pickandplace_geometry.py`).

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
# Pure-geometry unit tests (no robot, no server):
uv run pytest tests/test_pickandplace_geometry.py

# Build/import smoke test:
uv run python -c "from tasks.PickAndPlace.subtasks import build_pick_and_place_task; print('ok')"

# On the robot (after calibration; needs walkie-ai-server):
DISABLE_LISTENING=1 uv run python -m tasks.PickAndPlace.run
```
