# Restaurant — implementation checklist (rulebook 5.5)

Status legend: `[x]` done / real · `[~]` real motion but **needs on-robot calibration**
· `[ ]` stub / not implemented.

> Stub boundary: the **grasp planner** behind collect/serve (`tasks/manipulation.py`)
> is a heuristic that returns the hand pose from the item's 3D centroid + config; the
> robot then **really** commands the arm and gripper. The unattached-tray optional goal
> and online-SLAM tuning are out of scope (we only issue approach goals to the nav stack).

## Scored actions (§5.5 score sheet)

- [x] Detect a calling/waving customer (2×80) — raised-wrist pose heuristic
      (`skills.is_calling_gesture` / `detect_calling_customer`).
- [~] Reach a customer's table (2×80) — lift the customer to a world point + drive
      stop-short (`navigate_to_customer`); needs depth + nav-stack on the robot.
- [x] Understand + confirm the order (2×160) — `take_order` (ask → STT → LLM `Order`
      extract → spoken confirm). Robot faces the customer for eye contact.
- [x] Partial credit: clearly identify a customer it can't reach (`identify_customer`,
      captions the person and announces a description).
- [x] Communicate the order to the barman (2×80) — spoken relay at the kitchen-bar.
- [~] Pick the requested items from the kitchen-bar (4×100 + first-pick bonus) —
      `pick_bar_item` → shared `pick_object` (real arm motion; grasp pose to calibrate).
- [~] Return to the customer table with the order (2×80) — `navigate_to_customer` again.
- [~] Serve the order to the customer (4×100 + first-place bonus) — `serve_item` →
      `place_at_pose(RESTAURANT_SERVE_POSE)` (real arm motion; serve pose to calibrate).
- [ ] Use an unattached tray to transport (optional, 2×200) — gated stub
      (`RESTAURANT_USE_TRAY`); logs a note and serves items individually instead.

## Avoiding penalties

- [x] Reach the bar so the barman doesn't have to come out (drives to the bar pose).
- [x] No alternative-HRI penalty — orders are taken by direct speech (STT), not screens.
- [x] Eye contact when taking the order (robot rotates to face the customer first).
- [ ] Human-assistance penalties (handover / directions) — avoided by design; the robot
      asks the barman only to place the order, not to hand objects over.

## Implementation status (this repo)

- [x] Start at the kitchen-bar facing the dining area (`GoToStart`, critical).
- [x] Real wave/gesture detection from COCO keypoints (pure, unit-tested).
- [x] Customer 3D lift + stop-short approach; rotate-to-face + identify fallback.
- [x] Order dialogue + `Order` LLM schema (`prompts.py`).
- [x] Per-customer serve loop, one item per trip (`ServeCustomers`).
- [x] Collect + serve via the shared grasp planner (`tasks/manipulation.py`).
- [x] Restaurant config knobs (`config.toml`); robot-wide grasp/arm in root `[manipulation]`.
- [x] Pure gesture unit test (`tests/test_restaurant_gesture.py`).

## Calibration TODO (before a real run) — the `[~]` items depend on this

- [ ] Set `RESTAURANT_KITCHEN_BAR_POSE` to the start reference next to the bar
      (the venue is unmapped — this is only a relative start, online nav takes over;
      mapping in advance is a disqualification).
- [ ] Set `RESTAURANT_SERVE_POSE` ("x,y,z,roll,pitch,yaw", arm frame) to place an item
      on a customer's table.
- [ ] Tune the shared grasp orientations in root `config.toml`
      (`WALKIE_GRASP_RPY_TOPDOWN` / `WALKIE_GRASP_RPY_FRONT` + offsets).
- [ ] Adjust `RESTAURANT_BAR_CLASSES` to the official object set and
      `RESTAURANT_APPROACH_DISTANCE_M` / `RESTAURANT_WAVE_CONF` from observed behaviour.

## How to run / verify

```bash
# Pure unit tests (no robot, no server):
uv run pytest tests/test_restaurant_gesture.py tests/test_manipulation_geometry.py

# Build/import smoke test:
uv run python -c "from tasks.Restaurant.subtasks import build_restaurant_task; print('ok')"

# On the robot (after calibration; needs walkie-ai-server):
DISABLE_LISTENING=1 uv run python -m tasks.Restaurant.run
```
