# Restaurant Challenge — Design

> Rulebook 5.5 (RoboCup@Home 2026). Spec: [`docs/Restaurant.pdf`](Restaurant.pdf).
> Scaffold: [`tasks/Restaurant/`](../tasks/Restaurant). Built on the task framework
> in [`tasks/base.py`](../tasks/base.py); the reference task is [`tasks/HRI/`](../tasks/HRI).
>
> Status: **design** — agreed direction, not yet implemented. Restaurant work
> happens on its own branch (`feat/restaurant`).

---

## 1. The task, and where the points are

The robot is a waiter in a **real, undisclosed restaurant** (outside the arena).
It must: spot a customer **calling/waving**, drive to their table, **take and
confirm** their order (2 items), **relay** it to the *Professional Barman* at the
Kitchen-bar, **pick** the items from the bar, and **serve** them at the table.
At least two customers must be served; orders may be batched or interleaved.
Max time **15:00**.

Score map (total 2360) — used to prioritise the build:

| Action | Points | Needs |
|---|--:|---|
| Detect calling/waving customer | 2×80 = 160 | pose keypoints |
| Reach a customer's table | 2×80 = 160 | **nav (gates everything)** |
| Understand + confirm order (eye contact) | 2×160 = 320 | STT + LLM + gaze |
| Communicate order to barman | 2×80 = 160 | nav + TTS |
| Pick items from Kitchen-bar | 4×100 (+100 first) = 500 | **manipulation** |
| Return to customer table | 2×80 = 160 | nav |
| Serve the order | 4×100 (+100 first) = 500 | **manipulation** |
| Use an unattached tray | 2×200 = 400 | **manipulation (bonus)** |
| *(penalties)* | — | guided-to-table −80, ask directions −30, handover help −100 |

**Two reachable tiers:**
- **~960 pts with no arm** — detect + reach + order + relay. Gated only by nav.
- **~1400 more** behind manipulation (pick 500 + serve 500 + tray 400).

So: **nav gates 100% of points; manipulation gates ~60%.** Nav is unambiguously
first; manipulation is the long pole started early but landing late.

---

## 2. Why it's harder than HRI, and the one fact that shapes nav

| | HRI | Restaurant |
|---|---|---|
| World | pre-mapped arena, fixed waypoints | **real venue, mapping in advance = disqualification** |
| Flow | linear, 2 known guests | **reactive** — customers call at random, possibly at once |
| Manipulation | optional bag | **core of the score** |
| Risk | controlled | **public — any contact with people/furniture = instant e-stop** |

**Localization decision (resolved):** the robot **runs online SLAM / Nav2 and can
localize in the unmapped venue.** That means `walkie.nav.go_to(x, y, heading)`
works, and "approach a customer" can be: lift their pose-bbox to a map point with
`CameraSnapshot.bbox_world_xy` (the same lift HRI already uses), then `go_to` a
stand-off point in front of them. We do **not** need closed-loop `cmd_vel` visual
servoing for the MVP. (`cmd_vel` stays a fallback for the final cm of docking.)

This unblocks the nav design — but see principle **§5.1**: a SLAM pose is still
not to be trusted *blindly* ten minutes later in a crowded room.

---

## 3. Capabilities we have vs. lack (verified against the code)

**Have:**
- **Nav** — `walkie.nav.go_to(x,y,heading,blocking=)`, `cancel`, `stop`,
  `is_navigating`, `distance_remaining`, `current_pose` (Nav2). `cmd_vel_topic`
  exists for low-level velocity.
- **Arm** — `walkie.arm.go_to_pose(pos,rot)`, `go_to_pose_relative`,
  `go_to_home`, `control_gripper(value, norm=True)`, `get_ee_pose`; plus
  `walkie.lift.set/get` (torso height). Cartesian primitives, **no grasp planner.**
- **Perception** — `walkieAI.pose_estimation.estimate(img)` → `PersonPose` with
  **17 named COCO keypoints** (wrist/shoulder/nose + confidence); 
  `walkieAI.object_detection.detect(img, prompts=[...], return_mask=True)` →
  `DetectedObject(mask, bbox_xyxy, class_name, confidence)` (open-vocab);
  `CameraSnapshot.bbox_world_xy / bbox_world_point` (depth lift, frozen geometry);
  `image_caption`, `face_recognition`, `appearance`, `image_embed`.
- **Memory** — `walkie_graphs` scene graph; `PeopleStore` (face+attire re-ID, used
  by HRI).
- **Conversation** — `ctx.say / ask / listen / extract` (STT + TTS + LLM).
- **Agent stack** — `WalkieBrain` (walkie_agent + actuator/vision/database).

**Lack (must build):**
1. **Approach-to-stand-off** skill (lift person → `go_to` a facing point at a safe
   distance). 🔴
2. **Waving/calling detection** from keypoints (raised wrist; optional temporal
   wave). 🟡
3. **Continuous gaze tracking** while taking an order (a control loop). 🟡
4. **Re-detect-on-arrival** for the bar and each table (anti-drift). 🟡
5. **Pick / place** built on `go_to_pose` + `control_gripper` + `lift` — fully
   greenfield. ⚠️ `command_arm` (the actuator-agent tool) is a **silent no-op**
   today (it calls `arm.do/execute/command`, none of which exist in the SDK), so
   there is *no* working arm path at all yet. 🔴
6. **Tray** handling (place-on-tray → carry → deliver). 🔴 (bonus)
7. **Order/world state model + scheduler** (below). 🟢
8. **Public-space safety** (slow approach, generous stand-off, e-stop on contact)
   and the **restart** path (5.5.1). 🟡

---

## 4. Architecture — hybrid (deterministic control, LLM at the edges)

Neither a pure HRI-style state machine nor a pure LLM agent. Three layers:

```
┌──────────────────────────────────────────────────────────────────┐
│ Scheduler / policy        decides WHICH order to advance next      │
│   MVP:   deterministic greedy — "nearest unserved caller, finish   │
│          them, repeat" (serial, one customer at a time)            │
│   later: LLM planner for the interleave bonus + recovery           │
├──────────────────────────────────────────────────────────────────┤
│ Per-order state machine   the fixed physical procedure (skills)    │
│   Scan → Approach → TakeOrder → GoToBar → Relay/Receive → Deliver  │
│   deterministic, debuggable, maps 1:1 onto scored steps            │
├──────────────────────────────────────────────────────────────────┤
│ LLM used ONLY for: parse the order (done), word the confirmation / │
│   barman line, schedule across orders (later), decide recovery     │
└──────────────────────────────────────────────────────────────────┘
```

**Why:** the LLM is strong at *deciding* and *language* but must stay **out of
real-time control loops** (approach, gaze, grasp are slow/unsafe under an LLM).
This split mirrors how the repo already divides work (agents for high-level,
skills/SDK for control) — and it lets the deterministic skills carry the MVP
score with the LLM bolted on later for the bonus, exactly like GPSR delegates to
the agent stack.

---

## 5. Three load-bearing principles

### 5.1 Design against drift — *re-detect, don't trust stored coordinates*
Even with SLAM, an absolute table/bar `world_xy` captured at minute 0 is fragile
at minute 10 in a moving crowd. So:
- Use `go_to` to get *near* a remembered landmark, then **re-acquire it visually**
  on arrival (detect the Kitchen-bar / the barman / the customer's appearance)
  and do the final docking against the *fresh* detection.
- A landmark is stored as **`(rough world_xy, appearance caption, bearing)`**, not
  a coordinate we blindly drive onto. `walkie_graphs` can hold these, but the
  *truth* on return is the live camera, not the stored point.

### 5.2 Gaze tracking is a loop that fights blocking STT
"Continuously tracks the moving person" is scored, but `take_order` blocks inside
`ctx.ask`/`listen`. Single-threaded, you can't both block on STT and run a track
loop. Decision:
- **MVP:** re-center on the customer *between* utterances (face → ask → re-detect
  → face → ask). Good enough to score "looking at the person", not full continuous
  tracking.
- **Later:** a background tracker thread that nudges `cmd_vel`/base heading to keep
  the customer centered while STT blocks (true continuous gaze).

### 5.3 The interleave bonus is the *last* thing
It is 200 of 2360. The MVP is **strictly one customer serial**. Only after pick +
serve work do we add the LLM scheduler that batches/interleaves orders.

---

## 6. State model

### 6.1 `Order` (the blackboard unit)
```python
@dataclass
class Order:
    id: int
    # WHERE the customer is — rough anchor + how to re-find them (§5.1).
    world_xy: tuple[float, float] | None   # last good map point of the customer
    bearing: float | None                  # heading from the bar toward them
    appearance: str | None                 # caption, to re-identify on return
    # WHAT they want.
    items: list[str]                       # 2 objects, from take_order
    # Progress (drives the scheduler + scoring).
    status: OrderStatus                    # see below
```
```
OrderStatus:  DETECTED → APPROACHED → ORDERED → RELAYED → PICKED → SERVED
                                                        ↘ FAILED (logged, skipped)
```
Stored on `ctx.data["orders"]: dict[int, Order]`. The scheduler picks the next
order to advance by status + distance.

### 6.2 Per-order state machine (the `SubTask` flow)
The current scaffold (`GoToStart` → `ServeCustomers`) is replaced by an explicit
machine. MVP runs it serially per customer:

| State | Skill(s) | Scores | Notes |
|---|---|---|---|
| **ScanForCaller** | `scan_for_callers` | detect 160 | rotate-sweep the dining area (head can't pan); pick nearest raised hand |
| **ApproachCustomer** | `approach_to_standoff`, `face_person` | reach 160 | lift pose-bbox → `go_to` stand-off; slow, generous clearance |
| **TakeOrder** | `take_order` (✓), `face_and_recenter` | order 320 | ask+confirm; re-center between utterances (§5.2) |
| **GoToBar** | `return_to_bar` (re-detect) | — | `go_to` bar anchor, then re-acquire barman visually |
| **RelayAndReceive** | `relay_to_barman`, `receive_items` | relay 160 | speak order; receive via pick or (penalised) handover |
| **DeliverToCustomer** | `return_to_customer` (re-detect), `serve_items` | return 160 + serve 500 | re-find by appearance; place items |

### 6.3 Scheduler (MVP)
```python
def next_order(orders) -> Order | None:
    # MVP: serial. Newest DETECTED caller, else the in-flight order, finish it.
    # later: LLM chooses to batch two orders' GoToBar/pick to earn interleave 200.
```

---

## 7. Skills to build (`tasks/Restaurant/skills.py`)

Signatures over `TaskContext`, same style as `tasks/HRI/skills.py`. ✓ = scaffold
already has a stub to flesh out.

**Perception / detection**
```python
def scan_for_callers(ctx) -> list[Caller]
    # Sweep the base across RESTAURANT_SCAN_ARC, run pose estimation each step,
    # keep people whose hand is raised (is_calling). Return with world_xy + bearing.

def is_calling(person: PersonPose) -> bool          # ✓ replaces the "central person" stub
    # Keypoint heuristic: wrist.y < shoulder.y (image y grows downward) with
    # confidence gates. Optional temporal: wrist motion across N frames = waving.

def describe_customer(ctx, bbox) -> str             # appearance caption, for re-ID
```

**Navigation (SLAM-backed + re-detect, §5.1)**
```python
def approach_to_standoff(ctx, world_xy, *, standoff_m) -> bool   # ✓ navigate_to_customer
    # Compute a point standoff_m short of the customer along the bearing; go_to it;
    # then face them. Conservative speed; abort on proximity (public-space safety).

def face_person(ctx, world_xy) -> bool              # rotate base to face (head is tilt-only)
def return_to_bar(ctx) -> bool                      # go_to bar anchor, re-acquire barman/bar visually
def return_to_customer(ctx, order: Order) -> bool   # go_to order.world_xy, re-find by appearance
```

**Interaction**
```python
def take_order(ctx) -> list[str]                    # ✓ ask + STT + Order schema + confirm
def relay_to_barman(ctx, items) -> bool             # ✓ speak the order clearly to the barman
def receive_items(ctx, items) -> bool               # autonomous pick preferred; handover is penalised
```

**Manipulation (greenfield — Phase 2; consider promoting to `tasks/base.py`
since Pick&Place and Laundry need the same)**
```python
def pick_item(ctx, item: str) -> bool               # ✓ collect_items
    # detect(item, return_mask=True) → mask centroid → bbox_world_point → grasp
    # pose → lift height → arm.go_to_pose → control_gripper(close). Top-down grasp
    # heuristic first.
def serve_item(ctx, world_xy, item) -> bool         # ✓ serve_order
    # face table → arm.go_to_pose over the table surface → control_gripper(open).
def use_tray(ctx, items) -> bool                    # bonus: place-on-tray → carry → deliver
```

---

## 8. Config (`tasks/Restaurant/config.toml`, additive to the scaffold)
```toml
RESTAURANT_KITCHEN_BAR_POSE   # captured at start as the bar anchor (re-acquired visually on return)
RESTAURANT_CAMERA_HFOV_DEG    # pixel→bearing for caller aim          (have)
RESTAURANT_TARGET_CUSTOMERS   # serve at least 2                       (have)
RESTAURANT_SCAN_ARC_DEG       # base sweep arc for ScanForCaller       (new)
RESTAURANT_STANDOFF_M         # stop distance in front of a customer/table (new)
RESTAURANT_APPROACH_SPEED     # conservative cap for public space      (new)
RESTAURANT_CALL_WRIST_MARGIN  # is_calling keypoint threshold          (new)
RESTAURANT_USE_TRAY           # optional tray goal                     (have)
RESTAURANT_GAZE_TRACK         # 0 = re-center between utterances, 1 = background thread (new)
```

---

## 9. Mapping onto the scaffold
- `subtasks.py` — replace `ServeCustomers` with the §6.2 state machine + the §6.3
  scheduler; add the `Order`/`OrderStatus` dataclasses.
- `skills.py` — flesh out the §7 stubs (`is_calling`, `approach_to_standoff`,
  `take_order`, `relay_to_barman`) and add the new ones; manipulation later.
- `prompts.py` — already has `Order` + barman/confirm lines; add recovery wording.
- `config.toml` — add the §8 knobs.
- `run.py` — capture the bar anchor at startup; `ctx.people=None` for MVP (gesture,
  not face re-ID) — revisit if customer re-ID by face proves more robust than
  appearance caption.

---

## 10. Roadmap (by risk, validated on-robot — this box can't dry-run reactive loops)

- **Phase 0 — first vertical slice (riskiest first):** `ScanForCaller →
  ApproachCustomer → face`. Proves detection + the SLAM approach + safety on the
  real robot before anything is built on top.
- **Phase 1 — MVP ~960 pts (no arm):** add `TakeOrder` (✓) → `GoToBar` →
  `RelayToBarman`. Full serve loop minus manipulation. Deterministic serial.
- **Phase 2 — Manipulation ~1000 pts:** `pick_item` + `serve_item` on
  `go_to_pose`/`control_gripper`/`lift`. **Start the perception→grasp-pose spike
  during Phase 1** — it's the long pole.
- **Phase 3 — Throughput + bonus:** batched order-taking (take several orders per
  sweep — a throughput win within the 15-min limit, *not* a separate bonus line),
  `use_tray` (the real extra reward, 2×200), background gaze-tracking thread.

---

## 11. Open questions / risks
- **Grasp planning** is greenfield and shared with Pick&Place / Laundry — worth a
  common `tasks/base.py` pick/place rather than three copies. Decide where it lives.
- **STT in a noisy public venue** — re-asking is allowed/unpenalised; budget for it.
- **On-board compute** — "assume no wireless"; the AI server must travel with the
  robot (affects the two-machine workflow). Confirm the inference box is on-robot.
- **Barman handover vs. autonomous pick** — asking for handover help is −100;
  decide how hard to push autonomy vs. take the penalty early.
- **Customer re-ID** — appearance caption (current plan) vs. `PeopleStore` face
  re-ID. Start with appearance; escalate if return-to-customer mis-fires.
- **Restart (5.5.1)** — needs a clean "reset to start, void all `Order` state" path.

---

## 12. Implementation status (branch `feat/restaurant`)

- **Phase 0 — DONE (off-robot verified, on-robot untested):** `is_calling`
  (keypoint), `scan_for_callers` (base sweep), `approach_to_standoff`,
  `face_person`; `GoToStart` → `ScanAndApproach`. Run isolated with
  `RESTAURANT_PHASE0=1`.
- **Phase 1 — DONE (off-robot verified):** gaze re-center in `take_order`,
  `capture_appearance`, `find_person_near`, `return_to_bar` / `return_to_customer`
  (re-detect on arrival), full serial `ServeCustomers` loop (order + relay real).
  `take_order` confirms **and listens** — an explicit "no" triggers one re-take
  (protects the 2×160 confirm score); silence reads as agreement so venue noise
  can't drop a good order. The serial loop counts **distinct** customers via
  `exclude_handled` (`RESTAURANT_HANDLED_RADIUS_M`) — see the design-review note.
- **Phase 2 — CALIBRATION-READY SCAFFOLD (NOT VALIDATED):** `_map_to_base` (pure,
  unit-tested), `_in_reach`, `locate_item`, `pick_item` / `serve_item` /
  `collect_items` / `serve_order`. **Fail-safe by default** — they compute and log
  the target pose but DO NOT move the arm unless `RESTAURANT_ARM_CALIBRATED=1`.
  `collect_items`/`serve_order` are **per-item** (return the items actually
  picked/served) so one failed object doesn't forfeit the others' 4×100 credit.
  ⚠️ **Structural gap (calibration-gated):** the current flow picks *all* items
  then serves *all* — physically possible only with a tray (one gripper holds one
  object). The real no-tray flow is pick→deliver→return→pick→deliver per item;
  restructure `_pick_and_serve` during the on-robot pass once the arm is validated.
- **Phase 3 — PARTIAL:** batched order-taking (`ServeCustomersBatched`, opt-in via
  `RESTAURANT_BATCH`; pure scheduling, off-robot verified) DONE; tray
  (`transport_with_tray`) is a logged no-move stub (bonus, bimanual — needs
  calibration); background gaze-tracking thread is still future (the MVP re-centers
  between utterances instead).

### Offline test coverage (`tests/test_restaurant_skills.py`)

The pure logic carrying the no-arm tier (~960 pts) has an in-suite regression test
(was zero — the `manual_tests/` dry-runs are deliberately uncollected):
`is_calling` (caller detection, 160), `_dedup_callers` / `exclude_handled` (the
distinct-customer accounting behind the ">= 2 customers" gate), `_said_no` (order
confirmation, 320), `_scan_offsets`, `_cxcywh_to_xyxy`, and the manipulation
geometry `_map_to_base` / `_in_reach`. These import on a GPU-less dev box because
`tasks/base.py` keeps `WalkieInterface` / `WalkieAIClient` under `TYPE_CHECKING`
(annotations are lazy strings), so pure task logic doesn't pull `silero_vad` →
`torch` → CUDA. The teammate's grasp bring-up harness (the old `TestTask`) lives in
`manual_tests/test_restaurant_grasp.py`, not the serial task.

### Phase 2 calibration checklist (one ordered on-robot pass)
Do these with the robot stationary and a clear bench before setting
`RESTAURANT_ARM_CALIBRATED=1`. Each is a config knob — no code change.

1. **Groups** — confirm `RESTAURANT_ARM_GROUP` / `RESTAURANT_GRIPPER_GROUP` match
   the MoveIt group names (`left_arm`/`right_arm`, …).
2. **Gripper widths** — measure `RESTAURANT_GRIPPER_OPEN_M` / `_CLOSED_M` (m) for
   the actual objects.
3. **map→base z (`RESTAURANT_Z_OFFSET`)** — ON-ROBOT VERIFY #1: place an object at a
   known height, compare `pick_item`'s logged `base_grasp` z to reality; set the
   offset to close the gap.
4. **Grasp orientation (`RESTAURANT_GRASP_RPY`)** — jog the arm to a comfortable
   top-down (or side) grasp; read back the RPY; set it.
5. **Pre-grasp clearance (`RESTAURANT_PREGRASP_DZ`)** — enough to clear the rim.
6. **Reach envelope (`RESTAURANT_REACH_*`)** — jog to the arm's limits; set the box
   so unreachable targets are rejected (logged "needs base reposition").
7. **Lift heights (`RESTAURANT_LIFT_PICK` / `_CARRY`)** — torso height that brings
   the bar / table into the reach box.
8. **Place offset (`RESTAURANT_PLACE_OFFSET`)** — base-frame point over a typical
   table; verify it's inside the reach box.
9. Dry-run `pick_item` UNCALIBRATED first and eyeball every logged pose; only then
   flip `RESTAURANT_ARM_CALIBRATED=1` and test pick → serve on the bench, then in
   the loop.

### Design-review follow-ups (2026-06-18)

**Fixed (off-robot, verified):**
- **Distinct-customer serving.** `ServeCustomers` previously re-scanned after each
  order with no memory and incremented a raw `served` counter on the *relay* path;
  a still-waving customer could be re-selected and counted twice, exiting the loop
  having served ONE distinct person (failing the rulebook's "≥ 2 customers" on the
  ~960-pt no-arm tier). Now it loops on `len(handled)` distinct customers and skips
  callers within `RESTAURANT_HANDLED_RADIUS_M` of an already-ordered one.
  (`ServeCustomersBatched` never had this — it takes distinct callers from one
  dedup'd sweep.)
- **Failure-retry starvation (sibling bug).** A customer whose order never parsed
  (venue-noise STT failure, §11) was never marked handled, so the loop could spend
  all `max_attempts` retrying the same nearest caller while a second waving customer
  went unreached → 0–1 distinct served. Now a spot is abandoned after
  `RESTAURANT_MAX_FAILS_PER_SPOT` (default 2) failures — a transient retry is still
  allowed, but it can't monopolize the loop.
- **Per-item partial credit.** `collect_items`/`serve_order` no longer `all()`-gate;
  they return the items actually handled so a single failure keeps the rest.
- **Confirm-and-listen** in `take_order` (see Phase 1 above). Deliberate choice: if
  the customer rejects but the correction is *also* unparseable, keep the first
  parse and proceed (best-effort order > dropping the customer).

**Verified correct against the real SDK (was feared broken):** `walkie.arm.go_to_pose`
/ `control_gripper` / `go_to_home` / `robot.lift.set` signatures, and both bbox
conventions (`PersonPose.bbox` is `cxcywh`; object detections + the depth lift are
`xyxy`). The fail-safe arm gate is sound.

**Still open — validate ON ROBOT (not code defects):**
- The Phase-2 no-tray flow restructure (above) — calibration-gated.
- **Eye-contact via Nav2 rotation:** `face_person`→`rotate_to`→`go_to(same x,y, new
  heading)` — confirm Nav2 spins in place rather than refusing/path-planning; fall
  back to `cmd_vel` spin if jerky.
- **Scan-sweep timing:** 5 steps × (blocking rotate + settle + pose round-trip) per
  sweep, multiple sweeps — time it against the 15-min budget.

**Deliberately not built:**
- **Restart (5.5.1):** human-initiated — an operator procedure + a `ctx.data` reset,
  not a code path.
- **"Show a picture" partial credit:** hardware-blocked (no screen — camera/mic/
  speaker only); speaking the appearance would count but draws the Alternative-HRI
  −80 penalty.
