# Walkie — Score Estimation

A living worksheet for estimating Walkie's expected points per RoboCup@Home
challenge, given **current on-robot capability**. Update the capability columns as
primitives get validated/gated; re-read the per-category capture % when a draw is
announced to get a realistic point estimate.

Conventions used in every challenge section:

- **Capture %** = expected fraction of that line's points we actually earn, given
  partial scoring and current validation. Three columns:
  **Low** (bad luck / unvalidated path), **Exp** (realistic), **High** (clean run).
- A ✅ = validated on-robot, 🟡 = works but partial/untuned, ❌ = gated/not built.
- Estimates assume the **practice-arena** validation status on branch `feat/GPSR`
  as of **2026-06-20** (see [`GPSR_DESIGN.md`](GPSR_DESIGN.md),
  [`GPSR_ONROBOT_RUNBOOK.md`](GPSR_ONROBOT_RUNBOOK.md)).

---

## GPSR (General Purpose Service Robot) — rulebook 5.3

**Total 1490.** Operator speaks up to 3 CommandGenerator commands; robot must STT
them, **speak a plan** (the 300), execute (the 750), and may take all 3 at once
for the interleave bonus (200).

### Fixed (draw-independent) budget — 540 + 200

| Line | Pts | Capability | Low | Exp | High | Notes |
|---|--:|---|--:|--:|--:|---|
| Understand command (STT) | 3×80 = 240 | ✅ mic fixed | 0.83 | 0.92 | 1.0 | wrong STT can also forfeit the plan |
| Speak a plan (parse+TTS) | 3×100 = 300 | ✅ parser 56/56 | 0.80 | 0.90 | 1.0 | wrong parse forfeits the 100 **and** tanks that command's 250 |
| Interleave bonus | 200 | ✅ scored +200 | 0 | 1.0 | 1.0 | Low=play-safe serial; Exp/High=all-3-at-once |
| **Subtotal** | **740** | | **~439** | **~681** | **740** | |

### Solve budget — 750 = 3 commands × 250 (partial scoring), **draw-dependent**

The operator draws the 3 commands; we don't. Below is **expected % of one 250-pt
command** per CommandGenerator category. Multiply by 250, then sum the 3 drawn.

| Generator category | Our primitive(s) | Cap. | Low | Exp | High | Why |
|---|---|---|--:|--:|--:|---|
| `goToLoc` | navigate | ✅ | 0.70 | 0.90 | 1.0 | pure nav; risk = nav/localization fail |
| `findObj`, `findObjInRoom` | find_object (Tier-2) | ✅ | 0.40 | 0.70 | 0.90 | perception reliability; autonomous directive picks 1 candidate |
| `findPrs`, `findPrsInRoom`, `meetName`, `meetPrsAtBeac` | find_person | ✅ | 0.45 | 0.78 | 0.95 | name/clothing/gesture matching validated |
| `countPrsInRoom` | count (persons) | ✅ | 0.40 | 0.72 | 0.90 | full-turn scan + dedup; close-sitting people may merge |
| `countObjOnPlcmt` | count (objects) | 🟡 | 0.25 | 0.50 | 0.75 | object count at placement less validated |
| `greetNameInRm`, `greetClothDscInRm` | greet | ✅ | 0.45 | 0.78 | 0.95 | nav + find_person + greet |
| `talkInfoToGestPrsInRoom` | find_person(gesture)+say | ✅ | 0.45 | 0.72 | 0.92 | gesture find + spoken info |
| `tellPrsInfoInLoc` | find_person+get_person_info+say | 🟡 | 0.35 | 0.65 | 0.88 | multi-step; feature recognition |
| `tellObjPropOnPlcmt` | find_object+get_object_property+say | 🟡 | 0.30 | 0.55 | 0.82 | object feature recognition, less validated |
| `followPrsAtLoc`, `followPrsToRoom`, `followNameFromBeacToRoom` | follow | ✅ | 0.45 | 0.78 | 0.95 | ArrivalStopper; tracks operator, stops on arrival |
| `guidePrsToBeacon`, `guide*FromBeacToBeac` | guide | ✅ | 0.45 | 0.72 | 0.92 | single-drive (re-acquire OFF by choice) |
| `takeObjFromPlcmt` | pick | ❌ gated | 0.10 | 0.20 | 0.30 | manipulation gated; partial from nav+find only |
| `placeObjOnPlcmt` | place | ❌ gated | 0.05 | 0.15 | 0.25 | needs an object already grasped |
| `bringMeObjFromPlcmt`, `deliverObjToMe`, `deliverObjToPrsInRoom`, `deliverObjToNameAtBeac` | deliver (nav+find+pick+deliver) | ❌ gated | 0.15 | 0.30 | 0.40 | nav+find sub-steps partial-score; pick+deliver=0 |

**Reading the draw:** the generator samples roughly across high-level buckets —
~⅓ manipulation (take/place/bring/deliver), ~⅓ find/nav/count, ~⅓ people &
speak (find_person/greet/follow/guide/tell). So a *typical* 3-command draw is
often ~1 manipulation + ~2 non-manipulation.

### Putting it together — run scenarios

| Scenario | Solve (3 cmds) | + Fixed 740 | **Total / 1490** |
|---|--:|--:|--:|
| **Bad draw** (3 manipulation-ish, serial) | 3×250×0.25 ≈ 190 | 439 | **~630** |
| **Expected** (1 manip @0.30 + 2 nav-class @0.75) | 75 + 375 = 450 | 681 | **~1130** |
| **Good draw** (3 nav-class, clean) | 3×250×0.85 ≈ 640 | 740 | **~1380** |
| **Floor** (only the draw-independent budget) | 0 | 439–740 | **~440–740** |

> The Exp scenario above is optimistic on the draw (only 1 manipulation). A
> blended expectation across draws lands **~900–1130**; quote **~950–1000** as the
> single-number planning figure, **~700 as the confident floor** (fixed 540 + a
> likely interleave 200, even on a manipulation-heavy draw).

### Penalties (subtract from the above)

| Penalty | Each | Max | Mitigation in our design |
|---|--:|--:|---|
| Custom operator | −20 | 3×−20 = −60 | bounded re-ask + custom-op recovery flow |
| Requested rephrasing | −30 | 6×−30 = −180 | partial-grounded plan beats asking; "no non-essential questions" |
| Bypass STT (typed) | −50 | 3×−50 = −150 | only if mic fails; STT validated, mic pinned |

### Biggest score levers (in order)

1. **Unlock manipulation** (`GPSR_ENABLE_MANIPULATION`) — opens ~500 of the 750
   that is currently gated; single largest upside.
2. **Always take all 3 at once** → secures the +200 interleave (validated).
3. **Tune count dedup** (`GPSR_COUNT_DEDUP_M`) + object-count path → lifts the
   count/tell categories from 🟡 toward ✅. The object-count path is now
   median-stabilized over `GPSR_COUNT_OBJ_FRAMES` frames and superlative
   ("biggest object") queries pick by image-size — both offline-hardened; the
   remaining lift is on-robot tuning (`GPSR_DETECT_CONF_MIN`, the frame count).

---

## Pick and Place — rulebook 5.2

**Total 3515** (excluding special penalties & bonuses). The scoresheet is
overwhelmingly manipulation: ~3300 of the 3515 needs a working arm. The arm is a
**separate skill under development**, so the PickAndPlace flow gates every
grasp/place behind `PNP_ARM_CALIBRATED` (default 0) and earns the **non-arm
budget** on its own — and rulebook **remark 16** lets the robot *communicate
perception by pointing / announcing / visualizing one object at a time*, so
recognition + placement *indication* score with **no grasp required**.

### Achievable now (arm gated OFF) — the ~195 non-arm budget

| Line | Pts | Capability | Low | Exp | High | Notes |
|---|--:|---|--:|--:|--:|---|
| Navigate to the table | 15 | ✅ nav | 0.70 | 0.90 | 1.0 | pure nav; same stack as GPSR/Restaurant |
| Correctly recognize an object | 12×10 = 120 | 🟡 open-vocab detect + announce | 0.40 | 0.65 | 0.85 | `PerceiveDiningTable` speaks each object; cutlery / stacked / decoy objects are the risk |
| Perceive on a shelf + indicate placement | 2×30 = 60 | 🟡 shelf perceive + indicate | 0.30 | 0.55 | 0.80 | `perceive_and_indicate_shelf` names the cabinet groups + states the match |
| **Subtotal** | **195** | | **~77** | **~125** | **~165** | |

> Indicate-only also **dodges the drop/throw penalties entirely** (nothing is
> ever held), and asking the referee to open the dishwasher / milk is explicitly
> **penalty-free** (remarks 7/8/15). So the non-arm floor has almost no downside.

### Gated on the arm skill (`PNP_ARM_CALIBRATED=1`) — the ~3300 upside

| Bucket | Pts | Cap. | Why gated |
|---|--:|---|---|
| Picking up an object for transport | 12×50 = 600 | ❌ gated | needs the grasp skill |
| First Pick Bonus (one-time) | +100 | ❌ gated | first successful grasp of the run |
| Cutlery / Plate / Dishwasher-tab / floor picks | 100 + 100 + 100 + 30 | ❌ gated | per-class pick rewards |
| Place in designated location | 12×40 = 480 | ❌ gated | needs an object in hand |
| Correctly in the dishwasher | 3×70 = 210 | ❌ gated | rack-correct placement |
| Next to similar in cabinet | 2×20 = 40 | ❌ gated | grouped place (the *indication* scores now; the *place* is gated) |
| Dishwasher-tab in slot | +160 | ❌ gated | precise place |
| Dishwasher door/rack, milk-open, pour (extra rewards) | 200 + 400 + 400 + 400 | ❌ ask-referee / not built | "without assistance" lines need autonomy we don't have; asking is penalty-free but scores 0 |

### Run scenarios

| Scenario | Achievable-now | + arm budget | **Total / 3515** |
|---|--:|--:|--:|
| **Arm gated (today)** | ~125 (exp) | 0 | **~125** (floor ~75, ceiling ~165) |
| **Arm lands, modest** | ~140 | ~600–900 (some picks/places + First Pick) | **~750–1050** |
| **Arm lands, clean** | ~165 | ~1800+ (most picks/places, dishwasher) | **~2000+** |

> Quote **~125 as today's planning figure** (arm gated), **~75 confident floor**.
> The number is small only because the arm is the whole challenge — the point of
> this pipeline is to bank that 125 risk-free *and* be the harness the arm skill
> drops into, so the day the arm lands the 3300 unlocks with no flow rewrite.

### Penalties (subtract)

| Penalty | Each | Mitigation in our design |
|---|--:|---|
| Objects thrown / dropped while placing | −40 | indicate-only holds nothing → **not reachable while gated** |
| Objects dropped on the floor | −40 | same — no carry, no drop |
| Breakfast not in a typical setting | −50 | gated breakfast only *indicates* layout; no malformed place |
| Human assistance: handover | −100 | never request a handover |
| Human assistance: object repositioned | −30 | never ask the referee to move objects |
| Human assistance: environment change (chairs/decor) | −40/item | nav around decor; don't ask for moves |
| Dishwasher door/rack, milk-open assistance | −0 | explicitly penalty-free (remarks 7/8/15) |

### Biggest score levers (in order)

1. **Land + calibrate the arm skill** (`PNP_ARM_CALIBRATED=1`) — opens ~3300 of
   the 3515 that is gated; everything else is a rounding error against this.
   Within it, the **First Pick Bonus (+100)** rewards a single reliable grasp, so
   prioritize one robust pick before breadth.
2. **Tune open-vocab detection** (`PNP_TABLE_CLASSES`, `WALKIE_GRAPHS_DETECT_CONF_MIN`)
   — the recognize line (120) is the largest points available *pre-arm*; lift it
   from 🟡 toward ✅ on the real object set.
3. **Validate the shelf-perception indication** (`perceive_and_indicate_shelf`,
   `PNP_CABINET_CLASSES`) — the other 60 pre-arm points; needs the cabinet
   waypoint mapped and the shelf groups read reliably.

### Step-by-step bring-up (mirrors the Restaurant runner)

`PNP_SLICE=nav` (waypoint tour) → `perceive` (recognize each object) → `sort`
(recognize + indicate placement + shelf indication — the full non-arm scoring
path) → `breakfast` → `full`. Validate each on-robot before the next; the arm
stays gated the whole way until its skill is calibrated.

---

## Other challenges

_Add a section per challenge (Restaurant, HRI, Carry My Luggage, …) using the same
table shape: fixed budget → per-category capture % → run scenarios → penalties →
levers. Fill capability columns from each task's on-robot validation memory._
