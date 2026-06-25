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
- The per-line **points/counts are code-backed**: each challenge's scoresheet is
  encoded as a `ScoreSheet` in `tasks/<Challenge>/scoring.py`, the framework is
  [`tasks/scoring.py`](../tasks/scoring.py), and `tests/test_scoring.py` reconciles
  every sheet's positive lines against its rulebook total + checks each non-arm
  ceiling. The prose below (scenarios, levers) is hand-maintained; the *numbers*
  come from the code. The same `ScoreSheet` also drives a live runtime tally
  (`ScoreTracker`) — read it as *attempted/claimed* points, **not** referee-awarded.

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

## Restaurant — rulebook 5.5

**Total 2360** (code: [`tasks/Restaurant/scoring.py`](../tasks/Restaurant/scoring.py)).
The arm is gated (`RESTAURANT_ARM_CALIBRATED`); the Phase-0 serve pipeline scores the
**960-pt non-arm budget** without it: detect a waving customer, reach the table,
take + confirm the order, relay it to the barman, return.

### Achievable now (arm gated) — the 960 non-arm budget

| Line | Pts | Capability | Low | Exp | High | Notes |
|---|--:|---|--:|--:|--:|---|
| Detect a calling/waving customer | 2×80 = 160 | 🟡 gesture scan | 0.50 | 0.75 | 0.90 | full-arc wave detection |
| Reach a customer's table | 2×80 = 160 | ✅ nav | 0.60 | 0.85 | 0.95 | approach + stand-off |
| Understand + confirm the order | 2×160 = 320 | 🟡 STT + parse + confirm | 0.45 | 0.70 | 0.90 | the biggest single non-arm line |
| Communicate the order to the barman | 2×80 = 160 | ✅ speak | 0.60 | 0.85 | 0.95 | relay at the bar |
| Return to the customer's table | 2×80 = 160 | ✅ nav | 0.60 | 0.85 | 0.95 | |
| **Subtotal** | **960** | | **~512** | **~752** | **~888** | |

### Gated on the arm skill — the 1400 upside
Pick the items from the bar (4×100), First Pick Bonus (+100), serve the order
(4×100), First Place Bonus (+100), use an unattached tray (2×200).

### Penalties (non-arm)
Being guided to a table (2×−80), Alternative HRI (2×−80), not making eye-contact
(2×−60), not reaching the bar (2×−60), asking directional confirmation (2×−30),
being told where a table/bar is (2×−40). The serve loop's gaze re-facing + order
re-ask are the mitigations.

### Biggest levers
1. **Land the arm skill** — opens the 1400 (pick/serve/tray).
2. **Order-capture reliability** (the 320 line) — STT + parse + confirm is the
   largest non-arm line; tune re-ask + confirmation.
3. **Waving-customer detection** — gates everything downstream.

---

## HRI / Receptionist — rulebook 5.x

**Total 1450** (code: [`tasks/HRI/scoring.py`](../tasks/HRI/scoring.py)). Unusually,
HRI is **almost entirely non-arm** — the **950-pt non-arm budget** (gaze, seating,
guest recognition + introduction, following the host) is most of the challenge; only
the entrance door (2×200) and the bag (receive 50 + drop 50) need the arm.

> ⚠️ **The 12-step flow is currently commented out** (`tasks/HRI/subtasks.py` — only a
> follow-host test harness runs). Re-activating it is the single biggest unlock here:
> ~950 pts of positive budget **and** the penalty guard below, almost all offline work.

### Achievable now (once the flow is re-activated, arm gated) — 950 non-arm

| Line | Pts | Capability | Low | Exp | High |
|---|--:|---|--:|--:|--:|
| Detect the doorbell | 2×30 = 60 | ❌ out of scope today | 0.0 | 0.0 | 0.50 |
| Look at the person talking | 2×50 = 100 | 🟡 FaceTracker gaze | 0.50 | 0.80 | 0.95 |
| Offer a free seat | 2×100 = 200 | 🟡 seat detection | 0.40 | 0.70 | 0.90 |
| Look in the navigation direction | 2×15 = 30 | ✅ base heading | 0.50 | 0.80 | 0.95 |
| Correct visual attribute to 2nd guest | 4×20 = 80 | 🟡 appearance caption | 0.30 | 0.60 | 0.85 |
| No non-essential questions | 4×15 = 60 | ✅ prompt design | 0.50 | 0.80 | 0.95 |
| Name + favourite drink in intro | 4×30 = 120 | 🟡 memory + speak | 0.40 | 0.70 | 0.90 |
| Look at correct guest while introducing | 2×50 = 100 | 🟡 face re-ID | 0.40 | 0.70 | 0.90 |
| Follow the host to the bag drop | 200 | 🟡 nav + re-ID* | 0.30 | 0.60 | 0.85 |
| **Subtotal** | **950** | | **~347** | **~614** | **~826** |

\* `follow_host` is a non-arm nav + re-ID skill, but earning it presumes the bag was
received (an arm step) — a flow dependency, not an arm requirement of the follow.

### Non-arm penalties (the real downside — guarded by gaze + re-ID)
**Not acknowledging people 2×−200 = −400** and **wrong guest info 4×−40 = −160**
dwarf the positive deltas — so robust gaze-at-speaker + appearance/face re-ID guard
~560 pts, not just earn the smaller positive lines. Plus alternative-HRI (6×−20) and
the operator-handling penalties on the follow path.

### Biggest levers
1. **Re-activate the 12-step flow** (`tasks/HRI/subtasks.py`) + validate gaze runs
   during *listening*, not just asking — unlocks ~950 + guards ~560 (mostly offline).
2. **Appearance/face re-ID reliability** — lifts intro + follow lines and guards the
   −400 acknowledge penalty (see [Chalk's eic-human pipeline]).
3. **Doorbell detection** (60) — the one greenfield non-arm line, currently punted.

---

## Laundry — rulebook 5.x

**Total 4415** (code: [`tasks/Laundry/scoring.py`](../tasks/Laundry/scoring.py)).
**Laundry is an almost-pure manipulation challenge**: the only non-arm line on the
scoresheet is *navigate to the laundry area* (**15 pts**). Unlike PickAndPlace, there
are **no recognize / indicate-placement lines** to score via "communicate perception"
— so an arm-gated run earns ~15 and nothing more.

| Line | Pts | Capability | Exp |
|---|--:|---|--:|
| Navigate to the laundry area | 15 | ✅ nav | ~14 |
| *everything else* (pick / fold / stack / washer / basket) | 4400 | ❌ arm-gated | 0 |

**Lever:** the entire challenge is the folding/manipulation skill. Recommendation:
**deprioritise for non-arm work** — gate the manipulation behind a
`LAUNDRY_ARM_CALIBRATED` flag for parity, but there is no meaningful non-arm budget to
develop here until the arm + folding skill exists.

---

## Other challenges

_Same shape for any future challenge (Carry My Luggage, Stickler, …): encode the
scoresheet in `tasks/<Challenge>/scoring.py`, add a reconciliation test, then write the
fixed/non-arm budget → capture % → scenarios → penalties → levers prose here._
