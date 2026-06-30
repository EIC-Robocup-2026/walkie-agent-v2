# GPSR — On-Robot Validation Runbook

> Companion to [`docs/GPSR_DESIGN.md`](GPSR_DESIGN.md) and the live
> [`tasks/GPSR/CHECKLIST.md`](../tasks/GPSR/CHECKLIST.md). Everything in GPSR is
> offline-verified (99 tests + the OpenRouter coverage/split gates); this runbook
> is the **on-robot** half — the `[~]` items that the dev box (no robot/CUDA)
> cannot prove. Work top to bottom: each section gates the next.
>
> Knobs are env vars; set them in `.env` (gitignored) or inline before the run
> command. Precedence: shell env > `.env` > `tasks/GPSR/config.toml` > root
> `config.toml` > code default.

---

## 0. Pre-flight (before touching the robot)

- [ ] **walkie-ai-server is up** and reachable at `WALKIE_AI_BASE_URL` (default
      `http://localhost:5000`). It hosts STT, TTS, object detection, pose
      estimation, captioning, image-embed — GPSR is dead without it.
- [ ] **`OPENROUTER_API_KEY` set** (the parser + `say`/answer LLM).
- [ ] **walkie-sdk hardware reachable** over Zenoh (nav, arm, camera, mic, speaker).
- [ ] **Offline suite green on this exact checkout**:
      ```bash
      uv run pytest tests/ -k "gpsr and not coverage and not split" -q   # 99 passed
      uv run pytest tests/test_gpsr_coverage.py -s                        # 56/56 = 100%
      ```
- [ ] **Battery / e-stop / clear floor** — nav and the guide look-back spin the base.

---

## 1. Arena survey (the ~2-hour pre-competition window)

The arena nouns ship as defaults in the repo-root `world.toml`; the **poses are
placeholders** (`[0,0,0]`) until surveyed. Nav is a no-op until this is done.

1. [ ] **Update the vocabulary** if the announced CompetitionTemplate differs from
       the default (rooms/locations/objects/names/gestures). Edit `world.toml`.
2. [ ] **Re-run the parser gate** if you changed vocabulary — a noun the parser
       can't ground forfeits that command:
       ```bash
       uv run pytest tests/test_gpsr_coverage.py -s
       ```
3. [ ] **Drive-and-capture every pose** (preserves comments/aliases, saves after
       each capture):
       ```bash
       uv run python -m tasks.GPSR.tools.teach_poses          # only un-surveyed places
       uv run python -m tasks.GPSR.tools.teach_poses --all    # re-survey everything
       ```
       Drive to each room/placement, press Enter. **Mind the heading** — a wrong
       heading sends the robot the wrong way even with the right x,y.
4. [ ] **Instruction Point pose** → set `GPSR_INSTRUCTION_POINT_POSE = "x,y,heading"`
       in `config.toml` (the envelope's start/return waypoint).
5. [ ] **Team identity** for `say`/`tell` answers:
       `GPSR_ROBOT_NAME`, `GPSR_TEAM_NAME`, `GPSR_TEAM_AFFILIATION`.
6. [ ] **Sanity-read** `world.toml`: no remaining `[0,0,0]` on a place you'll use;
       headings look right on the map.

---

## 2. Bring-up smoke test (no nav, no arm — proves the 540)

Run the full task but verify only **understand + speak-a-plan** first. Type
commands instead of speaking to isolate the parser from STT:

```bash
DISABLE_LISTENING=1 uv run python -m tasks.GPSR.run
```

- [ ] Robot **drives to the Instruction Point** (`GoToInstructionPoint`).
- [ ] You give a command; robot **speaks back a correct plan** (`GPSR_SPEAK_PLAN=1`).
- [ ] Then **speak the same command through the mic** (drop `DISABLE_LISTENING`) to
      prove the **STT path** — bypassing STT is −50/command.

> If the spoken plan is wrong, stop and fix the parser/vocab — a wrong plan tanks
> both the 300 *and* that command's 250 (§5.1). This is the highest-leverage gate.

---

## 3. Phase-1 nav-class primitives (one at a time, serial)

Keep `GPSR_INTERLEAVE=0`, `GPSR_ENABLE_MANIPULATION=0`. Drive each primitive with
a single command and watch the behavior. **Watch for double-driving** (the
nav-dedup is the headline integration concern).

| Command | Expect | Watch for |
|---|---|---|
| `go to the kitchen` | one drive to the kitchen pose | reaches the right place + heading |
| `find the cola in the kitchen` | one drive, detect, honest result | **not two drives**; speaks found / not-found |
| `count the cups on the kitchen table` | drive, detect, speak a count | count matches reality (incl. zero) |
| `find a waving person in the living room` | drive, pose-match, face them | gesture match, not nearest-person; honest negative if absent |
| `greet Charlie in the kitchen` | drive, find person, greet by name | addresses the spoken name |
| `tell me the biggest object on the desk` | drive, inspect, answer | superlative query resolves at the placement |
| `tell me what day it is` | spoken answer (config identity + live clock) | correct day/time; no hallucination/refusal |
| `find the person wearing a red shirt in the office` | caption candidates, LLM picks one | **latency** — N captions + 1 LLM call (see §7) |

- [ ] Each above behaves as expected and **speaks** a result (no silent end).
- [ ] A genuinely ungrounded/exotic clause **falls through to Tier-2** (agent) and
      still does something sane.

---

## 4. follow / guide

**follow** (`follow me to the kitchen`):
- [ ] Robot warms up ("lead the way slowly"), tracks whoever is in front
      (`select_largest_person`), and **ends on arrival** at the named place
      (`ArrivalStopper`), announcing arrival — not on the follow timeout.
- [ ] Tune `GPSR_ARRIVAL_RADIUS_M` (default 1.0) if it stops short/late.

**guide** (`guide Charlie from the entrance to the exit`) — re-acquire is **OFF by
default**. Validate the legacy single-drive first, then enable re-acquire:
- [ ] OFF: drives to `from`, faces the person, leads to `to`, announces arrival.
- [ ] Flip on and bench-test the look-back loop:
      ```bash
      GPSR_GUIDE_REACQUIRE=1 GPSR_GUIDE_SEGMENT_M=2.0 uv run python -m tasks.GPSR.run
      ```
  - [ ] Robot leads in **segments**, turns back between hops, continues when the
        follower is visible.
  - [ ] Have the helper **fall behind** → robot prompts "please keep up", waits
        (`GPSR_GUIDE_MAX_MISSES` × `GPSR_GUIDE_REACQUIRE_WAIT_SEC`), then leads on.
  - [ ] **Tune against the clock**: smaller `GPSR_GUIDE_SEGMENT_M` = checks more
        often but slower. Decide whether the turn-arounds are worth it vs. §7.
- [ ] **Only leave `GPSR_GUIDE_REACQUIRE=1` if it's a net win on the robot**;
      otherwise keep it OFF (the safe default).

---

## 5. Interleaved Task Bonus (+200) — last

Default `GPSR_INTERLEAVE=0` (serial). Enable only after serial is solid:
```bash
GPSR_INTERLEAVE=1 GPSR_ISSUE_MODE=consecutive uv run python -m tasks.GPSR.run
```
- [ ] Give **all three commands at once**; robot merges them into one room-batched
      order and **visits each room once** (shared nav cache).
- [ ] **Measure wall-clock** vs. serial — the bonus rewards *meaningful* movement
      reduction. If it doesn't save time or it misbehaves, fall back to serial
      (it auto-falls-back on any scheduling error).

---

## 6. Manipulation (`pick`/`place`/`deliver`) — gated

- [ ] Stays `GPSR_ENABLE_MANIPULATION=0` until the **arm is calibrated** and the
      shared `tasks/base.py` grasp promotion lands (Restaurant §11). Until then
      these clauses fall through to Tier-2; expect ~0 on heavy-manipulation draws.
- [ ] When the arm is ready, flip on and validate `pick`/`place`/`deliver`
      separately before trusting them in a full run.

---

## 7. Clock budget (7:00 for three commands ≈ 2:20 each, incl. travel)

- [ ] **Time a representative full run** (3 commands, return to instruction point).
- [ ] Note the slow categories: clothing `find_person` (N captions + 1 LLM call),
      guide re-acquire turn-arounds, any Tier-2 fallback round-trip.
- [ ] If over budget: drop the slowest category from the strategy, lower
      `GPSR_MAX_REPHRASINGS`, increase `GPSR_GUIDE_SEGMENT_M`, or keep
      `GPSR_GUIDE_REACQUIRE`/`GPSR_INTERLEAVE` off.

---

## 8. Penalty-avoidance checks

- [ ] **STT used** (not typed) for the real run — bypassing STT is 3×−50.
- [ ] **Rephrasing recovery**: give a garbled command → robot re-asks only on an
      empty parse, bounded by `GPSR_MAX_REPHRASINGS` (each rephrase −30), then
      requests a **custom operator** (`GPSR_USE_CUSTOM_OPERATOR`, −20) before
      giving up — and **keeps attending** (never silently dies).
- [ ] **Returns to the Instruction Point** after all commands
      (`ReturnToInstructionPoint`).

---

## 9. viz / debug for the competition run

- [ ] `WALKIE_EXPLORE_VIZ=rerun` is **on by default** (`services/walkie_graphs/config.toml`)
      — it streams the scene graph to Rerun and serves a web viewer
      (`WALKIE_EXPLORE_RERUN_WEB_PORT=8008`, gRPC `9876`). Useful for debugging,
      but it costs CPU/GPU + a network server on the robot.
- [ ] **Decide for the scored run**: keep it for eyes-on debugging, or set
      `WALKIE_EXPLORE_VIZ=none` to free resources. (This is a perception-loop knob,
      independent of GPSR logic.)

---

## 10. Go / no-go + fast fallback flags

**Go** when §2 (speak-a-plan) and §3 (nav-class) are reliable — that's the
draw-independent 540 plus most of the 750, the realistic scoring floor.

Fast fallbacks if something misbehaves mid-competition (most-conservative config):

```bash
GPSR_INTERLEAVE=0            # serial only
GPSR_GUIDE_REACQUIRE=0       # legacy single-drive guide
GPSR_ENABLE_MANIPULATION=0   # no arm
GPSR_ISSUE_MODE=one_by_one   # take commands one at a time (no interleave bonus, simpler)
```

The parser + speak-a-plan path has no flags to disable — it's the safe core.

---

## Appendix — knob quick reference

| Knob | Default | Section |
|---|---|---|
| `GPSR_INSTRUCTION_POINT_POSE` | `0,0,0` | §1 |
| `GPSR_ROBOT_NAME` / `_TEAM_NAME` / `_TEAM_AFFILIATION` | Walkie / EIC / Chulalongkorn University | §1 |
| `GPSR_ISSUE_MODE` | `consecutive` | §5, §10 |
| `GPSR_MAX_COMMANDS` | `3` | §2 |
| `GPSR_SPEAK_PLAN` | `1` | §2 |
| `GPSR_START_PERCEPTION` | `1` | §3 |
| `GPSR_ARRIVAL_RADIUS_M` / `_POLL_SEC` | `1.0` / `0.5` | §4 |
| `GPSR_GUIDE_REACQUIRE` | `0` | §4 |
| `GPSR_GUIDE_SEGMENT_M` | `2.0` | §4, §7 |
| `GPSR_GUIDE_MAX_MISSES` / `_REACQUIRE_WAIT_SEC` | `3` / `2.0` | §4 |
| `GPSR_INTERLEAVE` | `0` | §5 |
| `GPSR_ENABLE_MANIPULATION` | `0` | §6 |
| `GPSR_MAX_REPHRASINGS` | `2` | §7, §8 |
| `GPSR_USE_CUSTOM_OPERATOR` / `_CUSTOM_OPERATOR_ATTEMPTS` | `1` / `3` | §8 |
| `WALKIE_EXPLORE_VIZ` | `rerun` | §9 |
| `DISABLE_LISTENING` | `0` | §2 |
