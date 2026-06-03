# Human recognition — design (RoboCup @Home HRI, Receptionist first)

Status: **proposal**. Scope: add a human-recognition capability to `walkie-agent-v2`,
targeting the **Receptionist** task first, via a new `human_agent` sub-agent and a
face-keyed people memory. This mirrors the structure of `scene_perception_design.md`
but for *people*, which are deliberately **not** stored in the scene catalogue.

---

## 1. Why people need their own path

The scene store (`perception/store.py`) is a **spatial** catalogue: it deduplicates
by 3D position (`SCENE_DEDUP_RADIUS_M`) and prunes by location. People are filtered
out of it on purpose — `SCENE_EXCLUDE_CLASSES` defaults to `person` and the pipeline
drops them *before* lift/caption/embed (`perception/pipeline.py`). The reason:

- people move every tick → position dedup would insert a new record each frame;
- "where I last saw a person" is rarely the question — "**who** is this person" is.

So human memory is keyed by **identity (face/appearance embedding), not position**.
It is a parallel subsystem, not an extension of the scene store.

> **The rulebook forces this.** RoboCup@Home 2026 §5.1 states *"Switching Places:
> After being seated, guests may switch seats."* Position therefore cannot identify
> a guest — the robot must recognise them by **face**. And *"Not recognizing people"*
> is the single heaviest penalty on the score sheet (**2×−200**), so face re-ID is
> not optional polish, it's the highest-value capability in the task.

## 2. The task (RoboCup@Home 2026 §5.1 — HRI Challenge = Receptionist)

Source: `docs/rulebook.pdf` §5.1 "Human Robot Interaction Challenge". Stated focus:
*"System Integration, Human-Robot Interaction, Person Detection, **Person Recognition**."*

Flow: the robot waits at a start position; a **doorbell** rings; it **approaches and
meets each of two guests at the door** (it is not allowed to make guests come to it);
**greets and asks name + favorite drink** (no confirmation/"non-essential" questions,
for bonus); **escorts** them to the living room while **keeping gaze on the person /
on the navigation direction**; **offers a free seat**; once both are seated,
**introduces them to each other** — looking at one guest and stating the *other*
guest's name + favorite drink. The 2nd guest hands over a **bag** which the robot
carries and drops while **following the host**.

Favorite drink = "any english named popular drink (it may **not** be in the list of
objects)" → it's a **free-form string captured from STT**, not an object-detector class.

### What the score sheet actually rewards (drives our priority)

| Action (rulebook score sheet) | Points | Human-recognition capability |
|---|---|---|
| **Not recognizing people** (penalty) | **2×−200** | **face re-ID** — the dominant term |
| Wrong guest information memorized (penalty) | 4×−40 | accurate name/drink capture |
| Say name + favorite drink during introduction | 4×30 | identity ↔ name/drink memory |
| Look at the person talking | 2×50 | **person tracking / gaze** |
| While introducing, look at the correct guest | 2×50 | re-ID + person tracking |
| Look in the navigation direction | 2×15 | (nav, out of scope here) |
| Offer a free seat | 2×100 | empty-seat finding |
| Tell a visual attribute of guest 1 to guest 2 | 4×±20 | **attribute description** |

Re-ID alone swings **±400+**. Description and gaze are smaller but cheap to add.

### Required capabilities

| # | Capability | Used for | New work |
|---|---|---|---|
| C1 | **Person detection + count** | meet at door, find seated guests | thin — reuse `pose_estimation` bboxes |
| C2 | **Face enroll** (face → embedding, bind name + drink) | remember each guest | server `/face-embed` + `people_store` |
| C3 | **Face recognize** (embedding → known person) | introduce / point out who is who, survive seat-swaps | `people_store` knn |
| C4 | **Attribute description** (clothing, hair, glasses, posture) | "tell a visual attribute of guest 1" | `image_caption` w/ steering prompt ✅ shipped |
| C5 | **Empty-seat finding** | "offer a free seat" | scene store (chairs/sofas) ∩ person occupancy |
| C6 | **Person tracking / gaze target** | "look at the person talking / the correct guest" | track a person across frames → a point to aim head/base at |
| C7 | Gesture/posture (wave, point, sit/stand) | *not* Receptionist — GPSR/Restaurant | partial (`count_people`); see §7 |

C2/C3 are the only genuinely new heavy-model dependency. C1/C4 already reuse existing
routes (C4 is shipped). C5/C6 reuse existing detection/pose but need new glue.

## 3. Cross-repo split

```
walkie-ai-server   (GPU box)        walkie-agent-v2  (this repo)
──────────────────────────────      ────────────────────────────────────────────
/face-embed   POST image            client/face_recognition.py   (thin HTTP client)
  → [{bbox, embedding[512], score}] perception/people_store.py    (face-keyed memory)
                                     agents/human_agent/           (HRI sub-agent)
(face detect + ArcFace/InsightFace)  agents/walkie_agent: delegate_to_human
```

Server contract proposal for `/face-embed` (same `{"success", "data"}` envelope the
other routes use, so `client/base.py` unwraps it unchanged):

```jsonc
// request: multipart image (same as /object-detection)
// response data:
[ { "bbox_xyxy": [x1,y1,x2,y2], "embedding": [..512 floats, L2-normed..], "det_score": 0.99 } ]
```

If standing up a new model on the server is too much for now, **C4 (attributes) can
ship first** using the existing `image_caption` route with a steering prompt — no
server change — and C2/C3 land when `/face-embed` is ready.

## 4. New code in this repo

```
client/
  face_recognition.py        FaceRecognitionClient.embed(pil) -> list[FaceEmbedding]
                             (register in client/__init__.py + walkie_client.py)
perception/
  people_store.py            PeopleStore — ChromaDB collection "people", keyed by face
                             embedding. enroll(name, drink, emb, attrs) / recognize(emb)
                             / get(name) / list_people(). Cosine knn; threshold
                             FACE_MATCH_THRESHOLD. NO position dedup, NO spatial prune.
  gestures.py                (deferred, C7) lift _summarize_pose out of vision tools
agents/human_agent/
  __init__.py                create_human_agent(...) factory (copy vision_agent shape)
  prompts.py                 system prompt — enforce the "speak-only" contract
  tools.py                   the tool surface (below)
```

`PeopleStore` deliberately reuses the `SceneStore` patterns that still apply
(ChromaDB `PersistentClient`, the `{"success"}` unwrap, caption-led text) but **drops**
the spatial machinery (dedup radius, prune-by-location, position lift).

### Tool surface (`agents/human_agent/tools.py`)

| Tool | Decorator | Does | Backed by |
|---|---|---|---|
| `enroll_person(name, drink)` | sequential | capture frame, embed largest face, store name+drink+attrs | C2 |
| `recognize_person()` | parallelable | embed faces in view, knn vs store, return names | C3 |
| `describe_person()` | parallelable | steered caption of the person in view | C4 |
| `count_people()` | parallelable | how many people + seated/standing split | C1 |
| `find_empty_seat()` | parallelable | chairs/sofas (scene store) minus occupied | C5 |
| `speak(text)` | sequential | TTS (same as every agent) | — |

`enroll_person` is **sequential** (it's a stateful write + we want it to block);
read-only lookups are **parallelable**, per the convention in CLAUDE.md.

## 5. Wiring

1. `main.py:build...` constructs `FaceRecognitionClient` and `PeopleStore`, passes them
   into `run_ready_stage` like `scene_store` is today.
2. `run_ready_stage` builds the human agent via `create_human_agent(...)`.
3. `agents/walkie_agent/tools.py` gains `delegate_to_human` (sequential — it invokes a
   sub-graph), and the walkie prompt learns when to route to it
   ("greet/remember/introduce a person" → human; "what's in front of me" → vision;
   "where did I see an object" → database).

No background loop is needed for Receptionist — enrollment/recognition are **on-demand**
(triggered by the conversation), unlike the always-on scene loop. The people store is
small (a handful of guests) and lives only for the run; durability is out of scope.

## 6. Config (`config.toml`, new `[human]` table)

Keys are the literal env-var names (per the config convention). Proposed:

```
FACE_EMBED_BACKEND      remote        # only remote for now (server-side model)
PEOPLE_CHROMA_DIR       chroma_db_people
PEOPLE_FRAMES_DIR       people_frames
FACE_MATCH_THRESHOLD    0.45          # cosine distance; tune on real faces
FACE_MIN_DET_SCORE      0.5
HUMAN_DESCRIBE_PROMPT   "Describe this person: clothing colors, hair, glasses, posture."
```

## 7. Deferred (after Receptionist)

- **C6 person tracking / gaze** — track one person across frames and emit a target
  the head/base can aim at, for "look at the person talking / the correct guest"
  (2×50 + 2×50 pts). Reuses `pose_estimation` bboxes; the open part is the
  head/camera control surface (actuator side), not recognition.
- **C7 gestures** (`perception/gestures.py`): wave / point-left-right / sit-stand-lie
  from pose keypoints → unlocks Restaurant (waving) and GPSR counting.
- **People counting with filters** ("how many men", "how many pointing left") → GPSR.
- **Person following the host** — the HRI task's bag-drop phase ("Following the host
  to the bag drop area" = 200 pts) needs follow = person tracking + nav loop. Blocked
  on the unreliable 3D lift (`get_3d_poses`) — see the scene-position notes;
  do this last. (Note: there is **no** Carry My Luggage task in the 2026 rulebook —
  following lives inside the HRI/Receptionist task itself.)

## 8. Open questions

- Server: is InsightFace/ArcFace acceptable for `/face-embed`, or reuse an existing
  model already loaded there?
- Re-ID robustness across lighting/pose at a party — may need >1 enrollment frame.
- Empty-seat detection: pure geometry (chair bbox vs person bbox overlap) or ask the
  VLM ("which chairs are empty")?
- Gaze/tracking (C6): which actuator surface aims the head/camera at the tracked
  person? Needs a target-point contract with the actuator agent / SDK.

### Resolved by the rulebook

- **Gender/age not needed.** §5.1 scores *"Tell a visual attribute"* (4×±20) — a
  single correct attribute. A clothing/posture description from `describe_person`
  satisfies it; avoid risky gender/age guesses (a wrong attribute is −20).
- **Favorite drink is a free-form string**, "any english named popular drink … may
  not be in the list of objects" — capture it from STT verbatim, don't map it to an
  object class.
- **Identity must be face-based** — "guests may switch seats" rules out position;
  "Not recognizing people" is −200×2.
```
