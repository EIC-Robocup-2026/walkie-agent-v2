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

## 2. Receptionist task → required capabilities

The Receptionist task: greet guests at the door one at a time; ask each guest's
**name** and **favorite drink**; lead them in; **introduce** the new guest to the host
and already-seated guests (say their name, drink, and one described attribute); then
find an **empty seat** and offer it.

| # | Capability | Used for | New work |
|---|---|---|---|
| C1 | **Person detection + count** | "is someone at the door", "who's seated" | thin — reuse `pose_estimation` bboxes |
| C2 | **Face enroll** (face → embedding, bind name + drink) | remember each guest | server `/face-embed` + `people_store` |
| C3 | **Face recognize** (embedding → known person) | introduce / point out who is who | `people_store` knn |
| C4 | **Attribute description** (clothing, hair, glasses, posture; maybe gender/age) | the spoken introduction | `image_caption` w/ steering prompt, or a VLM tool |
| C5 | **Empty-seat finding** | "please take a seat" | scene store (chairs/sofas) ∩ person occupancy |
| C6 | Gesture/posture (wave, point, sit/stand) | *not* Receptionist — GPSR/CML | deferred (see §7) |

C2/C3 are the only genuinely new heavy-model dependency. C1/C4/C5 reuse what exists.

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
  gestures.py                (deferred, C6) lift _summarize_pose out of vision tools
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

- **C6 gestures** (`perception/gestures.py`): wave / point-left-right / sit-stand-lie
  from pose keypoints → unlocks Restaurant (waving) and GPSR counting.
- **People counting with filters** ("how many men", "how many pointing left") → GPSR.
- **Person following** (Carry My Luggage): tracking + nav loop. Blocked on the
  unreliable 3D lift (`get_3d_poses`) — see the scene-position notes; do this last.

## 8. Open questions

- Server: is InsightFace/ArcFace acceptable for `/face-embed`, or reuse an existing
  model already loaded there?
- Re-ID robustness across lighting/pose at a party — may need >1 enrollment frame.
- Empty-seat detection: pure geometry (chair bbox vs person bbox overlap) or ask the
  VLM ("which chairs are empty")?
- Do we need gender/age estimation for the introduction, or is a clothing/posture
  description enough to satisfy the task's "describe" requirement?
```
