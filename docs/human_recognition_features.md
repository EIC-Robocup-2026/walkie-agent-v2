# Human-recognition features — summary & test guide

Everything the **Human (HRI) sub-agent** can do, for the RoboCup @Home
Receptionist task, plus how to test each on the real robot. Built on branch
`feat/human-recognition`; server side on `feat/face-recognition-service`.

The agent reaches these via the main Walkie agent's `delegate_to_human` tool.
"Live camera only" means each call captures one fresh frame.

---

## 1. Feature catalogue

| # | Tool | What it does | Needs (walkie-ai-server) | Offline test |
|---|---|---|---|---|
| C4 | `describe_person(focus=None)` | One-line description of a person (clothing, hair, glasses, posture) for the spoken introduction. `focus` steers toward one of several people. | image-caption | shape only |
| C1 | `count_people()` | Count visible people + how many have an arm raised (wave) + approx sitting/standing split. | pose-estimation | ✅ heuristics |
| C2 | `enroll_person(name, drink)` | Remember the guest up front: face → embedding, bound to name + favorite drink. Archives a face crop for the viewer. Re-enroll same name = refresh + centroid. | /face-recognition/embed | ✅ flow (fake) |
| C3 | `recognize_person()` | Match face(s) in view to remembered guests → name + drink, or "unknown". | /face-recognition/embed | ✅ flow (fake) |
| C3 | `list_known_people()` | Recall every remembered guest + their drink (for introductions). | – | ✅ |
| C5 | `find_empty_seat()` | Seats (chair/couch/sofa/bench) in view that no one sits on, with a direction. | object-detection + pose | ✅ geometry |
| C6 | `locate_person(name=None)` | Direction + approx turn to face a person — by face (named) or nearest (unnamed). | pose (unnamed) / face (named) | ✅ geometry |
| — | `speak(text)` | TTS out loud (used sparingly; the main agent usually speaks). | tts | — |

**Supporting infrastructure**
- `client/face_recognition.py` — `FaceRecognitionClient.embed / info / providers`.
- `perception/people_store.py` — `PeopleStore`: face-keyed cosine memory
  (`enroll / recognize / get / list_people / count / clear`), face-crop archive.
- `main.py::build_people_store` — builds the store, probes the face model, wires
  it into the human agent and the DB viewer. Graceful if the route is down.
- **DB viewer** (`:8500`) — the `people` collection shows name / favorite drink /
  enrollments columns + each guest's face thumbnail.

**What is NOT here** (other agents / future): walking to the door, guiding,
continuous gaze-follow loop, bag handover, follow-the-host, doorbell. `find_empty_seat`
and `locate_person` only *report* a target — the actuator does the turning/pointing.

---

## 2. Prerequisites for live testing

1. **walkie-ai-server** up at `WALKIE_AI_BASE_URL` with these routes enabled:
   `/face-recognition/*`, `/image-caption`, `/object-detection`, `/pose-estimation`,
   `/tts`. (Install `insightface onnxruntime-gpu` for the face route.)
2. Robot **camera** reachable (or a local webcam).
3. `uv sync`, `.env` set (`OPENROUTER_API_KEY`, `WALKIE_AI_BASE_URL`).
4. Start typed mode so you can drive without the mic:
   ```bash
   DISABLE_LISTENING=1 uv run python main.py
   ```
   Expect on startup: `[human] face recognition backend: insightface-buffalo_l`
   and `[human] people memory ON (N remembered)`. Open `http://<ip>:8500`.

> If you see `[human] face service probe failed …`, the face route is unreachable —
> describe/count still work, but enroll/recognize will report the error at call time.

---

## 3. Manual test script (type each line at the prompt)

You type natural commands to the **main** agent; it should delegate to the human
agent. Watch the console for `[walkie] -> human:` and the viewer for new faces.

| Step | Say / type | Expect |
|---|---|---|
| 1 | "How many people do you see?" | a count + arm-raised + posture line |
| 2 | "Describe the person in front of you." | a short clothing/posture description |
| 3 | "Remember this guest. Her name is Alice and she likes cola." | confirmation "Remembered Alice …"; a face row appears in the viewer's `people` collection |
| 4 | (have a 2nd person stand in) "Remember this guest, John, he likes water." | 2nd face row in the viewer |
| 5 | "Who is this?" (point camera at Alice) | "Alice, favorite drink 'cola'" with a match score |
| 6 | (point at a stranger) "Who is this?" | "unknown (not a remembered guest)" |
| 7 | "Who have you met so far?" | lists Alice + John with their drinks |
| 8 | (Alice & John swap seats) "Who is sitting where?" | still correct by face, not by seat |
| 9 | "Find an empty seat." | a free seat + direction ("to your right …") |
| 10 | "Where is Alice / look at Alice." | direction + approx turn to face her |
| 11 | "Tell John what Alice looks like." | recalls + describes (intro rehearsal) |

**Pass criteria:** steps 5–8 are the core re-ID — same person recognized across
frames/seats, strangers flagged unknown, never a wrong name.

---

## 4. Tuning knobs (`config.toml [human]`) — set these on real data

| Key | Default | Tune when |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | `0.4` | **most important.** Cosine distance. Same person should land well below it, different people well above. Read the match scores from `recognize_person` / the server test to pick it. |
| `CAMERA_HFOV_DEG` | `70` | set to the camera's real horizontal FOV, or `locate_person` angles are wrong. |
| `FACE_MIN_DET_SCORE` | `0.5` | raise if it grabs non-faces; lower if real faces are missed. |
| `SEAT_CLASSES` | `chair,couch,sofa,bench` | match what the detector actually labels seats. |
| `SEAT_OCCUPANCY_RATIO` | `0.2` | raise if standing-nearby people wrongly mark a seat taken. |
| `HUMAN_DESCRIBE_PROMPT` | (see file) | reword the description steering if captions miss the useful attributes. |
| `HUMAN_PERCEPTION_ENABLED` | `1` | `0` to skip building people memory entirely. |
| `PEOPLE_CHROMA_DIR` / `PEOPLE_FRAMES_DIR` | `chroma_db_people` / `people_frames` | storage locations (gitignored). |

Wipe remembered people between rehearsals: stop the robot, `rm -rf chroma_db_people
people_frames` (or call `PeopleStore.clear()`), then restart.

---

## 5. Automated coverage (already green, offline)

`uv run pytest` — 122 tests. Human-recognition ones:
- `tests/test_face_recognition_client.py` — face client deserialization, errors, cache.
- `tests/test_people_store.py` — enroll/recognize, centroid, threshold, frame archive.
- `tests/test_human_tools.py` — pose heuristics (arm-raised, sitting/standing).
- `tests/test_human_face_tools.py` — enroll→recognize flow, stranger=unknown, graceful paths.
- `tests/test_human_seat_gaze_tools.py` — empty-seat geometry, locate direction.

These mock the camera + server, so they verify **logic**, not the models. On-robot
testing (§3) is what validates detection/recognition/threshold against real faces.
