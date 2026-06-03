# Handoff: face-recognition service for `walkie-ai-server`

**Audience:** a Claude (or human) working in the **`walkie-ai-server`** repo.
**Goal:** add **one** new HTTP service — face detection + face embedding — so the
robot brain (`walkie-agent-v2`) can remember and re-identify people for the RoboCup
@Home **HRI / Receptionist** task. Everything else that task needs (STT, TTS, object
detection, image captioning, pose estimation) **already exists** on this server and
needs no change.

This is the only server-side work required. Once it's live, the agent repo wires up a
thin client, a face-keyed people store, and the `enroll_person` / `recognize_person`
tools on its side.

---

## Why this is needed (one paragraph of context)

In the Receptionist task the robot greets two guests, learns each guest's **name +
favorite drink**, then later **introduces them to each other** — which requires knowing
*which seated person is which guest*. The rulebook says **"guests may switch seats"**
and penalizes **"Not recognizing people" by −200 (×2)** — the heaviest penalty in the
task. So identity must be matched by **face**, not by position. A face embedding turns
a face crop into a vector; the agent stores `name+drink ↔ vector` and later finds the
nearest stored vector for whoever is in view.

---

## What to build: `POST /face-recognition/embed`

Detect every face in an image and return, per face, a **bounding box**, a
**fixed-length L2-normalized embedding**, and a **detection score**. No enrollment DB,
no names, no matching on the server — the server is **stateless**; the agent owns all
memory and matching.

### Request

`multipart/form-data`, identical shape to the existing `/object-detection` and
`/pose-estimation/estimate` routes:

| field | type | notes |
|---|---|---|
| `image` | file (JPEG) | the frame to analyze, e.g. `("image.jpg", bytes, "image/jpeg")` |

### Response

Wrapped in the **standard envelope** every route on this server already uses — the
agent client unwraps `body["data"]` and raises `WalkieAPIError` when `success` is false
(see `client/base.py::_unwrap` in the agent repo):

```jsonc
{
  "success": true,
  "data": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],   // ints, pixel coords (top-left, bottom-right)
      "embedding": [/* N floats, L2-normalized, ‖v‖₂ = 1 */],
      "det_score": 0.99                 // face-detection confidence, [0, 1]
    }
    // ... one object per detected face, or [] if none
  ]
}
```

**Hard contract points (the agent depends on these):**

1. **`bbox_xyxy` is `[x1, y1, x2, y2]`** (NOT `cxcywh`). Be explicit — the rest of this
   ecosystem mixes both conventions, so the field name pins it.
2. **`embedding` is L2-normalized** (unit length) and a **constant dimension** for every
   face and every call (e.g. 512). The agent uses **cosine distance**; pre-normalizing
   means it never has to. Don't change the dimension at runtime.
3. **No face → `"data": []` with `success: true`.** An empty frame is not an error.
4. **Return ALL faces**, unsorted is fine. The agent picks the largest bbox when it
   needs "the one person being enrolled," and filters by `det_score` itself.
5. A malformed request (missing `image`, undecodable bytes) → `success: false` +
   `"error": "<message>"` so the client raises `WalkieAPIError` instead of crashing.

### Recommended companion route: `GET /face-recognition/info`

Mirrors `/pose-estimation/providers`. Lets the agent record *which* model produced a
stored vector (so a future model swap can be detected) without hardcoding it:

```jsonc
{ "success": true, "data": { "model_name": "insightface-buffalo_l", "dim": 512 } }
```

Not strictly required to ship, but cheap and saves a migration headache later.

---

## Recommended model: InsightFace (no training)

Use **InsightFace `buffalo_l`** — RetinaFace detector + ArcFace (ResNet-100) recognizer.
Pip-installable, GPU via onnxruntime, gives a **512-d already-normalized** embedding as
`face.normed_embedding`. No fine-tuning or dataset needed.

```bash
pip install insightface onnxruntime-gpu   # or onnxruntime for CPU
```

Reference implementation sketch (adapt to this repo's blueprint/registration style):

```python
# face_recognition.py  (new blueprint)
from flask import Blueprint, request
import numpy as np, cv2
from insightface.app import FaceAnalysis

bp = Blueprint("face_recognition", __name__)

_app = FaceAnalysis(name="buffalo_l")          # loads detector + recognizer
_app.prepare(ctx_id=0, det_size=(640, 640))    # ctx_id=0 → GPU; -1 → CPU
_MODEL_NAME, _DIM = "insightface-buffalo_l", 512

@bp.post("/face-recognition/embed")
def embed():
    file = request.files.get("image")
    if file is None:
        return {"success": False, "error": "missing 'image' file field"}, 400
    buf = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)   # BGR, as InsightFace expects
    if img is None:
        return {"success": False, "error": "could not decode image"}, 400
    faces = _app.get(img)
    data = [
        {
            "bbox_xyxy": [int(v) for v in f.bbox],          # already xyxy
            "embedding": f.normed_embedding.tolist(),       # 512-d, unit length
            "det_score": float(f.det_score),
        }
        for f in faces
    ]
    return {"success": True, "data": data}

@bp.get("/face-recognition/info")
def info():
    return {"success": True, "data": {"model_name": _MODEL_NAME, "dim": _DIM}}
```

Then **register the blueprint** the same way the other services are wired in this repo
(e.g. alongside `app.register_blueprint(image_embed.bp)` — match the existing pattern,
whatever it is here):

```python
from blueprints import face_recognition          # wherever blueprints live
app.register_blueprint(face_recognition.bp)
```

---

## How the agent will call it (contract is frozen here)

For reference — this is the client the agent repo will add (`client/face_recognition.py`),
so the response shape above must hold:

```python
poses = walkieAI.face_recognition.embed(pil_image)
# -> list of FaceEmbedding(bbox_xyxy=(x1,y1,x2,y2), embedding=[...512], det_score=0.99)
# enroll: pick the largest-bbox face, store name+drink ↔ embedding
# recognize: embed faces in view, cosine-knn against the stored vectors
```

The agent applies its own thresholds (`FACE_MIN_DET_SCORE`, a cosine-distance
`FACE_MATCH_THRESHOLD`) — the server does **not** threshold or match.

---

## Done = these pass

1. `POST /face-recognition/embed` with a clear single-face photo → `data` has 1 item;
   `len(embedding)` equals the advertised dim; `abs(norm(embedding) - 1.0) < 1e-3`.
2. Two photos of the **same** person → cosine distance **small** (≈ < 0.4).
   Two **different** people → cosine distance **large** (≈ > 0.4). (Exact threshold is
   tuned on the agent side; just confirm same < different by a clear margin.)
3. A photo with **no face** → `success: true`, `data: []`.
4. Missing/!decodable `image` → `success: false` with an `error` message (not a 500
   stack trace).
5. `GET /face-recognition/info` → `{model_name, dim}` (if you ship it).

---

## Out of scope (the agent side handles all of this — do NOT build it on the server)

- Storing names / favorite drinks, the people database, matching/recognition logic.
- Attribute description ("blue shirt, glasses") — done already via the existing
  `image_caption` route with a steering prompt.
- People counting / pose / gestures — the existing `pose_estimation` route covers it.
- Person tracking, gaze, navigation, the bag handover.

The server's whole job here is: **image in → faces (box + normalized vector + score) out.**
