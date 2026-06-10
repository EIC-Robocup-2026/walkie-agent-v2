# Handoff: appearance (attire) re-ID service for `walkie-ai-server`

**Audience:** a Claude (or human) working in the **`walkie-ai-server`** repo.
**Goal:** add **one** new HTTP service — appearance/attire embedding — so the robot
brain (`walkie-agent-v2`) can re-identify a person **whose face is not visible**
(turned away, far, occluded). This is the second modality of the people memory; the
face service (`/face-recognition/*`) already exists and needs no change.

> **Credit:** the pipeline this service hosts was designed and validated **by Chalk
> (EIC team)** in the `eic-human` subproject. Use his
> `eic_human/pipeline/appearance.py` as the reference implementation — it is the
> exact model + preprocessing this spec describes (OSNet x1.0 via `torchreid`,
> 512-d L2-normalized output).

---

## Why this is needed (one paragraph of context)

Face recognition alone fails the moment a guest faces away from the robot — and in
Receptionist/Restaurant the robot routinely approaches people from behind or from the
side. Chalk's `eic-human` work showed that fusing a face embedding with an
**appearance embedding** (clothing, body shape — OSNet, a person re-ID model) keeps
identification working in exactly those cases: within one session a guest's attire is
stable, so the attire vector is a reliable secondary key. The agent stores both vectors
per guest and adaptively weights them by face-detection confidence (Chalk's fusion
design, now in `perception/people_store.py::recognize_fused`).

---

## What to build: `POST /appearance/embed`

One person crop in → one fixed-length L2-normalized appearance embedding out. No
storage, no matching on the server — the server is **stateless**; the agent owns all
memory and matching (same contract philosophy as `/face-recognition/embed`).

### Request

`multipart/form-data`, identical shape to the existing routes:

| field | type | notes |
|---|---|---|
| `image` | file (JPEG) | a **person crop** (the agent crops to the person bbox before sending), e.g. `("image.jpg", bytes, "image/jpeg")` |

### Response

Wrapped in the **standard envelope** every route on this server uses:

```jsonc
{
  "success": true,
  "data": {
    "embedding": [/* N floats, L2-normalized, ‖v‖₂ = 1 */]
  }
}
```

**Hard contract points (the agent depends on these):**

1. **`embedding` is L2-normalized** (unit length) and a **constant dimension** for
   every call (512 for OSNet x1.0). The agent uses cosine similarity directly.
2. The model embeds **whatever image it is given** — do not detect/crop people on the
   server. The agent sends the crop. (A full frame is still a valid input; it just
   embeds the whole scene.)
3. A malformed request (missing `image`, undecodable bytes) → `success: false` +
   `"error": "<message>"` so the client raises `WalkieAPIError` instead of crashing.

### Companion route: `GET /appearance/info`

```jsonc
{ "success": true, "data": { "model_name": "osnet_x1_0", "dim": 512 } }
```

Lets the agent stamp stored vectors with their producing model so a future model swap
is detectable. Cheap; ship it.

---

## Recommended model: OSNet via torchreid (no training)

Exactly what Chalk's reference implementation uses:

```bash
pip install numpy cython && pip install --no-build-isolation \
    git+https://github.com/KaiyangZhou/deep-person-reid.git
```

> ⚠️ `deep-person-reid` has a PEP 517 build-isolation bug — install numpy+cython
> first and pass `--no-build-isolation`, as above (Chalk documented this in
> `eic-human/setup.py`).

Reference sketch (adapt to this repo's blueprint/registration style; the core is
verbatim Chalk's `AppearancePipeline`):

```python
# appearance.py  (new blueprint) — pipeline by Chalk (EIC), eic_human/pipeline/appearance.py
from flask import Blueprint, request
import numpy as np, cv2
import torch
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor

bp = Blueprint("appearance", __name__)

_extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path="",            # auto-downloads pretrained weights
    device="cuda" if torch.cuda.is_available() else "cpu",
)
_MODEL_NAME, _DIM = "osnet_x1_0", 512

@bp.post("/appearance/embed")
def embed():
    file = request.files.get("image")
    if file is None:
        return {"success": False, "error": "missing 'image' file field"}, 400
    buf = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return {"success": False, "error": "could not decode image"}, 400
    img_rgb = img[:, :, ::-1].copy()           # torchreid expects RGB
    with torch.no_grad():
        feats = _extractor([img_rgb])          # (1, 512)
        emb = F.normalize(feats, p=2, dim=1)[0].cpu().numpy()
    return {"success": True, "data": {"embedding": emb.astype(float).tolist()}}

@bp.get("/appearance/info")
def info():
    return {"success": True, "data": {"model_name": _MODEL_NAME, "dim": _DIM}}
```

---

## How the agent calls it (contract is frozen here)

```python
emb = walkieAI.appearance.embed(person_crop)   # list[float], 512-d unit length
# enroll: stored alongside the face vector in PeopleStore (people_appearance)
# recognize: fused with the face score (recognize_fused), or appearance-only
#            when no face is visible
```

The agent applies its own thresholds (`APPEARANCE_MATCH_THRESHOLD`) and the
confidence-adaptive fusion weights — the server does **not** threshold or match.

---

## Done = these pass

1. `POST /appearance/embed` with a person photo → `len(embedding)` equals the
   advertised dim; `abs(norm(embedding) - 1.0) < 1e-3`.
2. Two crops of the **same clothed person** (different angles) → cosine similarity
   clearly **higher** than two **different people** in different clothes.
3. Missing/undecodable `image` → `success: false` with an `error` message (not a 500).
4. `GET /appearance/info` → `{model_name, dim}`.

---

## Out of scope (the agent side handles all of this — do NOT build it on the server)

- Person detection / cropping (the agent crops via its pose-estimation route).
- Storing identities, fusion scoring, thresholds, the people database.
- Face anything — `/face-recognition/*` already exists.

The server's whole job here is: **person crop in → one normalized attire vector out.**
