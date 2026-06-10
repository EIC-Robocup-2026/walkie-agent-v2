# `eic-human` review & adoption notes

**Pipeline by Chalk (EIC team).** This documents what `walkie-agent-v2` adopted from
Chalk's `eic-human` subproject (human identity pipeline: face + appearance re-ID),
how it maps onto this repo's architecture, and review feedback on the subproject
itself.

## What the pipeline is

`eic-human` is a self-contained human identity database:

- **Face**: InsightFace `buffalo_l` (SCRFD detect + ArcFace embed, 512-d), with
  facenet-pytorch and YOLOv8-face as alternative backends (`pipeline/face.py`).
- **Appearance**: OSNet x1.0 person re-ID embeddings via `torchreid` — clothing/body
  shape, 512-d (`pipeline/appearance.py`). This is the capability our stack lacked:
  re-identifying someone **whose face is not visible**.
- **Fusion** (`core.py::_fuse_score`): adaptive weighting by face-detection
  confidence — high-confidence face → 0.75 face / 0.25 appearance; medium → 0.55 /
  0.45; no face → appearance only.
- **Store**: SQLite + sqlite-vec, one row per identity, plus restaurant order
  tracking and a Gradio UI server.

## What we adopted (and where it went)

| Chalk's design | Landed in |
|---|---|
| OSNet appearance embedding as a second identity modality | `client/appearance.py` (HTTP client) + `docs/walkie_ai_server_appearance_service.md` (server handoff — hosts Chalk's `pipeline/appearance.py` verbatim) |
| Confidence-adaptive face↔appearance fusion (weights + thresholds) | `perception/people_store.py::recognize_fused` + `FUSION_DEFAULTS` |
| Appearance-only fallback when no face is visible | `agents/human_agent/tools.py::recognize_person` (flagged "less certain" in the spoken result) |
| Latest-wins appearance storage (clothing is session-specific) | `PeopleStore.enroll(app_embedding=...)` → `people_appearance` collection |
| Match-score → confidence tiers and the 0.5 default match floor | `APPEARANCE_MATCH_THRESHOLD` in `config.toml` |

## What we kept from our side (and why)

- **Models stay on `walkie-ai-server`.** `eic-human` loads torch/InsightFace/torchreid
  in-process; our robot brain is a thin client on a separate machine from the GPU box,
  so OSNet is hosted behind `/appearance/embed` exactly like the existing face route.
  The agent never imports torch for this.
- **ChromaDB `PeopleStore`, not sqlite-vec.** The people memory predates this
  integration, is covered by tests, feeds the DB viewer, and follows the repo's
  two-collections-one-id-space pattern (`SceneStore`). Only the *scoring logic*
  moved over, onto the existing backend.
- **Face centroid enrollment.** Our store folds repeat enrollments into a running
  centroid; `eic-human` replaces on re-enroll. Centroid is more robust across
  lighting/pose for the face; for appearance we follow Chalk (latest wins).
- **One fusion implementation.** `eic-human` has the scoring twice (`core.py` and
  `pipeline/store.py`, with drifted weights); we implemented it once in the store.

## Review feedback on `eic-human` (for Chalk)

Found while reading the code — worth fixing in the subproject:

1. **`core.py` `NameError`s in cold paths**: `_migrate_legacy_pt` uses `_torch`
   (line ~200) and the facenet fallback uses `_torch`/`_F` (lines ~293–295), but the
   module imports them as `torch` / `F`. Both paths crash the moment they run.
2. **`install.sh` unquoted version spec**: `pip install numpy>=1.23.0 ...` — the
   shell parses `>=1.23.0` as an output redirection, so it installs unpinned `numpy`
   and creates a file named `=1.23.0`. Quote it: `pip install "numpy>=1.23.0"`.
3. **`face.py::_run_facenet_all` builds a new `MTCNN` per call** — a model
   construction on every frame. Hoist into `_init_facenet` (a second `keep_all=True`
   instance) or cache it.
4. **NULL embeddings break `find()`**: enrolling someone whose face wasn't detected
   stores `face_emb = NULL`; `vec_distance_cosine(face_emb, ?)` over that row makes
   the whole query error. Guard with `WHERE face_emb IS NOT NULL` per modality.
5. **`yolov8` backend gaps**: `detect_and_embed_all` silently falls through to the
   facenet branch for `backend="yolov8"`, and the weights download uses
   `os.system(wget ...)` (no error check). Also `ultralytics` is a heavy hard dep
   that only this optional backend needs.
6. **Duplicated scoring** between `HumanDatabase` and `VectorStore` (fixed 0.65 face
   weight in one, adaptive in the other) — pick one home, have the other delegate.

None of these affect what we adopted — the InsightFace path, the OSNet pipeline, and
the fusion design are solid, and the adaptive-confidence weighting is a genuinely
nice idea that our face-only stack didn't have.
