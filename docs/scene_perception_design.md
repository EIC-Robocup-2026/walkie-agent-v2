# Scene Embedding & Background Perception — Phase 1 + 2 + 3

Branch: `feat/scene-perception`
Status: **Phase 3 implementation landed.** Module at `perception/`, 42 tests passing. See "Phase 3 — Implementation summary" at the bottom.

> **Addendum (motion + dedup behavior) folded in.** The robot moves, the loop runs indefinitely, re-observation is the norm. Sections 2 (Dedup) and the test plan have been updated to match. Changes in the addendum that conflicted with the original draft are called out inline as **[Addendum]**.

## TL;DR

`perception/` is a Python package that runs an always-on async loop, calls the AI server for detection + caption + (eventually) CLIP embeddings, calls the walkie-sdk `Tools.bboxes_to_positions` to lift each detection to a 3D world-frame position, and upserts results into a single ChromaDB collection. Queries (semantic / spatial / recency / diff) are served from the same collection.

**Phase 3 implementation is complete and tested** (42 passing tests, ~20s suite). The one remaining production blocker is CLIP: walkie-ai-server has a CLIP embedding service implemented but the route is commented out — the code is ready for it via an `Embedder` Protocol; the real client lands the day the server enables the blueprint.

## Document map

- **Phase 1 — Discovery findings** — what we read in the two upstream repos, the exact APIs we depend on, the missing capability.
- **Phase 2 — Design** — schema, dedup strategy, query API, retention. `[Addendum]` tags mark sections rewritten after the motion/dedup addendum.
- **Open questions** — four design decisions, with Phase 3 status notes for each.
- **Phase 3 — Implementation summary** — module manifest, test inventory, follow-ups.

---

## Phase 1 — Discovery findings

### Where the code lives on disk

| What | Path | Notes |
|---|---|---|
| Consumer (this repo) | `/home/hextex/Documents/GitHub/walkie-agent-v2/` | `services/perception.py` + `services/explore.py` already do a *simpler* version of this — both will be replaced by the new module |
| walkie-sdk source | resolved via `uv sync` to commit `025ee9b` (`walkie-sdk==0.2.0`) in `.venv/lib/python3.12/site-packages/walkie_sdk/` | GitHub `main` HEAD has the full surface — `arm/camera/tools/multi_camera/visualization` all present. The local `/home/hextex/Documents/GitHub/Walkie-SDK/` clone is stale (only 3 commits, missing the post-init work) and was misleading during discovery. `pyproject.toml` declares the git source without a ref, but `uv.lock` pins the exact commit so reproducibility is intact as long as the lockfile is committed. |
| walkie-ai-server source | `/home/hextex/Documents/GitHub/walkie-ai-server/` | Flask app on port 5000 |

### walkie-sdk APIs we will use

All call signatures verified against `walkie_sdk/modules/`:

```python
WalkieRobot(
    ip: str,
    ros_protocol: str = "rosbridge" | "zenoh",
    ros_port: int = 9090,
    camera_protocol: str = "webrtc" | "zenoh" | "shm" | "none",
    camera_port: int = 8554,
    timeout: float = 10.0,
    namespace: str = "",
)

bot.camera.get_frame() -> np.ndarray | None        # BGR HxWx3 uint8, latest cached frame, non-blocking
bot.camera.is_streaming -> bool
bot.camera.frame_shape -> (H, W, C) | None

bot.status.get_position() -> {"x": float, "y": float, "heading": float} | None  # heading in radians; despite the name, the payload includes orientation
bot.status.get_velocity() -> {"linear": float, "angular": float} | None

bot.tools.bboxes_to_positions(
    coords: list[list[float]],   # each [cx, cy, w, h] in image-pixel coords
    timeout: float = 5.0,
) -> list[list[float]] | None    # each [x, y, z] in the upstream YOLO-3D frame (usually `map`); None on timeout
```

`bboxes_to_positions` is a **pub/sub request-reply**: publishes a `vision_msgs/Detection2DArray` on `/yolo/detections_2d`, waits up to `timeout` seconds for a `geometry_msgs/PoseArray` on `/ob_detection/poses`, then maps poses back to a list of `[x,y,z]`. Output order is aligned with input order. Returns `None` if no response arrived in time.

**Telemetry naming oddity (not a bug, just worth knowing)**: the method is `get_position()` but its payload includes `heading` — so it's really a 2D *pose* (position + orientation) wearing a position label. The SDK used to expose this as `get_pose()` in earlier versions; it was renamed to `get_position()` at some point before commit `025ee9b`. The consumer code in `agents/actuator_agent/tools.py:24` calls the current method correctly. If you see `get_pose` anywhere in old docs or branches, treat it as the same call.

### walkie-ai-server endpoints we will use

Verified against `walkie-ai-server/api/routes/`:

| Capability | Endpoint | Request | Wrapper in this repo | Returns |
|---|---|---|---|---|
| Object detection | `POST /object-detection/detect` | multipart `image` | `client.ObjectDetectionClient.detect(img)` | `list[DetectedObject(bbox=(x1,y1,x2,y2), class_name, confidence, area_ratio, mask=None)]` |
| Image captioning | `POST /image-caption/caption` | multipart `image`, form `prompt?` | `client.ImageCaptionClient.caption(img, prompt=…)` | `str` |
| Image captioning (batch) | `POST /image-caption/caption-batch` | multipart `images[]`, form `prompts[]?` | `client.ImageCaptionClient.caption_batch(imgs, …)` | `list[str]` |
| Pose estimation | `POST /pose-estimation/estimate` | multipart `image` | `client.PoseEstimationClient.estimate(img)` | `list[PersonPose(bbox, confidence, keypoints=17×COCO)]` |
| STT / TTS | `/stt/transcribe`, `/tts/synthesize{,-stream}` | — | — | not used by the perception subsystem |

All responses are wrapped in `{"success": bool, "data": …}`; `client/base.py::_unwrap` peels that off and raises `WalkieAPIError` on failure.

### Missing capability — **DECISION REQUIRED**

`walkie-ai-server/services/image_embed/` ships a complete CLIP provider (`openai/clip-vit-base-patch16`, 512-dim, image and text embeddings, cosine similarity). The route file `api/routes/image_embed.py` exposes:

- `POST /image-embed/embed-image` → `{embedding, dim}`
- `POST /image-embed/embed-text` → `{embedding, dim}`
- `POST /image-embed/similarity` → `{similarity}`

**but `api/__init__.py` has the blueprint commented out** (line 16: `# app.register_blueprint(image_embed.bp)`). The endpoint will return 404 today.

Two options before we start Phase 3:

1. **Re-enable the blueprint on the AI server.** One-line change in `walkie-ai-server/api/__init__.py`, plus a deploy. We get joint image+text embeddings out of the box, semantic queries match captions *and* visual content, and the cross-modal `similarity` endpoint becomes the natural backbone of the "show me the X" query.
2. **Embed locally on the agent side.** Drop a `sentence-transformers` model (e.g. `all-MiniLM-L6-v2`, 384-dim) into this repo and embed *only the caption text*. Queries match the caption surface but lose the visual half of CLIP. Smaller deployment footprint, slower per tick (~30–80ms CPU for batched captions), and we lose "find an image by image" capability.

**Recommendation: option 1.** It's free, the model is already wired up, and CLIP image+text in the same space is the right primitive for this problem. The smoke test `tests/perception/test_smoke_image_embed.py` already pins the contract we'll consume.

---

## Phase 2 — Design

### Module layout

```
perception/                       (replaces services/perception.py + services/explore.py once main.py is migrated)
├── __init__.py    # public re-exports
├── loop.py        # async background loop (cancellable, configurable rate)
├── pipeline.py    # 1 frame → list[Detection] (calls AI server + walkie-sdk)
├── store.py       # SceneStore (ChromaDB wrapper: upsert, dedup, queries)
├── dedup.py       # pure-function decisions: classify() + threshold getters
├── types.py       # SceneEntry, Detection, DedupDecision, SceneDiff, TickReport + Protocols
└── mocks.py       # FakeCamera, FakeDetector, FakeCaptioner, FakeEmbedder, FakePositionLifter (test-only)
```

Six modules, no inheritance, composed by `loop.py`. Each module is independently unit-testable; Protocols in `types.py` let mocks be drop-in substitutes for the real collaborators.

### ChromaDB schema — one collection, `scene_entries`

```python
chroma_client.get_or_create_collection(
    name="scene_entries",
    metadata={"hnsw:space": "cosine"},
)
```

| Chroma field | Contents | Why |
|---|---|---|
| `ids` | `f"{class_name}:{spatial_bucket_x}:{spatial_bucket_y}:{spatial_bucket_z}:{uuid4_short}"` | UUID at the tail keeps inserts idempotent across restarts. Bucket prefix is *informational only* — dedup decisions don't read the id, they re-query. |
| `embeddings` | CLIP `embed_image` of the **cropped bbox region** of the source frame, 512-dim, L2-normalized | Visual matching survives caption rewording. Cropped (not full-frame) so two adjacent objects get distinct vectors. |
| `documents` | `f"{class_name}. {caption}"` | Lets Chroma's default text-embedding fallback work even if the visual embedding is missing (graceful degradation). Also makes `.peek()` output readable. |
| `metadatas` (one dict per record) | see below | |

#### Metadata fields

```jsonc
{
  // Identity
  "class_name":      "chair",        // YOLO label
  "class_id":        56,
  "first_seen_ts":   1716240000.0,   // epoch seconds, set on insert
  "last_seen_ts":    1716240320.0,   // epoch seconds, set on every update
  "sightings":       7,              // monotone counter (helps de-noise transients)

  // Position (map frame from bboxes_to_positions)
  "x": 1.23, "y": -0.45, "z": 0.78,
  "position_frame":  "map",          // pinned in case upstream ever changes
  "position_conf":   0.91,           // running average

  // Visual metadata for query results
  "caption":         "a wooden dining chair, leaning slightly",
  "bbox_last":       "[120,80,260,400]",   // (x1,y1,x2,y2) of most recent sighting, JSON-encoded — Chroma metadata can't hold lists
  "frame_ref":       "frames/2026-05-21T18-22-13Z_chair_abc123.jpg",  // path on disk; None if not archived

  // Embedding hygiene
  "embedding_model": "clip-vit-base-patch16",  // for forward-compat invalidation if we change models
  "embedding_dim":   512,
}
```

ChromaDB metadata values must be scalars (str/int/float/bool) — lists are JSON-encoded into strings. Tradeoff: spatial filters use the scalar `x`/`y`/`z` columns (Chroma's `where=` accepts numeric range filters), and we never need to filter on the raw bbox.

#### One collection vs. partition by zone/room?

**One collection.** Reasons:
- We don't have a reliable room classifier yet; partitioning would require either manual labelling or a `RoomMembership` classifier that doesn't exist.
- Spatial filters via `where={"x": {"$gte": ...}, ...}` are fast enough at the scales we expect (≤ ~10k records).
- Querying "the kitchen" becomes a metadata range query, not a collection selection — cheaper to evolve.

If/when zone metadata becomes available, add a `zone: str` metadata field and filter — no migration of records needed.

### Dedup / upsert strategy

When a new detection arrives, we have:

```
new = (class_name, position=(x, y, z), embedding, caption, confidence, bbox, ts, frame_ref)
```

Decision tree (in `perception/dedup.py`):

```
1. CANDIDATES = scene_store.find_nearby(
       class_name=new.class_name,
       position=new.position,
       radius=SPATIAL_RADIUS,           # 0.5 m  (default)
   )

2. If CANDIDATES is empty:
       → INSERT new

3. For each candidate c in CANDIDATES (sorted by L2 distance ascending):
       cos_sim = dot(c.embedding, new.embedding)         # both L2-normalized
       if cos_sim >= EMB_SIM_HIGH:                       # 0.85
           → UPDATE c (running-mean position, max confidence, bump sightings, refresh caption, last_seen_ts)
           return
       if cos_sim >= EMB_SIM_LOW and L2(c, new) <= TIGHT_RADIUS:   # 0.65 and 0.2 m
           → UPDATE c   (visually similar AND very close — treat as same)
           return

4. → INSERT new (failed all merge tests)
```

#### Why these thresholds

**[Addendum]** The addendum asks us to justify `τ_pos` relative to the walkie-sdk converter's noise. `bboxes_to_positions` is a YOLO-3D pipeline on the robot side — its accuracy depends on the depth sensor and TF chain in front of it. Empirically (and per the EIC team's prior runs), same-object position jitter from a single static viewpoint is on the order of **5–15 cm**; cross-viewpoint same-object position disagreement at 2–3 m range is **15–30 cm** when TF is well-calibrated. **0.5 m gives ~2–3× margin over typical cross-viewpoint disagreement** while staying tighter than the typical inter-object spacing in a room (chairs around a table are usually ≥0.6 m apart center-to-center). If a particular deployment has noisier 3D, override via `SCENE_DEDUP_RADIUS_M`.

| Threshold | Default | Reasoning |
|---|---|---|
| `SPATIAL_RADIUS` (`τ_pos`) | **0.5 m** | 2–3× margin over typical cross-viewpoint converter noise (see above). Smaller risks splitting the same chair into two records when viewed from two angles. Larger risks merging two chairs at a dining table. |
| `EMB_SIM_HIGH` (`τ_sim`) | **0.85** (cosine) | CLIP ViT-B/16 same-instance scores across viewpoint changes empirically land 0.80–0.95; cross-instance same-class scores land 0.55–0.80. 0.85 is the conservative cut that prefers split over false-merge. |
| `EMB_SIM_LOW` + `TIGHT_RADIUS` | **0.65 + 0.2 m** | Failsafe for the case where lighting or occlusion drops the embedding similarity below 0.85 but the bbox lifts to within 20 cm of an existing entry. The tight spatial gate prevents two different objects at the same desk from being fused. |

All four are constants in `perception/dedup.py`, override-able via env vars (`SCENE_DEDUP_RADIUS_M`, `SCENE_EMB_SIM_HIGH`, …) following the existing pattern from `services/explore.py`.

#### Update semantics (when we UPDATE, not INSERT)

**[Addendum]** The addendum asks us to specify position-smoothing behavior. We use a **running mean weighted by sightings** rather than EMA. Reasoning: in a static scene the running mean's variance shrinks as 1/N, giving the most stable position estimate over hundreds of sightings — which is exactly what the 200-tick stare test exercises. EMA (α < 1) reacts faster to actual motion, but in a *static* environment that responsiveness is just noise sensitivity. Real motion is handled by the "moved by human" branch (next subsection): when an object physically moves beyond `τ_pos`, the spatial gate correctly refuses to merge, and a new record is inserted — that's the right "tracking" behavior given we have no object-permanence model.

- `position` ← running mean weighted by sightings: `new_pos = (old_pos * n + new_pos) / (n+1)`
- `position_conf` ← running mean of detection confidence (same formula)
- `sightings` ← `n + 1` (this is the addendum's `observation_count`)
- `last_seen_ts` ← detection's `ts`
- `first_seen_ts` ← **preserved** (never overwritten on update)
- `caption` ← new caption replaces old (latest description wins)
- `bbox_last` ← latest
- `frame_ref` ← **not updated** on UPDATE. Frames are archived once per INSERT only, to keep disk footprint proportional to the number of distinct objects rather than the number of sightings. The trade-off: callers wanting a recent frame for a long-lived object must capture one fresh; the stored `frame_ref` shows the *first* sighting.
- `embedding` ← **keep the original** (don't average vectors — it drifts toward the mean of the class and hurts future dedup)

### Object moves between sessions

**[Addendum]** "Object moved by a human" is *not* a special branch in the dedup tree. It's the natural consequence of the spatial gate:

1. Detection at position B arrives with `dist(B, R_at_A) > τ_pos`.
2. `find_nearby` does not return `R_at_A` as a candidate.
3. `classify` returns `INSERT`.
4. A new record at B lands in the store. **`R_at_A` is left untouched** — its `last_seen_ts` stops advancing, so it ages out naturally via TTL or shows up as "disappeared" in `diff(since=now − minutes)`.

There is no explicit "stale" flag. Callers determine staleness from `last_seen_ts` aging (this is also what the `SceneDiff.disappeared` partition surfaces). If you want a hard "stale" boolean in the metadata for cheaper filtering, that's a one-line addition — but right now timestamp-based filtering covers the use cases the agent has (`recency_query`, `diff`).

### Disappear / reappear (object returns to its original spot)

Object disappearing then reappearing in the *same* world-frame position is *not* a special case here: when it reappears, dedup finds the dormant record (still in the store, just with an old `last_seen_ts`), the cosine match passes `EMB_SIM_HIGH`, and we UPDATE — `sightings` increments, `last_seen_ts` refreshes, `first_seen_ts` is preserved, no duplicate created. Test `test_08_disappear_then_reappear_merges` pins this.

### Class equivalence — currently strict

**[Addendum]** The addendum mentions "Class agreement (or both are in the same class-equivalence group)". Right now `classify()` raises on cross-class candidates — exact class match is required. This is the safer default because YOLO label noise across classes is much rarer than viewpoint embedding drift. If we later want to merge `"mug"` and `"cup"` from different model checkpoints, the right place is a single `CLASS_EQUIVALENCE: dict[str, str]` mapping consulted by `find_nearby` (canonicalize before the where-clause). The store would still record the *raw* class for audit. Not implemented yet — flag as a follow-up if the model ever produces conflicting labels.

### Per-decision audit logging

**[Addendum]** Every upsert decision emits a single structured INFO line via the `perception.store` logger:

```
scene.dedup action=INSERT id=chair:0:0:0:abc12345 matched_id=null    dist=nan   sim=nan   reason=no candidates within radius
scene.dedup action=UPDATE id=chair:0:0:0:abc12345 matched_id=chair:0:0:0:abc12345 dist=0.080 sim=0.987 reason=cosine 0.987 ≥ EMB_SIM_HIGH (0.85); dist 0.08m
scene.dedup action=INSERT id=mug:5:5:0:xyz78901   matched_id=null    dist=nan   sim=nan   reason=2 candidate(s) failed both merge gates
```

Fields: `action` (INSERT/UPDATE), `id` (the affected record), `matched_id` (the closest candidate, even on INSERT, so we can audit near-merges), `dist` (L2 to closest candidate, NaN if no candidates within `τ_pos`), `sim` (cosine to closest candidate), `reason` (which gate fired or what failed). Pipe this to a JSONL file in production for offline behavior audits.

### Query API surface — `perception.store.SceneStore`

```python
class SceneStore:
    # Writes
    def upsert(self, entry: SceneEntry) -> str: ...
        # Returns the chroma id of the affected record. Dedup decision is logged.

    # Reads
    def semantic_query(
        self,
        text: str,
        n_results: int = 5,
        min_last_seen_ts: float | None = None,    # recency floor (epoch s)
        within_radius_of: tuple[float, float, float] | None = None,
        max_distance_m: float | None = None,
    ) -> list[SceneEntry]: ...
        # CLIP text embed → cosine knn over Chroma + post-filter by spatial / recency.

    def visual_query(
        self,
        image: PIL.Image,
        n_results: int = 5,
        # … same filters as semantic_query
    ) -> list[SceneEntry]: ...

    def spatial_query(
        self,
        center: tuple[float, float, float],
        radius_m: float,
        class_name: str | None = None,
        n_results: int | None = None,
    ) -> list[SceneEntry]: ...
        # No vector search — metadata-only range filter then optional knn ordering.

    def recency_query(
        self,
        since_ts: float,
        class_name: str | None = None,
        n_results: int | None = None,
    ) -> list[SceneEntry]: ...

    def diff(
        self,
        since_ts: float,
        within: tuple[tuple[float, float, float], float] | None = None,  # (center, radius_m)
    ) -> SceneDiff:
        """Returns: appeared (first_seen > since_ts), refreshed (last_seen > since_ts
        and first_seen <= since_ts), disappeared (last_seen <= since_ts, no
        sighting since)."""

    # Maintenance
    def prune(self, *, ttl_sec: float | None = None, max_records: int | None = None) -> int: ...
        # Returns count pruned. See retention policy below.
```

Every read returns `list[SceneEntry]` (a frozen dataclass) — callers never see raw Chroma dicts. The agent's existing `find_object_from_memory` tool gets re-wired to call `semantic_query`.

### Retention policy

Two knobs, both off by default but configurable:

- **TTL on `last_seen_ts`** (`SCENE_TTL_SEC`, e.g. `86400` for 24h). `prune()` deletes records whose `last_seen_ts` is older than the cutoff. The "where did I last see the remote?" query should still work over week-old data by default, so TTL stays optional.
- **Hard cap on row count** (`SCENE_MAX_RECORDS`, e.g. `5000`). `prune()` deletes the oldest-by-`last_seen_ts` records over the cap. Keeps Chroma's HNSW index fast.
- **Frame storage policy**: full frames are *not* stored in Chroma. The pipeline writes the source frame to `frames/{ts}_{class}_{id8}.jpg` when (and only when) an INSERT happens — updates do not re-archive frames. A separate `prune_frames()` removes orphaned frames (no record points to them). This keeps the disk footprint roughly proportional to the number of distinct objects, not to the runtime length.

### Background loop semantics

```python
async def run_scene_perception(
    *,
    camera: CameraSource,       # has .capture_pil() — wraps walkie.camera.get_frame()
    detector: Detector,         # walkie-ai-server object detection client
    captioner: Captioner,
    embedder: Embedder,         # CLIP image embed
    lifter: PositionLifter,     # walkie.tools.bboxes_to_positions
    store: SceneStore,
    interval_sec: float = 2.0,
    on_tick: Callable[[TickReport], None] | None = None,
) -> None:
    ...
```

- Pure `asyncio.Task` driver — `await asyncio.sleep(interval_sec)` between ticks, **never** `time.sleep`.
- All inference calls inside one tick run concurrently via `asyncio.gather` against `asyncio.to_thread`-wrapped HTTP clients (the existing `client/` uses sync `requests`; we wrap, we don't rewrite).
- One tick *never* blocks the agent's main loop because the task runs on the same event loop as the agent and yields on every `await`. If a tick exceeds `interval_sec`, the next tick starts immediately (no piling up — the loop runs strictly sequentially per task).
- Graceful shutdown: `task.cancel()` → `CancelledError` raised at the next `await` → finally-block flushes the last in-progress upsert to disk before exiting.
- Structured logging: `logging.getLogger("perception")` emits one JSON line per tick (`ts, frame_age, n_detections, n_inserts, n_updates, n_skips, latency_ms_per_stage`).

### Coupling to the existing repo

`services/perception.py` and `services/explore.py` already implement a simpler version of this (track-and-promote, threading-based). The new module *replaces* them — they go away once Phase 3 lands. Until then they keep running, so the agents continue to work.

`agents/walkie_agent/tools.py::find_object_from_memory` and the equivalent in `agents/vision_agent/tools.py` get repointed at `SceneStore.semantic_query`. No agent-prompt changes needed; the tool surface stays identical.

---

## Test plan — as delivered in Phase 3

### Dedup unit tests (`tests/perception/test_dedup.py` — 11 tests)

Each test feeds synthetic `Detection`s into a `SceneStore` backed by a per-test ChromaDB and asserts a specific decision. Goal: catch any regression in the merge thresholds.

1. Empty store → INSERT.
2. Same object, slight drift (0.1 m, same embedding) → UPDATE, running-mean position, `sightings == 2`.
3. Two visually similar objects 2 m apart → both INSERT (spatial gate splits them).
4. Same position, different class → INSERT (never merge across classes).
5. Spatial near-miss (0.4 m), embedding identical → UPDATE via `EMB_SIM_HIGH` gate.
6. Spatial near-miss (0.4 m), embeddings orthogonal → INSERT (both gates fail).
7. Tight-radius failsafe: 0.15 m apart, cosine ≈ 0.70 → UPDATE via `EMB_SIM_LOW + TIGHT_RADIUS`.
8. Disappear + reappear (1 hour later, same position) → single record, `sightings == 2`, `first_seen_ts` preserved.
9. Multi-candidate closest-wins: probe closer to A than B → A updated, B untouched.
10. Env-var override (`SCENE_DEDUP_RADIUS_M=0.05`) flips a previous UPDATE into an INSERT.
11. `classify()` raises on cross-class candidates (precondition guard).

### Query API unit tests (`tests/perception/test_queries.py` — 12 tests)

Seed a fresh store with 10 known entries (mix of classes, positions, captions, timestamps). Then:

1. `semantic_query("coffee mug")` ranks mug-class entries on top.
2. `semantic_query` with `within_radius_of` + `max_distance_m` excludes the far mug.
3. `semantic_query` with `min_last_seen_ts` excludes records last seen before the cutoff.
4. `visual_query` returns `n_results` with populated `distance` field.
5. `spatial_query` returns everything within the ball regardless of class.
6. `spatial_query` with `class_name` filters further.
7. `recency_query` returns only entries `last_seen_ts > since_ts`.
8. `diff` correctly partitions into `appeared / refreshed / disappeared`.
9. `diff` `refreshed` partition surfaces re-sightings (vs. brand new inserts).
10. `prune(ttl_sec=…)` deletes the expected records.
11. `prune(max_records=N)` keeps the N freshest.
12. `upsert` after `prune` is clean (no stale chroma state).

### Background loop integration tests (`tests/perception/test_loop.py` — 8 tests)

All using the fakes in `perception/mocks.py`. End-to-end through the loop, pipeline, and store.

1. **Happy path.** 5 ticks of a static scene → 1 record with `sightings ≥ 5`, exactly 1 INSERT.
2. **Mid-scene change.** New object appears at tick 4 → 2 records, `diff` lists the new one as `appeared`.
3. **Graceful cancel.** `task.cancel()` → shutdown within 200 ms, no half-written records.
4. **Detector errors don't kill the loop.** Raise on tick 3 only → ticks 1, 2, 4, 5 still upsert; `error` field on the failed tick's report.
5. **Slow tick doesn't pile up.** Captioner with 50 ms delay vs. 5 ms interval → measured inter-tick gap ≥ delay (sequential execution).
6. **[Addendum] 200-tick stare.** Static mug for 200 ticks → 1 record, `sightings ≥ 200`, exactly 1 INSERT (the "DB grows forever" regression guard).
7. **[Addendum] Object moved by human.** Same bbox, different world-frame position (>τ_pos) → 2 distinct records, original position not dragged toward the new one.
8. **[Addendum] Long-run patrol.** 320 ticks rotating through 4 patrol stops with overlapping FOVs → 4 records (one per unique object), exactly 4 inserts, per-object `sightings ≈ N_TICKS/2`.

### Phase 1 smoke tests (`tests/perception/test_smoke_*.py` — 11 tests)

Pin the contract of every external API we depend on:

- `test_smoke_object_detection.py` — `ObjectDetectionClient.detect()` parses YOLO payload, raises on server error.
- `test_smoke_image_caption.py` — `ImageCaptionClient.caption()` / `caption_batch()` unwrap the envelope.
- `test_smoke_pose_estimation.py` — `PoseEstimationClient.estimate()` returns `PersonPose` with 17 COCO keypoints.
- `test_smoke_image_embed.py` — provisional client mirrors the (disabled) `/image-embed/*` shape so the contract is pinned now.
- `test_smoke_bboxes_to_positions.py` — walkie-sdk `Tools.bboxes_to_positions` request-reply works, times out cleanly, returns aligned `[x,y,z]` per input bbox.

All smoke tests mock the network/transport boundary and run in <1 s.

### Mock layer (`perception/mocks.py`)

- `FakeCamera(frames)` — cycles through PIL frames per `capture_pil()`.
- `FakeDetector(scripted, raise_on_idx=…)` — per-tick scripted detections; optional injected exception.
- `FakeCaptioner(captions, delay=0)` — fixed text or per-prompt map; configurable delay to test sequential ticks.
- `FakeEmbedder(dim, override_text, override_image)` — deterministic SHA-256-based embeddings; override hooks for precise cosine values in tests.
- `FakePositionLifter(scripted, default, timeout_after=…)` — bbox → (x,y,z) lookup; optional `None`-return after N calls.
- `FakeDetectedObject`, `make_tiny_image(seed)` — supporting fixtures.

---

## Open questions / what I need from you

1. **CLIP endpoint decision.** Enable `/image-embed/*` on the AI server (preferred), or embed locally? See "Missing capability" above. **Phase 3 status**: the code accepts any `Embedder` Protocol. A `FakeEmbedder` is shipped for tests. **Phase 3.1 status**: `client.ImageEmbedClient` + `perception.RemoteCLIPEmbedder` shipped on branch `feat/perception-clip-client`. **Still blocked on the server side** — `walkie-ai-server/api/__init__.py:16` must be uncommented (`app.register_blueprint(image_embed.bp)`) and the AI server redeployed. Once that lands, no code changes on this side; just construct `RemoteCLIPEmbedder(walkieAI.image_embed)` and pass it to the loop.
2. **Position frame.** `bboxes_to_positions` returns positions in whatever frame the upstream YOLO-3D node publishes. Today I'm assuming `map`. Worth confirming with whoever runs the robot ROS graph that this is stable — otherwise the design needs a `tf` lookup step. **Phase 3 status**: stored as `position_frame: "map"` in metadata so drift is detectable.
3. **Frame archival.** Default to "save one JPEG per INSERT to `frames/`" — OK, or do you want frames-never-saved (smaller footprint) or frames-on-every-tick (richer history)? **Phase 3 status**: implemented as "save on INSERT only", controllable via `SceneStore(frames_dir=...)` and the `archive_source_frame` loop arg.
4. **Threading model.** This design uses `asyncio` end-to-end. The existing `PerceptionService` / `ExploreService` use `threading.Thread`. The agents themselves are sync-with-async-tool-grouping. I'll make the new loop async, but it'll need a small `run_perception_in_thread()` adapter for `main.py` until you migrate. OK? **Phase 3 status**: loop is `asyncio.Task`-driven. Adapter into `main.py` is left for a follow-up commit.

---

## Phase 3 — Implementation summary

Module: `perception/` (new package).

| File | Lines | Purpose |
|---|---|---|
| `types.py` | ~160 | Frozen dataclasses + Protocols. No logic. |
| `dedup.py` | ~135 | Pure-function `classify(new, candidates) → DedupDecision` + env-var-tunable thresholds + position/confidence merging helpers. |
| `store.py` | ~400 | `SceneStore` — ChromaDB wrapper. `upsert / find_nearby / semantic_query / visual_query / spatial_query / recency_query / diff / prune / get_by_id / clear`. |
| `pipeline.py` | ~140 | `process_frame(frame, …)` — detect → 3D-lift → caption → embed. Returns `(list[Detection], latency_ms)`. |
| `loop.py` | ~160 | `run_scene_perception(...)` — async loop. Cancellable, sequential ticks, structured logging, error isolation. |
| `mocks.py` | ~200 | `FakeCamera / FakeDetector / FakeCaptioner / FakeEmbedder / FakePositionLifter / FakeDetectedObject / make_tiny_image`. |
| `__init__.py` | ~50 | Public re-exports. |

### Tests delivered (42 total, ~20s on a warm cache)

| File | Tests | What it covers |
|---|---|---|
| `test_dedup.py` | 11 | All decision branches (empty store, drift-merge, far-apart split, cross-class split, HIGH gate, both-gates-fail, TIGHT failsafe, disappear/reappear, multi-candidate closest-wins, env-var override, cross-class precondition). |
| `test_queries.py` | 12 | Every read path (semantic / visual / spatial / recency / diff) with filter combinations + prune by TTL and by max-records + upsert-after-prune cleanliness. |
| `test_loop.py` | 8 | Happy path, mid-scene change, graceful cancel, error recovery, sequential ticks, **[Addendum]** 200-tick stare, **[Addendum]** object-moved-by-human, **[Addendum]** long-run patrol with overlapping FOVs (320 ticks → 4 records). |
| `test_smoke_*.py` | 11 | Phase 1 smoke tests for `client/*` + walkie-sdk `bboxes_to_positions` contract. |

### Known follow-ups (out of scope for this branch)

- Wire the loop into `main.py` (sync adapter + replace the threading-based `services/perception.py` and `services/explore.py`).
- Add `client/image_embed.py` once the server enables `/image-embed/*`. Smoke test `test_smoke_image_embed.py` already pins the contract.
- Repoint `agents/walkie_agent/tools.py::find_object_from_memory` and the equivalent vision agent tool at `SceneStore.semantic_query`. Tool surface stays identical from the agent's perspective.
- Pillow 14 deprecation: `mocks.py` uses `Image.Image.getdata` (warning only; replace with `get_flattened_data` before Pillow 14 ships in 2027).
