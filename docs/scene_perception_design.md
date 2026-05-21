# Scene Embedding & Background Perception — Phase 1 + 2

Branch: `feat/scene-perception`
Status: **Discovery + Design draft for review.** No implementation code yet.

## TL;DR

Build a `perception/` Python package in this repo that runs an always-on async loop, calls the AI server for detection + caption + (eventually) CLIP embeddings, calls the walkie-sdk `Tools.bboxes_to_positions` to lift each detection to a 3D world-frame position, and upserts results into a single ChromaDB collection. Queries (semantic / spatial / recency / diff) are served from the same collection.

The big blocker found in discovery is flagged at the bottom: **walkie-ai-server has a CLIP embedding service implemented but the route is commented out**. We need a decision before coding Phase 3.

---

## Phase 1 — Discovery findings

### Where the code lives on disk

| What | Path | Notes |
|---|---|---|
| Consumer (this repo) | `/home/hextex/Documents/GitHub/walkie-agent-v2/` | `services/perception.py` + `services/explore.py` already do a *simpler* version of this — both will be replaced by the new module |
| walkie-sdk source | `/home/hextex/Documents/GitHub/Walkie-SDK/` *(stale `main`)* + `walkie-agent/.venv/lib/python3.11/site-packages/walkie_sdk/` *(rich, pre-installed)* | the GitHub `main` branch has been stripped of `arm/camera/tools/visualization`; the rich surface lives on `feat/refact_support_multi_transport`. **`pyproject.toml` here pins the git URL without a ref**, so `uv sync` will follow whatever main is — that's a latent breakage. |
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

bot.status.get_pose() -> {"x": float, "y": float, "heading": float} | None  # heading in radians
bot.status.get_velocity() -> {"linear": float, "angular": float} | None

bot.tools.bboxes_to_positions(
    coords: list[list[float]],   # each [cx, cy, w, h] in image-pixel coords
    timeout: float = 5.0,
) -> list[list[float]] | None    # each [x, y, z] in the upstream YOLO-3D frame (usually `map`); None on timeout
```

`bboxes_to_positions` is a **pub/sub request-reply**: publishes a `vision_msgs/Detection2DArray` on `/yolo/detections_2d`, waits up to `timeout` seconds for a `geometry_msgs/PoseArray` on `/ob_detection/poses`, then maps poses back to a list of `[x,y,z]`. Output order is aligned with input order. Returns `None` if no response arrived in time.

⚠ **Bug latent in `interfaces/walkie_interface.py` (not part of this work)**: `services/perception.py:108` calls `walkie.tools.bboxes_to_positions(...)` correctly, but `agents/actuator_agent/tools.py:25` calls `walkie.status.get_position()` — the SDK only exposes `get_pose()`. Flagged for a separate fix.

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

### Module layout (planned, not built yet)

```
perception/                       (new package, replaces services/perception.py + services/explore.py)
├── __init__.py
├── loop.py        # async background loop (cancellable, configurable rate)
├── pipeline.py    # 1 frame → list[SceneEntry] (calls AI server + walkie-sdk)
├── store.py       # SceneStore (ChromaDB wrapper: upsert, dedup, queries)
├── dedup.py       # pure-function decisions: match()/merge()/new()
├── types.py       # SceneEntry, Detection, Query dataclasses
└── mocks.py       # FakeCamera, FakeDetector, FakeCaptioner, FakeEmbedder, FakePositionLifter (test-only)
```

Three modules, three responsibilities, composed by `loop.py`. No inheritance, just function calls between them. Each module is independently unit-testable.

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

| Threshold | Default | Reasoning |
|---|---|---|
| `SPATIAL_RADIUS` | **0.5 m** | Wider than the typical YOLO-3D position jitter (~10 cm) and wider than typical object footprints (chair ~0.4 m). Smaller risks splitting the same chair into two records when the robot views it from two angles. Larger risks merging two chairs at the same dining table. |
| `EMB_SIM_HIGH` | **0.85** (cosine) | CLIP ViT-B/16 same-instance scores across viewpoint changes empirically land 0.80–0.95; cross-instance same-class scores land 0.55–0.80. 0.85 is the conservative cut that prefers split over false-merge. |
| `EMB_SIM_LOW` + `TIGHT_RADIUS` | **0.65 + 0.2 m** | Failsafe for the case where lighting or occlusion drops the embedding similarity below 0.85 but the bbox lifts to within 20 cm of an existing entry. We only allow this looser merge inside a very tight spatial gate so two different objects at the same desk aren't fused. |

All four are constants in `perception/dedup.py`, override-able via env vars (`SCENE_DEDUP_RADIUS_M`, `SCENE_EMB_SIM_HIGH`, …) following the existing pattern from `services/explore.py`.

#### Update semantics (when we UPDATE, not INSERT)

- `position` ← running mean weighted by sightings: `new_pos = (old_pos * n + new_pos) / (n+1)`
- `position_conf` ← running mean of detection confidence (same formula)
- `sightings` ← `n + 1`
- `last_seen_ts` ← `time.time()`
- `caption` ← **new caption replaces old** (latest description wins; old is gone but `frame_ref` history can be reconstructed from archived frames if needed)
- `bbox_last`, `frame_ref` ← latest
- `embedding` ← **keep the original** (don't average vectors — it drifts toward the mean of the class and hurts future dedup)

### Disappear / reappear

Object disappearing then reappearing is *not* a special case here: when it reappears, dedup finds the dormant record (still in the store, just with an old `last_seen_ts`), the cosine match passes `EMB_SIM_HIGH`, and we UPDATE — `sightings` increments, `last_seen_ts` refreshes, no duplicate created. Queries can use `last_seen_ts` to label the entry as "stale" or "fresh" in the response.

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

## Phase 2 — Test plan (what we'll write *before* implementation in Phase 3)

### Dedup unit tests (`tests/perception/test_dedup.py`)

Each test feeds synthetic `Detection`s into a `SceneStore` backed by an in-memory ChromaDB and asserts a specific decision. Goal: would catch a regression in the merge thresholds.

1. **Empty store → insert.** First detection of "chair" inserts.
2. **Same object, slightly moved.** Same class, same embedding, position drift 0.1 m → UPDATE (not INSERT); `sightings == 2`; position is the running mean.
3. **Two visually similar objects at different positions.** Same class, same embedding, position 1.5 m apart (> `SPATIAL_RADIUS`) → INSERT (two distinct records).
4. **Same position, different class.** Position identical, class differs → INSERT (we never merge across classes).
5. **Spatial near-miss, embedding identical.** 0.4 m apart, cosine = 0.99 → UPDATE (passes `EMB_SIM_HIGH`).
6. **Spatial near-miss, embedding far.** 0.4 m apart, cosine = 0.40 → INSERT (fails both gates).
7. **Spatial very-close, embedding mid.** 0.15 m apart, cosine = 0.70 → UPDATE (passes the `EMB_SIM_LOW + TIGHT_RADIUS` failsafe).
8. **Disappearance + reappearance.** Insert, advance `time.time` 1 hour (monkeypatched), insert the same object again → single record, `sightings == 2`, `last_seen_ts` is the new time.
9. **Multiple candidates, closest wins.** Two existing chairs at 0.3 m and 0.45 m; a new chair at the same spot as the 0.3 m one → that one is updated, the 0.45 m one is untouched.
10. **Threshold env vars override.** Set `SCENE_DEDUP_RADIUS_M=0.1`, repeat test (2) — now an INSERT, not UPDATE.

### Query API unit tests (`tests/perception/test_queries.py`)

Seed an in-memory Chroma with ~10 known entries (mix of classes, positions, captions, timestamps). Then:

1. `semantic_query("coffee mug")` returns the mug-class entries ranked by similarity.
2. `semantic_query("mug", within_radius_of=(0,0,0), max_distance_m=1)` excludes the mug at (5,5,0).
3. `semantic_query("mug", min_last_seen_ts=cutoff)` excludes records last seen before the cutoff.
4. `visual_query(image_of_known_chair)` returns the chair entry as top hit.
5. `spatial_query(center=(0,0,0), radius_m=1)` returns only entries within the ball, regardless of class.
6. `spatial_query(center=…, radius_m=…, class_name="chair")` filters further.
7. `recency_query(since_ts=cutoff)` returns only entries `last_seen_ts > cutoff`.
8. `diff(since_ts)` correctly partitions into `appeared / refreshed / disappeared`.
9. `prune(ttl_sec=…)` deletes the expected records and returns the right count.
10. `prune(max_records=N)` keeps the N freshest and deletes the rest.
11. `upsert` after `prune` of the same id works (no stale Chroma state).

### Background loop integration test (`tests/perception/test_loop.py`)

One test that exercises the whole pipeline end-to-end with `FakeCamera`, `FakeDetector`, `FakeCaptioner`, `FakeEmbedder`, `FakePositionLifter` from `perception/mocks.py`.

- **Happy path.** Run for 5 ticks with a scripted fake camera that returns the same scene each frame. Assert: store ends with exactly the expected count (one record per unique object, with `sightings == 5`).
- **Mid-scene change.** Tick 1–3 yield "chair@(0,0)"; tick 4–5 yield "chair@(0,0)" + "mug@(1,1)". Assert: 2 records, mug has `first_seen_ts >= tick4_ts`, `diff(since=tick3_ts)` lists mug as `appeared`.
- **Graceful cancel.** Start the loop, `await asyncio.sleep(0.1)`, `task.cancel()`. Assert: task finishes within 100 ms, no half-written records, log line "perception loop stopped" emitted.
- **Detector errors don't kill the loop.** `FakeDetector` raises on tick 3 only. Assert: ticks 1, 2, 4, 5 still upsert; tick 3 logs an error; the loop keeps running.
- **Slow tick doesn't pile up.** `FakeCaptioner.delay = 3 * interval_sec`. Assert: the loop runs sequentially (no concurrent ticks), and the inter-tick gap is `~max(interval, tick_duration)`, not `interval` flat.

### Mock layer (`perception/mocks.py`)

Each fake mirrors the real protocol class via duck typing — no Protocol/ABC required. Each has:

- `FakeCamera(frames: list[Image] | Iterator[Image])` — `.capture_pil()` cycles.
- `FakeDetector(scripted: dict[frame_idx → list[DetectedObject]])`.
- `FakeCaptioner(captions: dict[(class_name) → str], delay: float = 0)`.
- `FakeEmbedder(seed: int)` — deterministic embeddings via NumPy RNG seeded by class+bbox.
- `FakePositionLifter(scripted: dict[bbox_tuple → (x,y,z)])`.

Tests stay <5 s in aggregate because there's no real network and no real model load.

---

## Open questions / what I need from you

1. **CLIP endpoint decision.** Enable `/image-embed/*` on the AI server (preferred), or embed locally? See "Missing capability" above.
2. **Position frame.** `bboxes_to_positions` returns positions in whatever frame the upstream YOLO-3D node publishes. Today I'm assuming `map`. Worth confirming with whoever runs the robot ROS graph that this is stable — otherwise the design needs a `tf` lookup step.
3. **Frame archival.** Default to "save one JPEG per INSERT to `frames/`" — OK, or do you want frames-never-saved (smaller footprint) or frames-on-every-tick (richer history)?
4. **Pinning the SDK ref.** `pyproject.toml`'s `walkie-sdk = { git = "..." }` has no `ref =`. With main currently stripped of `tools`, the next `uv sync` will break this repo. Worth pinning to a known-good commit *as part of this branch*, or out of scope?
5. **Threading model.** This design uses `asyncio` end-to-end. The existing `PerceptionService` / `ExploreService` use `threading.Thread`. The agents themselves are sync-with-async-tool-grouping. I'll make the new loop async, but it'll need a small `run_perception_in_thread()` adapter for `main.py` until you migrate. OK?

When you've signed off (or pushed back) on the above, I'll move to Phase 3.
