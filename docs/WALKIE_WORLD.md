# walkie_world — the robot's world model + spatial memory

`walkie_world/` is the part of Walkie that **knows about its world** — the arena map, the
objects it has seen and where, and the people it has met. It is the single query engine
every task and agent reaches through **`ctx.world`**.

As the robot moves, its camera keeps spotting objects — a mug, a chair, a bottle. The
perception producer (`services/realtime_explore`) works out *where in the room* each one is
and folds it into a running 3D scene graph that `walkie_world` owns. Later you can ask
*"where's the mug?"*, *"what's near the table?"*, *"which room am I in?"*, or *"is this the
same customer?"* and it answers from that model — the robot's **memory of places, things,
and people**, not the eyes (that's the camera + the detection server).

> **One line:** `walkie_world` is the import-light DOMAIN MODEL (map + object scene graph +
> people); `services/realtime_explore` is the PERCEPTION PRODUCER that feeds it. The old
> `services/walkie_graphs` package was split along exactly that seam.

---

## Two packages, one seam

```
  consumers: tasks/* , agents/*  ───────────────┐   (query: ctx.world / the Database agent)
                                                ▼
  producer:  services/realtime_explore  ──►  walkie_world   (the model)
             (capture + batch build)       observe_objects   │  ├─ map/    rooms·locations·doors·objects·vocab
                                                             │  ├─ scene/  numpy SceneStore + relations + ingest
                                                             │  └─ people/ face + appearance re-ID (ChromaDB, lazy)
                                                             └─ interfaces.perception.geometry (voxel; pure numpy)
```

- **`walkie_world`** (top-level, import-light: numpy + lazy ChromaDB; **no** Open3D / camera
  / SDK) is the model. `import walkie_world` pulls nothing heavy; ChromaDB loads only when a
  people method is first used.
- **`services/realtime_explore`** is the producer (the old capture/build half). It depends on
  `walkie_world` and pushes observations in via `world.observe_objects(...)`. The dependency
  only ever points producer → model.
- **Exactly one `WalkieWorld` per process**, built in each task's `run.py` (or `main.py`) and
  injected into the producer, the Database agent, and `TaskContext.world`. One instance ⇒ one
  scene lock and one ChromaDB owner.

---

## Architecture — two decoupled loops

```
  capture thread (every INTERVAL_SEC)          build worker (every REBUILD_EVERY_N snapshots)
  ┌───────────────────────────────┐            ┌─────────────────────────────────────────────┐
  │ 1 RGB-D frame + 1 detect/      │            │ window of buffered snapshots                 │
  │   caption/embed round-trip     │  buffer    │  → refine_poses (baseline=nav | auto=Open3D) │
  │ → write live perception.json   │ ─────────► │  → lift each mask with its optimized pose    │
  │ → append a compact Snapshot to │  (on-disk  │  → BATCH associate (constrained agglomerative)│
  │   the on-disk ring buffer      │   ring)    │  → world.observe_objects(observations):       │
  └───────────────────────────────┘            │       merge (never shrink) → derive relations │
       no ICP, no fusion, no maintenance        │       → atomic install   [+ optional TSDF]    │
                                                └─────────────────────────────────────────────┘
                                                   queries read the last installed scene
```

**Capture thread** (`realtime_explore/service.py`) — every `INTERVAL_SEC`: grab one
synchronized RGB-D frame (`CameraSnapshot.capture` — depth, RGB, camera→map pose, intrinsics,
robot pose, all read back-to-back *before* the slow detection round-trip so the pose matches
the image), run **one** fused `walkieAI.image.process` call (masks + per-detection caption +
CLIP embed), write the live `perception.json` straight from those detections, and append a
compact `Snapshot` to the on-disk ring buffer (`graph_buffer/`). No ICP, no fusion — the
capture tick is cheap and never blocks on the build.

**Build worker** (`realtime_explore/builder.py`, single-flight) — after `REBUILD_EVERY_N` new
snapshots (and no more often than `REBUILD_MIN_INTERVAL_SEC`; sooner on a cold start via
`FIRST_BUILD_N`):

1. **refine poses** (`poses.py`) — `baseline` returns the nav/TF poses; `auto` runs Open3D
   RGB-D odometry + pose-graph optimization, seeded and sanity-bounded by nav.
2. **lift** every detection's mask to a world-frame cloud with its frame's optimized pose
   (`interfaces.perception.geometry.deproject_mask`).
3. **associate** (`associate.py`) — one constrained-agglomerative pass over *all* the window's
   detections → fused `ObjectObservation` clusters.
4. **`world.observe_objects(observations)`** — the producer hands the clusters to the model,
   which, under its scene lock, **merges** them into the persisted `SceneStore` (match-or-insert,
   **never shrink**), **derives relations**, and **atomically installs** the new immutable scene.
5. **(optional)** fuse a clean **volumetric TSDF map** (`tsdf.py`) → `graph_scene/map.npz`.

`perception.json` ("what's in front of me **now**") is written by the capture thread, decoupled
from the lagging batch build, so live perception never goes stale even while a build runs.

---

## Why a batch design — the failures it fixes

Batch reconciles pose error **once over the whole window** and decides instance identity **once
over all detections**:

| Failure mode | Batch fix |
|---|---|
| fuzzy clouds (per-frame pose jitter) | clean poses before any fusion (pose-graph; or just trust settled nav) |
| ghost duplicates (a mis-posed frame inserts a node a few cm off) | one association pass; near lifts union together |
| flat object absorbed into the table | **mutual-min** cloud overlap — spoon→table≈1 but table→spoon≈0, so min≈0 ⇒ no merge |
| identical-twin fusion (two chairs become one) | **hard centroid cap**, independent of CLIP |
| a row of chairs blobbed into one | **complete-linkage + per-class AABB-extent veto** (no transitive chaining) |
| detector label flip-flop (cup↔mug) → duplicates | strict **cross-class CLIP gate** — same object fuses, distinct objects don't |
| ~3-sighting confirmation lag | `n_obs` = cluster member count, available from one build |

---

## The model: `WalkieWorld` (`walkie_world/world.py`)

One facade composing three sub-stores. Construction is cheap and import-light; pass
`embed_text` (bound to `walkieAI.image.embed_text`) to enable CLIP text search for objects and
semantic attire re-ID for people.

```python
world = WalkieWorld(embed_text=lambda q: walkieAI.image.embed_text(q), enable_people=True)
```

**Map / rooms / vocab** (`walkie_world/map/`) — the static arena:
- grounding: `room/location/obj/category/name/gesture(text)`, `location_pose(name)`,
  `is_barrier(name)`, `vocab_prompt()`, `categories`/`objects`/`names`/`gestures`.
- waypoints + geometry: `pose(name)`, `resolve_pose(name, env_fallback=, default=)`,
  `rooms`/`locations`/`doors`.
- **polygons (new):** `room_at(x, y)` (point-in-polygon "which room am I in"),
  `is_near_door(x, y)` (a surveyed doorway **polygon region** OR the trigger radius),
  `map_objects()` (the world editor's surveyed object shapes: XY footprint + Z height).

**Objects / scene** (`walkie_world/scene/`) — the query contract the Database agent
(`agents/database_agent/tools.py`) and GPSR (`tasks/GPSR/skills.py`) depend on:
- `query_text(query, k=5, *, near=None, radius=None)` — embed the query, rank by cosine over
  the L2-normalized `(N,D)` matrix in **one matmul** (≤`PRUNE_MAX_RECORDS` objects), then
  confirmation-gate + optional spatial filter. **Keyword fallback** when the embed server is
  down (embed None / raises / empty / op fails) — searches `class_name + best_caption + captions`.
- `query_near(center, radius)`, `recently_seen(limit)`, `all_objects()`, `get(id)`,
  `relations_of(id)`, `to_text_description()`, `count()`.
- **ingest:** `observe_objects(observations)` (producer → model; the only writer path).

**People** (`walkie_world/people/`, ChromaDB, lazy) — face + appearance memory unified across
HRI and Restaurant:
- `enroll_person(name, drink, face_embedding, *, person_id=, app_embedding=,
  appearance_caption=, appearance_caption_embedding=, last_seen_pose=, ...)` — folds the face
  into a running centroid; appearance + caption are latest-wins. **Attire-only enrollment**
  (empty `face_embedding`) is supported for Restaurant (no faces): a zero placeholder face is
  stored so the id still registers, and re-ID goes via appearance.
- `recognize_person(face)`, `recognize_person_fused(face, app, …)`,
  `find_person_by_caption(query)` (semantic CLIP-text match in the third
  `people_appearance_caption` collection, lexical fallback), `get_person`, `list_people`.

`ObjectNode`: `id`, `class_name`, `centroid (x,y,z)`, `extent`, `aabb_min/max`, `clip_emb`
(L2-normalized), `captions`, `best_caption`, `n_obs`, `first/last_seen_ts`, plus **`source`**
(`"map"`/`"perception"`) and **`footprint_polygon`**. `PersonRecord` adds `appearance_caption`,
`appearance_caption_embedding`, `last_seen_pose`, `last_seen_room`, `pose_label`, `seat` to the
face/drink/attributes fields.

### Map-defined objects → promoted to point clouds

The world editor emits object shapes (an XY bounding box + a Z height). `walkie_world` seeds each
as a `source="map"` **placeholder node** with a stable id (`map:<name>`), no cloud, `n_obs=0`.
These bypass the confirmation gate (authoritative — queryable immediately) and are **never
pruned**. When perception detects a same-class object whose centroid falls in the placeholder's
box, `observe_objects` **promotes** it: the synthetic bbox geometry is replaced by the real fused
cloud and `n_obs` climbs. In the Rerun viz a map placeholder draws as its **bounding box** and a
perceived/promoted object draws as its **point cloud** — plus rooms draw as **wall** line strips
from their boundary polygons. Re-seeding on every startup is idempotent (it refreshes only
never-perceived placeholders), so a re-survey updates geometry without clobbering real clouds.

### No ChromaDB for the scene

The object store is `graph_scene/{nodes.json, embeddings.npy, edges.json}` (+ `map.npz` when TSDF
is on) — a numpy `(N,D)` matrix, not ChromaDB. It **survives restart** and **accretes** (builds
merge, never shrink; capped by `PRUNE_MAX_RECORDS` by `last_seen`). Thread-safety is one `RLock`;
`install()` swaps an *immutable* `BuiltScene` pointer, so a rebuild never blocks a query.
(ChromaDB is used only by the people sub-store — faces, appearance, and caption re-ID.)

---

## Staging: object recall first, volumetric map second

- **Stage 1 — object recall (default):** `POSE_MODE=baseline` + `TSDF=0`. No Open3D on the build
  path; already fixes the duplicate/absorption/twin/lag failures above.
- **Stage 2 — clean volumetric map (validate first):** `POSE_MODE=auto` + `TSDF=1`.
  - `auto` (`poses.py`): Open3D RGB-D odometry + sparse loop closures → `PoseGraph` →
    `global_optimization`, node 0 anchored at its nav pose, every edge sanity-bounded vs the nav
    delta and reverted past tolerance — so `auto` can **never do worse than nav**. Needs `KEEP_RGB=1`.
  - `TSDF` (`tsdf.py`): `VoxelBlockGrid` integration → a clean structural cloud (voxels seen by
    ≥3 frames). Depth-only, Open3D, CUDA preferred.
  - **Measure on a replayed buffer before flipping these on.**

Both Open3D modules are import-guarded and degrade to `baseline` / `None`, so a box without
Open3D still maps via Stage 1.

---

## On-robot facts worth knowing

- **Depth is float32 metres** (NaN/≤0 invalid), aligned to the RGB intrinsics → TSDF/odometry use
  `depth_scale=1.0` and integration extrinsic = `inv(camera→map pose)`.
- **Trusted depth band ≈ 0.3–4 m** — stereo error grows ~quadratically; `MAX_DEPTH_M=4.0` drops
  far, noisy geometry at the source.
- **Pose-at-capture** (sensors + TF read together, before detection) is an architectural invariant.
- **CUDA** is auto-probed (`WALKIE_EXPLORE_O3D_DEVICE=auto`, `tools/check_gpu`); the Open3D path
  falls back to CPU silently.

---

## Commands

```bash
# Record a run on the robot (the capture thread fills graph_buffer/), then build OFFLINE — no
# robot, deterministic — to tune association/poses/TSDF or A/B baseline vs auto on the SAME buffer:
uv run python -m services.realtime_explore.tools.replay graph_buffer
uv run python -m services.realtime_explore.tools.replay graph_buffer --pose-mode auto --tsdf
uv run python -m services.realtime_explore.tools.replay graph_buffer --store graph_scene   # persist + print

# Wipe the scene store + buffer for a clean slate (robot stopped):
uv run python -m services.realtime_explore.tools.reset -y          # or ./run.sh reset

# Probe Open3D GPU support:
uv run python -m services.realtime_explore.tools.check_gpu

# Tests:
uv run pytest tests/graphs/    # producer pipeline + scene store (bare numpy/scipy)
uv run pytest tests/world/     # walkie_world: facade, polygon/room_at, observe, map objects, people re-ID
```

## Config

Tuning knobs use the **`WALKIE_EXPLORE_*`** prefix (renamed from the legacy `WALKIE_GRAPHS_*`),
split by owner:

- **Producer** — `services/realtime_explore/config.toml` (capture / build / lift / association /
  camera TF / TSDF / pose / viz), grouped under `[explore] / [explore.camera] / [explore.lift] /
  [explore.capture] / [explore.build] / [explore.assoc]`.
- **Model scene store + relations** — `walkie_world/config.toml` (`[scene.store]` /
  `[scene.relations]`): `STORE_DIR`, `MIN_OBS_CONFIRM`, `REQUIRE_CONFIRMATION`,
  `PRUNE_MAX_RECORDS`, `ASSOC_MAX_DIST_M` (= merge distance), and the relation thresholds.
- **People** — root `config.toml` `[people]` (`PEOPLE_*`, `FACE_MATCH_THRESHOLD`,
  `APPEARANCE_MATCH_THRESHOLD`, `WORLD_PEOPLE_CAPTION_MATCH_THRESHOLD`).

The keys *are* the env-var names; precedence is `shell env > .env > root config.toml > module
config.toml > code default`. `walkie_config.load_config` globs `services/*/config.toml` plus the
top-level `walkie_world/config.toml`.

## Module map

**`walkie_world/`** — `world.py` (the `WalkieWorld` facade + scene lock + `observe_objects`) ·
`map/` (`locations.py` rooms/locations/doors/`MapObject` + `LocationBook`; `polygon.py`
point-in-polygon; `vocab.py` `WorldModel` grounding) · `scene/` (`store.py` `SceneStore` +
`ObjectNode`/`Relation` + persistence + queries; `relations.py` AABB edges; `ingest.py`
`ObjectObservation` contract) · `people/` (`store.py` `PeopleStore` + `PersonRecord`;
`vector_db.py` ChromaDB plumbing) · `config.{py,toml}`.

**`services/realtime_explore/`** — `service.py` (`RealtimeExplore` facade + capture thread + build
worker) · `buffer.py` (snapshot ring buffer) · `builder.py` (build orchestration) · `poses.py`
(baseline/auto pose refinement) · `associate.py` (batch association) · `tsdf.py` (VoxelBlockGrid
map) · `snapshot.py` (perception.json writer) · `fusion.py` (`nn_ratio` overlap math) ·
`pcd_ops.py` (Open3D device/ICP helpers) · `viz.py` (Rerun scene + room walls) · `tools/`
(`replay`, `reset`, `check_gpu`). Geometry primitives are reused from `interfaces/perception/`
and `interfaces/devices/camera.py` (`CameraSnapshot`).
