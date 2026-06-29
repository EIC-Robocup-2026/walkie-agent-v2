# Walkie Graphs — the robot's spatial memory

`services/walkie_graphs/` is the part of Walkie that **remembers what it has seen and where**.

As the robot moves, its camera keeps spotting objects — a mug, a chair, a bottle. Walkie Graphs works
out *where in the room* each one is and keeps a running 3D map. Later you can ask *"where's the mug?"*
or *"what's near the table?"* and it answers from that map. It's the robot's **memory of places and
things** — not the eyes (that's the camera + the detection server), just the memory the eyes fill in.

> **Design in one line:** cheap continuous **capture** into an on-disk snapshot buffer, then occasional
> **offline batch builds** that fuse a window of snapshots — with globally-consistent poses — into clean
> object nodes (and, optionally, a volumetric map). This replaced an earlier real-time incremental
> pipeline whose complexity all existed to fight pose drift one frame at a time.

---

## Architecture — two decoupled loops

```
  capture thread (every INTERVAL_SEC)          build worker (every REBUILD_EVERY_N snapshots)
  ┌───────────────────────────────┐            ┌─────────────────────────────────────────────┐
  │ 1 RGB-D frame + 1 detect/      │            │ window of buffered snapshots                 │
  │   caption/embed round-trip     │  buffer    │  → refine_poses (baseline=nav | auto=Open3D) │
  │ → write live perception.json   │ ─────────► │  → lift each mask with its optimized pose    │
  │ → append a compact Snapshot to │  (on-disk  │  → BATCH associate (constrained agglomerative)│
  │   the on-disk ring buffer      │   ring)    │  → MERGE into persisted SceneStore (never    │
  └───────────────────────────────┘            │     shrink) → derive relations → install      │
       no ICP, no fusion, no maintenance        │  → (optional) TSDF volumetric map             │
                                                └─────────────────────────────────────────────┘
                                                   queries read the last installed scene
```

**Capture thread** (`service.py`) — every `INTERVAL_SEC`: grab one synchronized RGB-D frame
(`CameraSnapshot.capture` — depth, RGB, camera→map pose, intrinsics, robot pose, all read back-to-back
*before* the slow detection round-trip so the pose matches the image), run **one** fused
`walkieAI.image.process` call (masks + per-detection caption + CLIP embed), write the live
`perception.json` straight from those detections, and append a compact `Snapshot` (depth + masks + pose
+ per-detection metadata) to the on-disk ring buffer (`graph_buffer/`). No ICP, no fusion, no
maintenance — the capture tick is cheap and never blocks on the build.

**Build worker** (`builder.py`, single-flight) — triggered after `REBUILD_EVERY_N` new snapshots (and
no more often than `REBUILD_MIN_INTERVAL_SEC`; sooner on a cold start via `FIRST_BUILD_N`):

1. **refine poses** (`poses.py`) over the window — `baseline` returns the nav/TF poses as captured;
   `auto` runs Open3D RGB-D odometry + pose-graph global optimization, seeded and sanity-bounded by nav.
2. **lift** every detection's mask to a world-frame cloud with its frame's optimized pose
   (`interfaces.perception.geometry.deproject_mask`, the same flying-pixel cleanup throughout).
3. **associate** (`associate.py`) — one constrained-agglomerative pass over *all* the window's
   detections → fused `ObjectObservation` clusters.
4. **merge** the clusters into the persisted `SceneStore` (`scene.py`) — match-or-insert, **never
   shrink** — then **derive relations** (`relations.py`) and **atomically install** the new immutable
   scene. Queries always read the last installed scene.
5. **(optional)** fuse a clean **volumetric TSDF map** (`tsdf.py`) and save it to `graph_scene/map.npz`.

`perception.json` ("what's in front of me **now**") is written by the capture thread, **decoupled from
the lagging batch build**, so live perception never goes stale even while a build runs.

---

## Why a batch redesign — the failures it fixes

The old pipeline fused every frame immediately, correcting one camera pose at a time, which spawned a
long tail of duplicate/fuzzy/absorbed-object failures. Batch reconciles pose error **once over the whole
window** and decides instance identity **once over all detections**:

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

## The store & query contract (`scene.py`)

One mapped object = one `ObjectNode`: `id`, `class_name`, `centroid (x,y,z)`, `extent`, `aabb_min/max`,
`clip_emb` (L2-normalized), `captions` (union of member captions), `best_caption`, `n_obs`,
`first/last_seen_ts`. Relations are `Relation(src_id, dst_id, predicate ∈ {near,on,above,inside},
weight)`.

`WalkieGraphs` (the facade) and `SceneStore` expose the API the Database agent
(`agents/database_agent/tools.py`) and GPSR (`tasks/GPSR/skills.py`) depend on:

- `query_text(query, k=5, *, near=None, radius=None)` — embed the query, rank by cosine over the
  L2-normalized `(N,D)` matrix in **one matmul** (≤`PRUNE_MAX_RECORDS` objects), then confirmation-gate +
  optional spatial filter. **Keyword fallback** when the embed server is unavailable (embed is None /
  raises / returns empty / the vector op fails) — searches `class_name + best_caption + captions`.
- `query_near(center, radius)`, `recently_seen(limit)`, `all_objects()`, `get(id)`, `relations_of(id)`,
  `to_text_description()`, plus `start()`, `stop()`, `observe()`.
- **Confirmation gate** (`n_obs >= MIN_OBS_CONFIRM`, default 2) hides one-off false positives from every
  query method; `get(id)` is not gated.

**No ChromaDB for the scene** — the store is `graph_scene/{nodes.json, embeddings.npy, edges.json}`
(+ `map.npz` when TSDF is on). It **survives restart** (loads at construction; queries work before the
first build) and **accretes** (builds merge into it, never shrinking, capped by `PRUNE_MAX_RECORDS` by
`last_seen`). Thread-safety is one `RLock`; `install()` swaps an *immutable* `BuiltScene` pointer, so a
rebuild never blocks a query and an in-flight query keeps a consistent snapshot. (ChromaDB remains a
project dep — `perception/people_store.py` uses it for faces — but the scene no longer touches it.)

---

## Staging: object recall first, volumetric map second

- **Stage 1 — object recall (default):** `POSE_MODE=baseline` (trust the nav/TF pose) + `TSDF=0`. No
  Open3D on the build path; this already fixes the duplicate/absorption/twin/lag failures above.
- **Stage 2 — the clean volumetric map (validate first):** `POSE_MODE=auto` + `TSDF=1`.
  - `auto` (`poses.py`): Open3D RGB-D odometry between nearby frames + sparse loop closures →
    `PoseGraph` → `global_optimization` (Levenberg–Marquardt), node 0 anchored at its nav pose. Every
    edge is sanity-bounded against the nav delta and a final per-node guard reverts any pose that
    wanders past tolerance back to nav — so `auto` can **never do worse than nav**. Needs `KEEP_RGB=1`.
  - `TSDF` (`tsdf.py`): `VoxelBlockGrid` integration with the optimized poses → a clean structural
    cloud (`extract_point_cloud(weight_threshold=3)` keeps only voxels seen by ≥3 frames). Depth-only
    (no RGB needed). Open3D, CUDA preferred.
  - **Measure on a replayed buffer before flipping these on** — a pose graph *can* make poses worse than
    settled nav, which is exactly why `baseline` is the permanent default until proven.

Both Open3D modules are import-guarded and degrade to `baseline` / `None` on any failure, so a box
without Open3D (or CUDA) still maps via Stage 1.

---

## On-robot facts worth knowing

- **Depth is float32 metres** (NaN/≤0 invalid), aligned to the RGB intrinsics. So TSDF/odometry use
  `depth_scale=1.0` (not the usual 1000), and the integration **extrinsic = inv(camera→map pose)**.
- **Trusted depth band ≈ 0.3–4 m** — stereo error grows ~quadratically with range; `MAX_DEPTH_M=4.0`
  drops far, noisy geometry at the source.
- **Pass `max_size=None`** to any detection call whose masks feed 3D (the build does).
- **Pose-at-capture** (sensors + TF read together, before detection) is an architectural invariant — it
  keeps the lift pose matched to the frame even while the robot moves.
- **CUDA** is auto-probed (`WALKIE_GRAPHS_O3D_DEVICE=auto`, `tools/check_gpu`); the Open3D path falls
  back to CPU silently.

---

## Commands

```bash
# Record a run on the robot (the capture thread fills graph_buffer/), then build OFFLINE — no robot,
# fully deterministic — to tune association/poses/TSDF or to A/B baseline vs auto on the SAME buffer:
uv run python -m services.realtime_explore.tools.replay graph_buffer
uv run python -m services.realtime_explore.tools.replay graph_buffer --pose-mode auto --tsdf
uv run python -m services.realtime_explore.tools.replay graph_buffer --store graph_scene   # also persist + print

# Wipe the store + buffer for a clean slate (robot stopped):
uv run python -m services.realtime_explore.tools.reset -y

# Probe Open3D GPU support:
uv run python -m services.realtime_explore.tools.check_gpu

# Unit tests (bare numpy/scipy/sklearn — no robot, no Open3D):
uv run pytest tests/graphs/
```

## Config

All knobs are `WALKIE_GRAPHS_*` in `services/realtime_explore/config.toml` (~46, grouped under
`[graphs] / [graphs.camera] / [graphs.lift] / [graphs.capture] / [graphs.build] / [graphs.assoc] /
[graphs.store] / [graphs.relations]`). The keys *are* the env-var names; precedence is
`shell env > .env > root config.toml > this file > code default`. The load-bearing ones: `INTERVAL_SEC`,
the `*_CLASSES` lists, `CONFIDENCE_THRESHOLD`, the `[graphs.lift]` cleanup knobs, `SNAPSHOT_CAP`,
`REBUILD_EVERY_N` / `BUILD_WINDOW`, `POSE_MODE`, `TSDF`, `KEEP_RGB`, the `ASSOC_*` thresholds,
`MIN_OBS_CONFIRM`, `PRUNE_MAX_RECORDS`, and the `[graphs.relations]` predicates.

## Module map

`service.py` (facade + capture thread + build worker) · `buffer.py` (snapshot ring buffer) ·
`builder.py` (build orchestration) · `poses.py` (baseline/auto pose refinement) · `associate.py` (batch
association) · `scene.py` (`SceneStore` + `ObjectNode`/`Relation` + persistence + queries) ·
`relations.py` (AABB edges) · `tsdf.py` (VoxelBlockGrid map) · `snapshot.py` (perception.json writer) ·
`fusion.py` (`nn_ratio` overlap math) · `pcd_ops.py` (Open3D device/ICP helpers) · `tools/`
(`replay`, `reset`, `check_gpu`). Geometry primitives are reused from `interfaces/perception/`
(`geometry`, `dbscan`, `surfaces`) and `interfaces/devices/camera.py` (`CameraSnapshot`).
