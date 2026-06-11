# Walkie Graphs — the robot's spatial memory

`services/walkie_graphs/` is the part of Walkie that **remembers what it has seen and where**.

As the robot moves around, its camera keeps spotting objects — a mug, a chair, a bottle.
Walkie Graphs takes each sighting, works out *where in the room* the object is, and keeps a
running 3D map of everything it knows. Later you can ask it questions like *"where's the
mug?"* or *"what's near the table?"* and it answers from that map.

Think of it as the robot's **memory of places and things** — not the eyes (that's the camera
and the detection server), just the memory that the eyes fill in over time.

> It's modeled on a research system called **ConceptGraphs** (an open-vocabulary 3D scene-graph
> method). If you've read that paper, this is a real-time, on-robot reimplementation of its
> data-processing pipeline. If you haven't, you don't need to — this doc stands alone.

---

## What it does, in one picture

```
   Camera sees a frame            Walkie Graphs                You ask a question
  ┌───────────────────┐      ┌────────────────────────┐     ┌────────────────────┐
  │  📷 RGB + depth    │ ───► │  1. find objects       │     │ "where is the mug?"│
  │  "there's a mug"   │      │  2. place them in 3D   │ ──► │        ▼           │
  └───────────────────┘      │  3. remember / update  │     │ "on the table,     │
                             │  4. link them together  │     │  1.2 m to my right"│
                             └────────────────────────┘     └────────────────────┘
                                        │
                                   long-term map
                                (objects + where + how they relate)
```

Three things make it useful:

1. **It fuses many glimpses into one object.** See the same mug from five angles across ten
   seconds and you get **one** remembered mug, not ten — with a position that gets more accurate
   each time.
2. **It understands space.** It knows the mug is *on* the table and *near* the kettle, because it
   reasons about the actual 3D shapes.
3. **It answers in plain language.** Other parts of Walkie (the "database agent") query it to help
   the robot find things and plan tasks.

---

## How a sighting becomes a memory

Every few seconds the robot processes one camera frame. Here's the journey of a single detected
object, in plain terms — each step expands if you want the real mechanics.

### 1. See it
The detection server finds objects in the image and outlines each one with a **mask** (a precise
pixel outline, not just a box).

<details>
<summary>Details — detection &amp; filtering</summary>

- The perception loop ([services/perception.py](services/perception.py)) captures an RGB frame +
  an aligned depth image, then calls the detection server with the list of "interested classes"
  as open-vocabulary prompts and `return_mask=True`.
- Walkie Graphs is therefore the **CG-Detect** flavour of ConceptGraphs: an open-vocabulary
  detector + masks, rather than class-agnostic SAM segmentation.
- Before doing any 3D work, two cheap **size filters** drop junk
  ([services/walkie_graphs/service.py](services/walkie_graphs/service.py), `_passes_size_filters`):
  - **`MAX_BBOX_AREA_RATIO`** — a box covering most of the frame is almost always a wall/floor/
    background misfire, so it's rejected.
  - **`MIN_MASK_AREA_PX`** — masks too small to be a real object are dropped.
- **Containment subtraction** (`MASK_SUBTRACT`, ConceptGraphs' `mask_subtract_contained`): when one
  detection sits inside another — a mug on a table — the mug's pixels are removed from the table's
  mask before any 3D work, so the table's point cloud and image crop aren't polluted by the objects
  resting on it (`subtract_contained_masks` in [services/walkie_graphs/fusion.py](services/walkie_graphs/fusion.py)).
- Then per-class scoping (`_keep`): an `EXCLUDE_CLASSES` list (default `person` — people move and
  can't be position-mapped) and an optional `INTERESTED_CLASSES` allow-list.
</details>

### 2. Place it in 3D
Using the depth image and where the camera is pointing, each pixel of the mask is turned into a
point in the room. The result is a little **3D point cloud** shaped like the object.

<details>
<summary>Details — depth back-projection &amp; camera pose</summary>

The camera math lives in [services/walkie_graphs/geometry.py](services/walkie_graphs/geometry.py) (pure numpy); the
service feeds it real calibration and pose straight from the **walkie-sdk**.

- **Intrinsics** (`_intrinsics` in service.py): `bot.camera.get_intrinsics()` returns the ZED's true
  pinhole `fx, fy, cx, cy` from its `CameraInfo` (cached by the SDK — intrinsics are static — and
  registered to the depth image too). If the depth stream is a different resolution, the intrinsics
  are rescaled (`Intrinsics.scaled_to`). No field-of-view guessing.
- **Camera pose** (`_camera_pose`): `bot.transform.lookup("map", "<camera>_optical_frame")` returns
  the camera **optical** frame's pose in the map frame — lift height, head tilt, and every mount
  offset already baked in by the TF tree. The rotation (built from the quaternion via the SDK's
  `quaternion_to_matrix`) maps camera-optical points *straight into the map*. If the lookup fails,
  the tick is skipped.
- **Deprojection** (`deproject_mask`): every masked pixel with valid depth is back-projected with
  the pinhole model `X = (u−cx)·d/fx` into the optical frame, then mapped into the world in one step,
  `P_map = P_optical @ R.T + t`. Because the lookup already uses the *optical* frame (whose axes —
  `x right, y down, z forward` — match the pinhole math), there's no intermediate axis swap or manual
  lift/tilt composition. The cloud is voxel-downsampled (`VOXEL_M`, 2 cm) and capped at
  `MAX_POINTS_PER_OBJ` (2000).
- **Config**: just the two TF frame names (`TF_MAP_FRAME`, `TF_CAMERA_FRAME`) and a lookup timeout —
  the calibration and mounts come from the robot, not config.
</details>

### 3. Clean it up
Depth cameras are noisy — at an object's edge, a pixel can mix the object with the wall behind it
and report an in-between distance, so the point lands *behind* the object like a shadow ("flying
pixels"). Two filters remove this at the source, then DBSCAN mops up anything left, so the object's
size and position stay accurate.

<details>
<summary>Details — flying-pixel edge cleanup</summary>

These run during back-projection ([services/walkie_graphs/geometry.py](services/walkie_graphs/geometry.py)), where the
depth discontinuity is still visible in 2D — much more reliable than trying to spot the smear in the
finished 3D cloud:

- **Mask erosion** (`MASK_ERODE_PX`, default 2): shrink each object's mask inward by a couple of
  pixels, dropping the unreliable rim along the silhouette where foreground and background mix.
- **Depth-discontinuity rejection** (`DEPTH_EDGE_THRESH_M`, default 0.05 m): `depth_discontinuity_mask`
  flags every pixel that borders a depth jump bigger than the threshold — the flying pixels sit
  exactly on these jumps — and they're dropped. It's computed once per frame and shared by all
  detections. Set either knob to 0 to disable it.
</details>

<details>
<summary>Details — 3D denoising (SOR + DBSCAN, the backstops)</summary>

Two complementary 3D filters live in [services/walkie_graphs/dbscan.py](services/walkie_graphs/dbscan.py),
each with a library fast path and a pure scipy fallback:

- **Statistical outlier removal** (`statistical_outlier_removal`, Open3D's C++
  `remove_statistical_outlier` — the library ConceptGraphs builds on): drops points sitting in
  anomalously sparse space (large mean distance to their `SOR_K` nearest neighbours). It runs on
  every lifted cloud at deprojection AND periodically over each stored object's **accumulated**
  cloud — the second pass is what erases the fuzzy halo that builds up across sightings (each
  frame's few surviving edge artifacts) and stops neighbouring objects' inflated clouds from
  bleeding into each other. Density-based, so disjoint multi-view clusters survive.
- **DBSCAN clustering** (scikit-learn fast path, cKDTree + union-find fallback):
  - *per detection* (`GraphMemory._denoise`): keeps only the **largest cluster** — exactly
    ConceptGraphs' `pcd_denoise_dbscan`; a single view of one object is one blob.
  - *periodically* (`denoise_nodes`, after SOR): drops remaining **noise points** (isolated
    scatter, no cluster), keeping *every* real cluster — a cloud accumulated from disjoint
    partial views (the two ends of a bed) is legitimately multi-cluster and must never be
    truncated to its newest view.
- A safety rule: if the combined cleanup would throw away most of the points, the node is
  skipped (`DENOISE_KEEP_MIN_FRAC`).
</details>

### 4. Describe it
The object's image crop is turned into two things: a short **caption** ("a white ceramic mug") and
a **CLIP embedding** — a list of numbers that captures what it looks like, so similar-looking
things can be compared mathematically.

<details>
<summary>Details — captions &amp; CLIP embeddings</summary>

- The crop is sent to the caption server (`image_caption.caption_batch`) and the CLIP image-embed
  server (`image_embed.embed_image`), both via the AI client.
- The **CLIP embedding** is the key to recognising the same object again and to text search: a
  text query like "the mug" is embedded with the *same* CLIP model, so its numbers land near the
  mug's numbers (cross-modal search).
- If the embed server is unavailable, the system degrades gracefully — matching falls back to pure
  geometry, and text search falls back to keyword matching on captions.
</details>

### 5. Match it, or add it
This is the heart of the system. Walkie Graphs checks: *"Have I seen this object before?"*

- If the new point cloud **overlaps in space** with a remembered object **and** looks similar, the
  two are **merged** — the memory is updated, not duplicated.
- If nothing matches, it's added as a **new** object.

<details>
<summary>Details — the association algorithm (the core of ConceptGraphs)</summary>

Implemented in [services/walkie_graphs/memory.py](services/walkie_graphs/memory.py) (`_associate`) and
[services/walkie_graphs/fusion.py](services/walkie_graphs/fusion.py).

Each new detection is scored against existing **same-class** objects that are nearby (a cheap
bounding-box / radius prefilter keeps this fast even with hundreds of objects). The score combines
two cues, exactly as in the paper:

```
phi = W_GEO · nn_ratio  +  W_SEM · (0.5 · cosine + 0.5)
```

- **`nn_ratio`** (geometry): the fraction of the detection's points that have a neighbour in the
  stored object's cloud within `NN_VOXEL_M` (2.5 cm). This is *physical overlap* — far sharper than
  comparing single centre points, which can't tell a mug from the table beneath it.
- **`cosine`** (appearance): CLIP similarity, rescaled to `[0, 1]`.

Both terms live in `[0, 1]`, so the combined score is in `[0, 2]`. The detection merges into the
**highest-scoring** object whose score clears `SIM_THRESHOLD` (default **1.1**); otherwise it's a
new object. With equal weights, a *purely visual* match tops out at 1.0 < 1.1, so this path only
ever fires on **real geometric overlap**.

**Why it can match across class labels.** The detector's class names flip-flop — the same object
can be "cup" one frame and "mug" the next, which would otherwise create a duplicate. So a candidate
of a *different* class may also merge, but only past a **stricter** gate
(`CROSS_CLASS_SIM_THRESHOLD`, default 1.5 ≈ near-full physical overlap *and* agreeing appearance).
ConceptGraphs goes further and ignores classes entirely; the stricter gate keeps the label as a
soft prior instead. Set it to 0 to forbid cross-class merging.

**Why there's a fallback.** A re-sighting whose depth drifted (so the clouds no longer overlap)
would be missed by overlap alone. So when `_associate` finds no geometric match, the original
walkie matcher takes over (`_classify`): it merges on high CLIP similarity within a tight distance,
which recovers drifted re-sightings without fusing two genuinely-different look-alikes. Tuning knobs:
`SIM_HIGH`, `SIM_LOW`, `DEDUP_TIGHT_M`, `DEDUP_RADIUS_M`, `DEDUP_VISUAL_K`.
</details>

<details>
<summary>Details — what "merge" actually updates</summary>

When a detection merges into an existing object (`_merge`):
- **Point cloud**: whenever the new points geometrically **overlap** the stored cloud (or land
  near it), the two clouds are **unioned**, voxel-downsampled, and capped — so a *partial* view
  of a large object (one end of the bed) adds to the accumulated cloud rather than replacing it,
  and the object fills in across sightings. Old points are never discarded by a re-sighting.
- **CLIP embedding**: blended as a running average weighted by how many times the object's been
  seen (so one bad frame can't hijack it).
- **Captions**: accumulated; the longest is kept as `best_caption` (until the optional LLM step
  rewrites it — see below).
- **Confidence & timestamps**: updated; `n_obs` (sighting count) increments.
- The only exception: a matched re-sighting that is far away AND doesn't overlap the stored
  cloud at all (a drifted depth estimate, or an object that physically moved). There the system
  keeps the higher-confidence geometry instead of smearing one object across two places.
</details>

### 6. Link it to its neighbours
Periodically, Walkie Graphs works out how objects **relate** — the mug is *on* the table, the fork
is *inside* the drawer, the kettle is *near* the mug.

<details>
<summary>Details — spatial relations (edges)</summary>

`GraphMemory.derive_relations` recomputes geometric edges from the objects' bounding boxes
(see [services/walkie_graphs/memory.py](services/walkie_graphs/memory.py)). For each pair within `RELATION_MAX_DIST`:

| Relation  | Rule |
|-----------|------|
| `near`    | centres within `NEAR_M` (default 0.6 m); weighted by closeness |
| `on`      | footprints overlap (`XY_OVERLAP_MIN`) and one sits just above the other (gap ≤ `ON_GAP`) |
| `above`   | same footprint overlap but a larger vertical gap |
| `inside`  | one box is contained within a larger one (`CONTAIN_TOL`) |

These are *directed* (mug→table "on", not table→mug) and stored in a NetworkX graph mirrored to
`graph_edges.json`. They're richer than ConceptGraphs' base relations, which only has on/in via an
LLM — here they're computed directly from geometry, so they're free and instant.
</details>

### 7. Store it
The object is saved so it survives and can be searched.

<details>
<summary>Details — storage layout</summary>

Each remembered object is one `ObjectNode` (id, class, centre, bounding box, CLIP embedding,
captions, sighting count, timestamps, …). It's stored across three places:

- **ChromaDB** (`CHROMA_DIR`, default `chroma_db_graph/`) — the CLIP embedding + all scalar
  metadata, in a cosine-similarity collection. This is what powers text search.
- **`.npz` sidecars** (`PCDS_DIR`, default `graph_pcds/`) — one point cloud per object.
- **`graph_edges.json`** (`EDGES_PATH`) — the relations, mirrored from the in-memory NetworkX graph.

> ⚠️ Single-process only. ChromaDB's persistent store isn't safe for concurrent writers, so only
> the robot process writes here. Viewer/debug tools open a snapshot copy.
</details>

---

## Keeping the map clean over time

A live robot accumulates clutter — duplicates, noise, one-off false detections. A few background
jobs run on a slow cadence (every ~20 frames, staggered so they never pile up) to tidy the map.

<details>
<summary>Details — the maintenance passes</summary>

All in [services/walkie_graphs/memory.py](services/walkie_graphs/memory.py), scheduled by `_maybe_tick` in
[services/walkie_graphs/service.py](services/walkie_graphs/service.py):

- **`merge_overlapping_nodes`** — fuses two objects that turned out to be the same thing seen from
  different sides (high cloud overlap + similar appearance). This is the cleanup the per-frame
  matcher can't do, since at first sighting the two clouds were on opposite faces. Candidate pairs
  come from a KD-tree radius query, so the pass stays fast even on a full map. Cross-class pairs
  (label flip-flops) are eligible when `CROSS_CLASS_SIM_THRESHOLD` > 0, and those additionally
  *require* CLIP agreement — geometry alone never overrides the labels.
  Knobs: `MERGE_OVERLAP_THRESH`, `MERGE_VISUAL_SIM_THRESH`, `MERGE_RADIUS_M`.
- **`denoise_nodes`** — re-runs DBSCAN on objects whose cloud has grown, clearing accreted
  cross-view noise (with the "don't gut a spread object" guard).
- **`evict_stale_provisional`** — deletes flicker/false-positive objects that were seen once and
  never again (only if `GHOST_GRACE_SEC` > 0; off by default).
- **`prune`** — capacity cap; evicts the oldest objects beyond `PRUNE_MAX_RECORDS` (500).

These do the heavy read-only computation on a lightweight snapshot **outside** the lock, then take
the lock only briefly to commit — so they never stall the agent's queries.
</details>

### Node precision: confirmed vs. provisional

By default Walkie Graphs only trusts an object once it's been seen a few times. A one-frame
detection is remembered but treated as **provisional** and **hidden from answers** until it's
re-confirmed. This is what keeps "did I really see that?" flickers out of the robot's answers.

<details>
<summary>Details — confirmation gate</summary>

- An object is **confirmed** once `n_obs ≥ MIN_OBS_CONFIRM` (default **3**, matching ConceptGraphs'
  `obj_min_detections`).
- With `REQUIRE_CONFIRMATION = 1` (the production default), provisional objects are filtered out of
  `query_text`, `query_near`, `recently_seen`, `all_objects`, and `to_text_description`. They are
  **not deleted** — a re-sighting promotes them — and `count()` and the 3D viewer still show
  everything.
- **Trade-off:** this means a genuinely new object won't appear in answers until it's been seen ~3
  times. If you'd rather see everything immediately while watching the robot, set
  `WALKIE_GRAPHS_REQUIRE_CONFIRMATION = 0` in `config.toml`.
</details>

---

## Asking it questions

Other parts of Walkie (the **database agent**, [agents/database_agent/](agents/database_agent/))
call into the map through a small set of methods on the `WalkieGraphs` facade:

| Question | Method | What it does |
|----------|--------|--------------|
| "Where is the mug?" | `query_text` | CLIP text-search over objects (keyword fallback) |
| "What's near me / near here?" | `query_near` | objects within a radius of a point |
| "What did I just see?" | `recently_seen` | most recently observed objects |
| "What do you know about?" | `all_objects` | the full (confirmed) inventory |
| "Describe the scene" | `to_text_description` | a plain-text dump of objects + relations for an LLM |

<details>
<summary>Details — how text search works</summary>

`query_text` embeds the query string with CLIP and searches the ChromaDB collection by cosine
similarity (text-vector vs. each object's image-vector — cross-modal). If the embed server is down,
it falls back to keyword overlap on the stored captions and class names, labelled accordingly.
Results can be spatially filtered (`near`, `radius`) and are run through the confirmation gate.
</details>

---

## The optional LLM layer (off by default)

For higher-quality output, Walkie Graphs can use a language model to:

1. **Rewrite captions** — combine an object's many rough per-view captions into one clean label
   ("a white ceramic coffee mug").
2. **Infer richer relations** — label relationships between nearby objects in natural language.

These are **off by default** (they cost API calls) and never block the robot — turn them on per
deployment.

<details>
<summary>Details — caption refinement &amp; LLM edges</summary>

Both live in [services/walkie_graphs/memory.py](services/walkie_graphs/memory.py) and use the chat model already
threaded into `WalkieGraphs`:

- **`refine_captions(model)`** — summarises each object's accumulated captions into one coherent
  noun phrase (the ConceptGraphs node-captioning step). Text-only by default; can also send the
  object's best crops to a multimodal model (`CAPTION_REFINE_USE_IMAGES`). It stores up to
  `BEST_VIEWS` highest-confidence crops per object to support this.
- **`infer_edges_llm(model)`** — builds a minimum spanning tree over nearby objects and asks the
  model to label each adjacency, storing accepted ones as separate `llm:<label>` edges. The
  geometric `near/on/above/inside` edges stay primary and are never overwritten.
- Enable via cadence knobs `CAPTION_REFINE_EVERY_N` and `LLM_EDGES_EVERY_N` (0 = off), or trigger
  on demand via `WalkieGraphs.refine_captions()` / `WalkieGraphs.infer_edges()`. The model calls
  always happen **outside the lock**, so a slow API response can't freeze perception.
</details>

---

## Configuration

Every knob is a `WALKIE_GRAPHS_*` setting in [config.toml](config.toml), grouped into tables. The
TOML keys *are* the environment-variable names; precedence is shell env > `.env` > `config.toml` >
code default.

<details>
<summary>The config tables at a glance</summary>

| Table | What it controls |
|-------|------------------|
| `[graphs]` | enable flag, tick interval, class scoping, the Rerun 3D visualizer |
| `[graphs.camera]` | the two transform-tree frame names + lookup timeout (calibration comes from the SDK) |
| `[graphs.fusion]` | **matching & denoising** — `SIM_THRESHOLD`, `W_GEO`/`W_SEM`, `NN_VOXEL_M`, DBSCAN, size filters, the legacy cascade |
| `[graphs.maintenance]` | **map cleanup** — re-merge thresholds, denoise/ghost cadences, the confirmation gate |
| `[graphs.relations]` | geometric edge thresholds (`NEAR_M`, `ON_GAP`, …) |
| `[graphs.store]` | storage paths and the capacity cap |
| `[graphs.semantic]` | the optional LLM layer (all default off) |

The defaults are tuned to match ConceptGraphs' quality settings (3-sighting confirmation,
background-box rejection, DBSCAN on). See the comments in `config.toml` for each key.
</details>

---

## Architecture &amp; files

<details>
<summary>What each file does</summary>

| File | Role |
|------|------|
| [services/walkie_graphs/__init__.py](services/walkie_graphs/__init__.py) | `WalkieGraphs` facade — ties store + observer + visualizer together; what the rest of the app uses |
| [services/walkie_graphs/service.py](services/walkie_graphs/service.py) | `WalkieGraphsService` — the per-frame ingestion pipeline + maintenance scheduling |
| [services/walkie_graphs/memory.py](services/walkie_graphs/memory.py) | `GraphMemory` — the store: association, merging, relations, queries, maintenance, persistence |
| [services/walkie_graphs/geometry.py](services/walkie_graphs/geometry.py) | camera math — intrinsics, pose, depth→world deprojection |
| [services/walkie_graphs/fusion.py](services/walkie_graphs/fusion.py) | association math — `nn_ratio` overlap, AABB prefilter, additive score |
| [services/walkie_graphs/dbscan.py](services/walkie_graphs/dbscan.py) | point-cloud denoising (Open3D statistical outlier removal + DBSCAN clustering, with scipy fallbacks) |
| [services/walkie_graphs/viz.py](services/walkie_graphs/viz.py) | optional real-time 3D visualization via Rerun |
| [services/walkie_graphs/tools/reset.py](services/walkie_graphs/tools/reset.py) | CLI to wipe the store |

**Data flow:** `PerceptionService` → `WalkieGraphs.ingest_frame` → `WalkieGraphsService.ingest_frame`
→ `GraphMemory.upsert`. Queries flow `database_agent` → `WalkieGraphs` → `GraphMemory`. The module
depends on the **AI client** (detection, caption, CLIP embed) and the **walkie-sdk robot**
(RGB + depth, camera intrinsics, and the camera optical-frame transform) — both passed in, never
created here.
</details>

<details>
<summary>The two data structures</summary>

- **`Detection3D`** — one masked detection lifted to 3D (input to `upsert`): class, confidence,
  bbox, world point cloud, CLIP embedding, caption, crop.
- **`ObjectNode`** — one remembered object (a graph node): id, class, centroid, extent, axis-aligned
  bounding box, fused CLIP embedding, all captions + `best_caption`, sighting count `n_obs`,
  confidence, first/last-seen timestamps, and references to its point cloud + thumbnail(s).
</details>

---

## Running &amp; developing

<details>
<summary>Visualize the map live</summary>

Set `WALKIE_GRAPHS_VIZ = "rerun"` in `config.toml` (needs `uv sync --extra graphs`) to stream the
point clouds, bounding boxes, relations, and the robot/camera pose to a [Rerun](https://rerun.io)
viewer — a local window on the robot, or a browser viewer over the LAN
(`WALKIE_GRAPHS_RERUN_SERVE = "1"`).
</details>

<details>
<summary>Reset the store</summary>

```bash
uv run python -m services.walkie_graphs.tools.reset      # confirm first
uv run python -m services.walkie_graphs.tools.reset -y   # no prompt
```

Wipes the ChromaDB collection, point-cloud sidecars, thumbnails, and the edges file. Run with the
robot stopped.
</details>

<details>
<summary>Tests</summary>

The whole pipeline is unit-tested without a robot or server (pure numpy/scipy, fake embeddings and
a fake LLM):

```bash
uv run pytest tests/graphs -q
```

- `test_fusion.py` / `test_dbscan.py` — the association and denoising math
- `test_memory.py` — fusion/dedup, queries, persistence
- `test_relations.py` — the geometric edges
- `test_maintenance.py` — re-merge, denoise, confirmation, ghost eviction
- `test_semantic.py` — caption refinement + LLM edges (with a fake model)
- `test_service.py` — detection filters + maintenance scheduling
- `test_geometry.py` — camera math
</details>

---

## How this compares to ConceptGraphs

<details>
<summary>Mapping to the paper, and where Walkie differs</summary>

| ConceptGraphs | Walkie Graphs |
|---------------|---------------|
| Class-agnostic SAM masks | Open-vocabulary detector + masks (the "CG-Detect" variant) |
| CLIP per-mask features (20 px padded crops) | Same (CLIP ViT-B/16 image embeddings, same 20 px crop margin) |
| `mask_subtract_contained` (containers don't absorb their contents) | Same (`subtract_contained_masks`) |
| Depth back-projection + DBSCAN denoise | Same (`geometry.py` + `dbscan.py`; sklearn fast path, scipy fallback) |
| Association = point overlap (`nn_ratio`) + CLIP, additive, greedy, **class-agnostic** | Same (`fusion.py` + `_associate`) — cross-class merges pass a stricter gate (label as a soft prior), **plus** a visual fallback that recovers drifted re-sightings ConceptGraphs would duplicate |
| Periodic merge of overlapping objects | `merge_overlapping_nodes` (KD-tree pair prefilter) |
| Filter objects seen < 3 times | The confirmation gate (hides rather than deletes, so it's reversible) |
| LLM node captions (best-N views) | `refine_captions` (optional) |
| LLM scene-graph edges via MST | `infer_edges_llm` (optional) — **plus** always-on geometric `near/on/above/inside` edges the paper doesn't have |
| Offline, batch, GPU (open3d/faiss/torch) | Online, incremental, numpy/scipy/sklearn — runs on the robot inside the perception loop |

In short: the same data-processing backbone, adapted to run live on the robot, with a few
robustness additions (the drift-recovery fallback, reversible confirmation, free geometric
relations, in-memory cloud caching) on top.
</details>
