# Map locations — the shared named-waypoint layer

Every challenge that drives to a fixed spot ("the dining table", "the laundry
area", "the kitchen bar") now resolves that name through **one shared location
book** instead of a hand-set per-challenge env var. The **map editor**
([`EIC-Robocup-2026/walkie-map-editor`](https://github.com/EIC-Robocup-2026/walkie-map-editor),
a separate web app) writes the location file as a `world.toml`; this repo only
*reads* it.

- **Reader:** [`walkie_world/map/locations.py`](../walkie_world/map/locations.py) —
  `LocationBook`, `load_location_book`, `get_location_book`, `resolve_pose`.
- **Schema:** the `world.toml` rooms/locations/doors/objects schema (below).
- **GPSR** is built on this layer too: [`walkie_world/map/vocab.py`](../walkie_world/map/vocab.py)'s
  `WorldModel` loads its rooms/locations via the same code (and adds object/name/gesture
  vocabulary on top). Both reach `ctx.world`.

## The file

There is **ONE global map** shared by every challenge — no per-task map. Both readers
(the `LocationBook` and the GPSR `WorldModel`) resolve through the *same* chain:

> **explicit path → `$WALKIE_MAP_FILE` → the repo-root `world.toml`.**

`WALKIE_MAP_FILE` (root `config.toml`, `[map]`) is the **one canonical knob**: because
both readers honour it, the map and the GPSR vocabulary can never read different files.
(The legacy `GPSR_WORLD_FILE` var was removed — set `WALKIE_MAP_FILE` instead.)

**Zero-config:** drop the editor's output in as the repo-root `world.toml` and set no
env var at all. One arena file serves every challenge, and
`tasks/GPSR/tools/teach_poses.py` (drive the robot, record its current pose) edits that
same file.

If the file is **missing**, the book is empty (never raises) — each challenge then
falls back to its own `*_POSE` env var, so a dev box with no map keeps running.
(GPSR's `load_world` is the one exception: its arena vocabulary is mandatory, so it
raises `FileNotFoundError` when no file resolves — a missing GPSR arena is a setup
error worth failing loudly on.)

### Schema (TOML — same as `world.toml`)

```toml
[rooms]
kitchen      = { pose = [1.0, 2.0, 0.5] }
living_room  = { pose = [5.0, -1.0, 1.57], aliases = ["lounge"], barrier = true }
office       = { pose = [0.0, 0.0, 0.0], present = false }   # absent -> dropped

[locations]
dining_table = { room = "kitchen", pose = [1.5, 2.5, 3.14], aliases = ["dinner table"] }
kitchen_bar  = { room = "kitchen", pose = [0.2, 0.3, -1.0] }

[doors]
entrance     = { pose = [0.0, 0.0, 1.57], radius = 1.5 }   # a physical door
```

- `pose = [x, y, heading_rad]` in the **map frame**. `[0,0,0]` = "not yet surveyed".
- `aliases` — extra phrasings; lookup is also article- and whitespace-tolerant and
  fuzzy (`"dining tabel"` still grounds), so the editor needn't be exhaustive.
- `barrier = true` — a human-operated door/partition blocks the route here (used by
  GPSR's barrier-aware nav today; available to other challenges as a follow-up).
- `present = false` — in the template but **not** in the running arena; the entry is
  dropped, and any location whose room was dropped cascades out too.

### `[doors]` — physical door locations (proximity-gated asking)

Optional. Each entry is a door's geographic position, drawn with the map editor's
**Door** tool. When the map defines *any* door, the shared door-opening skill
([`tasks/skills/door.py`](../tasks/skills/door.py) `go_to_through_door`) asks for a
door **only where one is mapped** — a mapped door within `WALKIE_DOOR_NEAR_RADIUS_M`
(default 1.5 m) of the robot — instead of on every nav block, so it won't pester a
human at a cabinet/wall that merely reads "closed". With no `[doors]` (or
`WALKIE_DOOR_MAP_GATE=0`) the gate is inert and the depth check decides alone — the
original behaviour, so map-less arenas are unchanged.

- `pose = [x, y, heading_rad]` — `heading` is the passage direction (display-only).
- `radius` — optional per-door trigger radius (m); overrides `WALKIE_DOOR_NEAR_RADIUS_M`.
- `present = false` — drop a door not in the running arena (like rooms/locations).
- Read via `LocationBook.doors` / `door_near()` / `nearest_door()`; doors are matched
  by **proximity**, never by name (they are not navigation destinations). This is
  distinct from `barrier` (above), which gates per *destination* rather than by geometry.

## Name-resolution contract per challenge

Each challenge maps its waypoint(s) to a canonical location name, looked up
book-first with the old env var as fallback (`resolve_pose(name, env_fallback=...)`).

| Challenge | location name | env-var fallback |
|---|---|---|
| PnP | `kitchen` | `PNP_KITCHEN_POSE` |
| PnP | `dining_table` | `PNP_DINING_TABLE_POSE` |
| PnP | `dishwasher` | `PNP_DISHWASHER_POSE` |
| PnP | `cabinet` | `PNP_CABINET_POSE` |
| PnP | `trash_bin` | `PNP_TRASH_BIN_POSE` |
| PnP | `breakfast_surface` | `PNP_BREAKFAST_SURFACE_POSE` |
| PnP | `extra_surface` | `PNP_EXTRA_SURFACE_POSE` |
| Laundry | `laundry_area` | `LAUNDRY_AREA_POSE` |
| Laundry | `laundry_basket` | `LAUNDRY_BASKET_POSE` |
| Laundry | `folding_table` | `LAUNDRY_TABLE_POSE` |
| Laundry | `washing_machine` | `LAUNDRY_WASHER_POSE` |
| HRI | `entrance_door` | `HRI_DOOR_POSE` |
| HRI | `living_room` | `HRI_LIVING_ROOM_POSE` |

GPSR resolves arbitrary rooms/locations by name directly through its `WorldModel`
(`world.location_pose`), so it has no fixed table — the operator's command names the
place.

Notes:
- **Restaurant is deliberately NOT in the table above** — it never reads the shared
  LocationBook. Rulebook 5.5 forbids a pre-mapped arena, so `GoToStart` anchors the
  bar on wherever the robot is standing and does **not** drive. A `kitchen_bar`
  waypoint in `world.toml` (e.g. defined for GPSR) is **ignored** by Restaurant. The
  only override is an explicit `RESTAURANT_KITCHEN_BAR_POSE = "x,y,heading_rad"` env
  pose (manual bring-up); its `"current"` sentinel keeps the anchor-in-place default.

### NOT in the map (arm-frame poses, deliberately excluded)

These are 6-DOF **arm/end-effector** place poses consumed by
`tasks/manipulation.py::place_at_pose`, not base-frame nav waypoints — they stay as
env vars and must never be added to the location book:
`PNP_PLACE_POSE_{DISHWASHER,TRASH,CABINET,DEFAULT}` and the breakfast arm slots
`PNP_BREAKFAST_{BOWL,SPOON,MILK,CEREAL}_POSE`. (`HRI_FOLLOW_POSE_KP_CONF` is a
keypoint-confidence threshold, also unrelated.)
