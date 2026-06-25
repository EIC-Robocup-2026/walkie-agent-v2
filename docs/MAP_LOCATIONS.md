# Map locations — the shared named-waypoint layer

Every challenge that drives to a fixed spot ("the dining table", "the laundry
area", "the kitchen bar") now resolves that name through **one shared location
book** instead of a hand-set per-challenge env var. The **map editor**
([`EIC-Robocup-2026/walkie-map-editor`](https://github.com/EIC-Robocup-2026/walkie-map-editor),
a separate web app) writes the location file as a `world.toml`; this repo only
*reads* it.

- **Reader:** [`tasks/skills/locations.py`](../tasks/skills/locations.py) —
  `LocationBook`, `load_location_book`, `get_location_book`, `resolve_pose`.
- **Schema:** the existing GPSR `world.toml` rooms/locations schema (below).
- **GPSR** is built on this layer too: `tasks/GPSR/world.py`'s `WorldModel` loads
  its rooms/locations via the same code (and adds object/name/gesture vocabulary
  on top).

## The file

There are **two readers** with slightly different resolution chains:

- **`LocationBook`** (`resolve_pose`, used by PnP / Restaurant / HRI / Laundry) reads:
  explicit path → `$WALKIE_MAP_FILE` → `$GPSR_WORLD_FILE` → sibling `tasks/GPSR/world.toml`.
- **GPSR `WorldModel`** (`tasks/GPSR/world.py::load_world`, called with no arg in
  `parse.py`/`run.py`) reads: `$GPSR_WORLD_FILE` → sibling. **It does *not* read
  `$WALKIE_MAP_FILE`.**

> ⚠️ **Use `GPSR_WORLD_FILE` (root `config.toml`, `[map]`) — it's the one var both
> readers honour, so it covers all five challenges.** Setting `WALKIE_MAP_FILE`
> alone points the four location challenges at a new map but leaves GPSR reading the
> stale sibling file — treat `WALKIE_MAP_FILE` as a 4-challenge-only override.

**Zero-config alternative:** drop the editor's output in as `tasks/GPSR/world.toml`
(the sibling fallback both readers land on) and set no env var at all. Defaulting to
that one file means **one arena file serves every challenge**, and the existing
`tasks/GPSR/tools/teach_poses.py` (drive the robot, record its current pose) already
feeds them all.

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
```

- `pose = [x, y, heading_rad]` in the **map frame**. `[0,0,0]` = "not yet surveyed".
- `aliases` — extra phrasings; lookup is also article- and whitespace-tolerant and
  fuzzy (`"dining tabel"` still grounds), so the editor needn't be exhaustive.
- `barrier = true` — a human-operated door/partition blocks the route here (used by
  GPSR's barrier-aware nav today; available to other challenges as a follow-up).
- `present = false` — in the template but **not** in the running arena; the entry is
  dropped, and any location whose room was dropped cascades out too.

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
| Restaurant | `kitchen_bar` | `RESTAURANT_KITCHEN_BAR_POSE` |
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
- **Restaurant** keeps its `"current"` sentinel: with no `kitchen_bar` in the map
  *and* `RESTAURANT_KITCHEN_BAR_POSE` unset/`"current"`, the robot anchors the bar on
  wherever it stands and does **not** drive (rulebook 5.5 — the arena isn't
  pre-mapped). Define `kitchen_bar` (or set the env pose) to make it drive.

### NOT in the map (arm-frame poses, deliberately excluded)

These are 6-DOF **arm/end-effector** place poses consumed by
`tasks/manipulation.py::place_at_pose`, not base-frame nav waypoints — they stay as
env vars and must never be added to the location book:
`PNP_PLACE_POSE_{DISHWASHER,TRASH,CABINET,DEFAULT}` and the breakfast arm slots
`PNP_BREAKFAST_{BOWL,SPOON,MILK,CEREAL}_POSE`. (`HRI_FOLLOW_POSE_KP_CONF` is a
keypoint-confidence threshold, also unrelated.)
