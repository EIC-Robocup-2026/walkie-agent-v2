"""Shared named-location ("map") layer — the arena waypoints challenges drive to.

A :class:`LocationBook` is the rooms + locations subset of the GPSR ``world.toml``
schema: each entry carries a map-frame ``pose = [x, y, heading_rad]``, optional
``aliases``, a ``barrier`` flag (a human-operated door/partition blocks the route),
and a ``present`` flag (drop entries absent from the running arena). An external
**map editor** writes this file; every challenge reads it through
:func:`resolve_pose`.

This is the shared home for the location primitives GPSR's ``WorldModel``
(``tasks/GPSR/world.py``) is built on — WorldModel imports them from *here*, never
the reverse. Pure and offline: no robot, no LLM, no network.

Resolution + fallback (so a dev box with no map still runs): :func:`resolve_pose`
tries the book first, then the challenge's existing ``*_POSE`` env var, then a
literal default. :func:`load_location_book` returns an **empty book (never raises)**
when no map file exists — unlike GPSR's ``load_world``, which fails loudly because
its arena vocabulary is mandatory.
"""

from __future__ import annotations

import difflib
import math
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from walkie_world.map.polygon import point_in_polygon

Pose = tuple[float, float, float]
# An ordered XY polygon (CCW, implicitly closed) as a tuple of (x, y) vertices.
Polygon = tuple[tuple[float, float], ...]


def parse_pose(s: str) -> tuple[float, float, float]:
    """Parse a waypoint string "x,y,heading_rad" -> (x, y, heading_rad).

    Inlined (not imported from ``tasks.skills.geometry``) so the world model has
    no back-edge into the ``tasks`` layer — ``tasks`` depends on ``walkie_world``,
    never the reverse. ``tasks.skills.geometry`` keeps its own copy for its callers.
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 'x,y,heading_rad', got {s!r}")
    x, y, heading_rad = (float(p) for p in parts)
    return x, y, heading_rad


# --- pure text matching (shared by LocationBook and GPSR's WorldModel) ------

def _fuzzy_cutoff() -> float:
    """difflib ratio an STT/LLM near-miss must clear to match a known name
    (GPSR_GROUNDING_FUZZY_CUTOFF, default 0.84). 0 / unparsable disables fuzzy
    matching (exact-alias behaviour only).

    0.84, not 0.8: genuine STT/LLM misspellings of a vocab word score >=0.92
    ("kitchen tabel" 0.92, "cabinett" 0.93), while short DISTINCT English words
    collide at exactly 0.8 ("plant" vs "plate") — at 0.8 the robot grounded
    "locate the plant" to the plate and searched for the wrong thing."""
    try:
        return float(os.getenv("GPSR_GROUNDING_FUZZY_CUTOFF", "0.84") or 0)
    except ValueError:
        return 0.0


def _fuzzy_match(key: str, candidates, cutoff: float) -> str | None:
    """Closest candidate to *key* at/above *cutoff* (difflib ratio), or None.

    Belt-and-suspenders for STT/LLM near-misses ("kitchen tabel" -> kitchen_table)
    that survive vocab-grounded extraction and would otherwise be an ungrounded
    gap. Only consulted on an exact-lookup miss, so it can never override a correct
    exact match. ``cutoff <= 0`` disables it. Pure -> unit-tested directly."""
    if not key or cutoff <= 0:
        return None
    hits = difflib.get_close_matches(key, list(candidates), n=1, cutoff=cutoff)
    return hits[0] if hits else None


def _norm(s: str) -> str:
    """Canonical key for matching: lowercase, spaces/hyphens->underscore, trimmed.

    "the Kitchen Table" -> "the_kitchen_table"; callers strip leading articles.
    """
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^\w]", "", s)
    return s


_ARTICLES = ("the_", "a_", "an_", "some_", "my_", "your_")


def _strip_article(key: str) -> str:
    for art in _ARTICLES:
        if key.startswith(art):
            return key[len(art):]
    return key


def _alias_keys(canonical: str, aliases: list[str] | None) -> list[str]:
    keys = {_norm(canonical), _norm(canonical.replace("_", " "))}
    for a in aliases or []:
        keys.add(_norm(a))
    return list(keys)


def _pose_of(raw: dict) -> Pose:
    p = raw.get("pose", [0.0, 0.0, 0.0])
    return (float(p[0]), float(p[1]), float(p[2]))


def _polygon_of(raw: dict) -> Polygon:
    """Parse an optional ``polygon = [[x, y], ...]`` into a tuple of (x, y) tuples.

    Missing / empty / malformed → ``()`` (entries without a surveyed shape just fall
    back to pose/radius — polygon support is additive and back-compatible).
    """
    poly = raw.get("polygon") or []
    out: list[tuple[float, float]] = []
    for v in poly:
        try:
            out.append((float(v[0]), float(v[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return tuple(out)


def _zrange_of(raw: dict) -> tuple[float, float]:
    """Parse an object's Z extent, tolerant of several editor encodings.

    Accepts ``z = [z_min, z_max]``, explicit ``z_min``/``z_max``, or a ``height``
    (with ``z_min`` defaulting to 0). Returns ``(z_min, z_max)``.
    """
    z = raw.get("z")
    if isinstance(z, (list, tuple)) and len(z) >= 2:
        return float(z[0]), float(z[1])
    z_min = float(raw.get("z_min", 0.0) or 0.0)
    if raw.get("z_max") is not None:
        return z_min, float(raw["z_max"])
    height = raw.get("height")
    if height is None and isinstance(z, (int, float)):
        height = z  # a bare scalar `z` reads as a height
    return z_min, z_min + float(height or 0.0)


def _lookup(table: dict[str, str], text: str | None) -> str | None:
    """Alias + article + fuzzy tolerant lookup: free text -> canonical name."""
    if not text:
        return None
    key = _norm(text)
    hit = table.get(key) or table.get(_strip_article(key))
    if hit is not None:
        return hit
    # Exact-alias miss: conservative fuzzy match so a mis-heard name still grounds.
    cand = _fuzzy_match(_strip_article(key), table.keys(), _fuzzy_cutoff())
    return table.get(cand) if cand else None


# --- schema entities --------------------------------------------------------

@dataclass(frozen=True)
class Room:
    name: str
    pose: Pose = (0.0, 0.0, 0.0)
    # A human-operated barrier (door/partition/screen) blocks the route here. A
    # depth check reads it as "open" (it can't see a too-narrow gap), so barrier
    # navigation asks for it to be opened on a block.
    barrier: bool = False
    # The room BOUNDARY (walls) as an XY polygon; () when not surveyed. Drives
    # point-in-polygon "which room am I in" (LocationBook.room_at) and the wall viz.
    polygon: Polygon = ()


@dataclass(frozen=True)
class Location:
    name: str
    room: str | None = None
    placement: bool = False
    category: str | None = None
    pose: Pose = (0.0, 0.0, 0.0)
    barrier: bool = False  # see Room.barrier
    # The furniture's 2D FOOTPRINT as an XY polygon; () when not surveyed.
    polygon: Polygon = ()


@dataclass(frozen=True)
class Door:
    """A map-defined door/passage the robot may need a human to open.

    Unlike a Room/Location this is **not** a navigation destination — it marks
    *where a door physically is*, so the door-opening skill engages only when the
    robot is within ``radius`` metres of one (precision over the reactive depth-only
    check). ``pose`` is map-frame ``(x, y, heading_rad)``; ``heading`` is the passage
    direction, kept for display only. ``radius=None`` → the caller's default
    (env ``WALKIE_DOOR_NEAR_RADIUS_M``).
    """
    name: str
    pose: Pose = (0.0, 0.0, 0.0)
    radius: float | None = None
    # Optional doorway REGION as an XY polygon; () when not surveyed. When set, the
    # door-opening skill can trigger by point-in-polygon (am I in the doorway?)
    # alongside / instead of `radius`.
    polygon: Polygon = ()


@dataclass(frozen=True)
class MapObject:
    """A known object instance surveyed by the world editor: an XY footprint +
    Z height (a 3D box) the perception loop later promotes to a real point cloud.

    ``footprint_polygon`` is the object's 2D outline (CCW, closed); ``z_min``/``z_max``
    its vertical extent. Together they form an axis-aligned 3D box. ``class_name`` is
    the detector class (the key's prefix before ``.`` — ``pringles.0`` → ``pringles``)
    used to match a detection to this placeholder. ``on`` is the supporting furniture.
    """

    name: str
    class_name: str
    footprint_polygon: Polygon = ()
    z_min: float = 0.0
    z_max: float = 0.0
    pose: Pose = (0.0, 0.0, 0.0)
    on: str | None = None


def build_rooms_locations(data: dict, *, include_absent: bool = False):
    """Parse the ``[rooms]``/``[locations]`` tables of a world.toml-schema dict.

    Shared by :func:`load_location_book` and GPSR's ``load_world`` so the
    present-drop + cascade-drop semantics live in one place. Returns
    ``(rooms, locations, room_alias, loc_alias)``. With ``include_absent=False``
    (default) drops ``present = false`` entries, and drops a location whose room
    was dropped — a robot must not navigate to an un-surveyed place.
    """
    rooms: dict[str, Room] = {}
    locations: dict[str, Location] = {}
    room_alias: dict[str, str] = {}
    loc_alias: dict[str, str] = {}

    for name, raw in (data.get("rooms") or {}).items():
        if not include_absent and not raw.get("present", True):
            continue
        canonical = _norm(name)
        rooms[canonical] = Room(
            name=canonical,
            pose=_pose_of(raw),
            barrier=bool(raw.get("barrier", False)),
            polygon=_polygon_of(raw),
        )
        for k in _alias_keys(canonical, raw.get("aliases")):
            room_alias[k] = canonical
        # Redundant "<name>_room" naming (e.g. "kitchen_room"): also accept the bare
        # name ("kitchen"), so "the kitchen" / "kitchen table" ground to it. setdefault
        # so an explicitly-named room always wins over the derived alias.
        if canonical.endswith("_room") and len(canonical) > len("_room"):
            room_alias.setdefault(canonical[: -len("_room")], canonical)

    for name, raw in (data.get("locations") or {}).items():
        if not include_absent and not raw.get("present", True):
            continue
        room = _norm(raw["room"]) if raw.get("room") else None
        if room is not None and room not in rooms:
            continue  # its room was dropped (present=false / unlisted) — cascade-drop
        canonical = _norm(name)
        locations[canonical] = Location(
            name=canonical,
            room=room,
            placement=bool(raw.get("placement", False)),
            category=_norm(raw["category"]) if raw.get("category") else None,
            pose=_pose_of(raw),
            barrier=bool(raw.get("barrier", False)),
            polygon=_polygon_of(raw),
        )
        for k in _alias_keys(canonical, raw.get("aliases")):
            loc_alias[k] = canonical

    return rooms, locations, room_alias, loc_alias


def build_doors(data: dict, *, include_absent: bool = False) -> dict[str, Door]:
    """Parse the ``[doors]`` table of a world.toml-schema dict into Door records.

    Each entry carries a map-frame ``pose = [x, y, heading_rad]`` and an optional
    ``radius`` (the proximity-trigger override). Drops ``present = false`` entries
    unless *include_absent*. Returns an empty dict when the file has no ``[doors]``
    table (the common case — doors are opt-in per arena).
    """
    doors: dict[str, Door] = {}
    for name, raw in (data.get("doors") or {}).items():
        if not include_absent and not raw.get("present", True):
            continue
        radius = raw.get("radius")
        canonical = _norm(name)
        doors[canonical] = Door(
            name=canonical,
            pose=_pose_of(raw),
            radius=float(radius) if radius is not None else None,
            polygon=_polygon_of(raw),
        )
    return doors


def build_map_objects(data: dict, *, include_absent: bool = False) -> list[MapObject]:
    """Parse the ``[object_instances]`` table into :class:`MapObject` records.

    Each entry is keyed ``<class>.<index>`` (e.g. ``pringles.0``) and carries an XY
    ``polygon`` (footprint) + a Z extent (``z``/``height``/``z_min``/``z_max``), an
    optional grasp ``pose`` and the ``on`` surface. The class name is the key prefix
    before the first ``.``. Drops ``present = false`` unless *include_absent*. Returns
    ``[]`` when the file has no ``[object_instances]`` table (the common case — these
    are populated by the world editor / perception cache, not hand-surveyed).
    """
    objs: list[MapObject] = []
    for name, raw in (data.get("object_instances") or {}).items():
        if not isinstance(raw, dict):
            continue
        if not include_absent and not raw.get("present", True):
            continue
        z_min, z_max = _zrange_of(raw)
        # Keep the raw instance key as the name (e.g. "pringles.0") so the seeded
        # node id is stable + readable; only the class prefix is normalized.
        class_name = _norm(str(name).split(".", 1)[0])
        objs.append(
            MapObject(
                name=str(name).strip(),
                class_name=class_name,
                footprint_polygon=_polygon_of(raw),
                z_min=z_min,
                z_max=z_max,
                pose=_pose_of(raw),
                on=_norm(raw["on"]) if raw.get("on") else None,
            )
        )
    return objs


# --- the location book ------------------------------------------------------

@dataclass
class LocationBook:
    """Named map waypoints (rooms + locations) with alias/article/fuzzy lookup.

    A name resolves location-first, then room (locations are the more specific
    waypoint). Empty book = no map file -> every lookup misses, so
    :func:`resolve_pose` falls through to the challenge's env var.
    """

    rooms: dict[str, Room] = field(default_factory=dict)
    locations: dict[str, Location] = field(default_factory=dict)
    doors: dict[str, Door] = field(default_factory=dict)
    map_objects: list[MapObject] = field(default_factory=list)
    _room_alias: dict[str, str] = field(default_factory=dict)
    _loc_alias: dict[str, str] = field(default_factory=dict)

    def _canonical(self, text: str | None) -> str | None:
        return _lookup(self._loc_alias, text) or _lookup(self._room_alias, text)

    def pose(self, name: str | None) -> Pose | None:
        canon = self._canonical(name)
        if canon is None:
            return None
        loc = self.locations.get(canon)
        if loc is not None:
            return loc.pose
        room = self.rooms.get(canon)
        return room.pose if room is not None else None

    def is_barrier(self, name: str | None) -> bool:
        """True if a human-operated door/partition blocks the route to this place."""
        canon = self._canonical(name)
        if canon is None:
            return False
        loc = self.locations.get(canon)
        if loc is not None:
            return loc.barrier
        room = self.rooms.get(canon)
        return room.barrier if room is not None else False

    def has(self, name: str | None) -> bool:
        return self._canonical(name) is not None

    def names(self) -> list[str]:
        return sorted({*self.locations, *self.rooms})

    # --- polygons (point-in-polygon "which room am I in") ---------------

    def room_at(self, x: float, y: float) -> str | None:
        """Canonical name of the room whose boundary polygon contains (x, y).

        Iterates rooms that have a surveyed ``polygon`` and returns the first match
        (rooms shouldn't overlap). Rooms without a polygon are skipped, so this is a
        no-op until the map editor traces boundaries — back-compatible.
        """
        for name, room in self.rooms.items():
            if room.polygon and point_in_polygon(x, y, room.polygon):
                return name
        return None

    # --- doors (proximity, not name lookup) -----------------------------

    def has_doors(self) -> bool:
        """True if the map defines any door — enables proximity-gated door asking."""
        return bool(self.doors)

    def nearest_door(self, x: float, y: float) -> tuple[Door, float] | None:
        """The closest mapped door to (x, y) and its planar distance (m), or None."""
        best: tuple[Door, float] | None = None
        for d in self.doors.values():
            dist = math.hypot(d.pose[0] - x, d.pose[1] - y)
            if best is None or dist < best[1]:
                best = (d, dist)
        return best

    def door_near(self, x: float, y: float, *, default_radius: float) -> Door | None:
        """A mapped door whose trigger circle contains (x, y), else None.

        Each door's own ``radius`` is used when set, otherwise *default_radius*.
        """
        for d in self.doors.values():
            r = d.radius if d.radius is not None else default_radius
            if math.hypot(d.pose[0] - x, d.pose[1] - y) <= r:
                return d
        return None

    def is_near_door(self, x: float, y: float, *, default_radius: float) -> bool:
        """True if (x, y) is inside any door's polygon REGION or its trigger circle.

        Combines the two door-trigger mechanisms: a surveyed doorway ``polygon``
        (point-in-polygon, handles wide/angled openings) OR the ``radius`` circle.
        """
        for d in self.doors.values():
            if d.polygon and point_in_polygon(x, y, d.polygon):
                return True
        return self.door_near(x, y, default_radius=default_radius) is not None


def _default_map_path() -> Path:
    """The single global arena map: ``$WALKIE_MAP_FILE``, else the repo-root ``world.toml``.

    One map serves every challenge — there is no per-task map. ``WALKIE_MAP_FILE`` is
    the one canonical override (point it at the map editor's output / the deployed nav
    arena); with it unset, the in-repo fallback is the top-level ``world.toml``
    (``walkie_world/map/locations.py`` -> ``parents[2]`` is the repo root). Both the map
    layer and the GPSR vocab (:func:`walkie_world.map.vocab.load_world`) resolve through
    here, so they can never read different files.
    """
    explicit = os.getenv("WALKIE_MAP_FILE")
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parents[2] / "world.toml"


def load_location_book(path: str | os.PathLike | None = None, *,
                       include_absent: bool = False) -> LocationBook:
    """Load a LocationBook from a world.toml-schema file.

    Resolution: explicit *path* -> ``$WALKIE_MAP_FILE`` -> the repo-root ``world.toml``
    (the single global map). **Returns an empty book (never raises) when the file is
    missing** so a box with no map still runs on the env-var fallback.
    """
    p = Path(path) if path is not None else _default_map_path()
    if not p.exists():
        return LocationBook()
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    rooms, locations, room_alias, loc_alias = build_rooms_locations(
        data, include_absent=include_absent
    )
    doors = build_doors(data, include_absent=include_absent)
    map_objects = build_map_objects(data, include_absent=include_absent)
    return LocationBook(rooms=rooms, locations=locations, doors=doors,
                        map_objects=map_objects,
                        _room_alias=room_alias, _loc_alias=loc_alias)


_CACHE: LocationBook | None = None


def get_location_book() -> LocationBook:
    """Process-wide cached book (the map is read once, reused across call sites)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_location_book()
    return _CACHE


def _reset_cache() -> None:
    """Drop the cached book — for tests that swap WALKIE_MAP_FILE between cases."""
    global _CACHE
    _CACHE = None


def resolve_pose(name: str | None, *, env_fallback: str | None = None,
                 default: str = "0.0,0.0,0.0") -> Pose:
    """Resolve a named waypoint to a map-frame pose: book -> env var -> default.

    The single replacement for the per-task ``_pose()`` helpers. Looks *name* up
    in the shared :func:`get_location_book`; on a miss reads the challenge's
    existing ``*_POSE`` env var (``env_fallback``); on a further miss parses the
    literal ``default``. Parsing reuses ``tasks.skills.geometry.parse_pose`` and
    raises on a malformed string, exactly like the old ``_pose``.

    Args:
        name: canonical location name to look up in the map (e.g. "dining_table").
        env_fallback: env var holding "x,y,heading_rad" when the map lacks *name*.
        default: literal "x,y,heading_rad" used when neither the book nor the env
            var has a pose.
    """
    if name:
        p = get_location_book().pose(name)
        if p is not None:
            return p
    if env_fallback:
        raw = os.getenv(env_fallback)
        if raw is not None and raw.strip():
            return parse_pose(raw)
    return parse_pose(default)
