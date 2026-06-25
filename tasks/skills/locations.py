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
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from tasks.skills.geometry import parse_pose

Pose = tuple[float, float, float]


# --- pure text matching (shared by LocationBook and GPSR's WorldModel) ------

def _fuzzy_cutoff() -> float:
    """difflib ratio an STT/LLM near-miss must clear to match a known name
    (GPSR_GROUNDING_FUZZY_CUTOFF, default 0.8). 0 / unparsable disables fuzzy
    matching (exact-alias behaviour only)."""
    try:
        return float(os.getenv("GPSR_GROUNDING_FUZZY_CUTOFF", "0.8") or 0)
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


@dataclass(frozen=True)
class Location:
    name: str
    room: str | None = None
    placement: bool = False
    category: str | None = None
    pose: Pose = (0.0, 0.0, 0.0)
    barrier: bool = False  # see Room.barrier


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
            name=canonical, pose=_pose_of(raw), barrier=bool(raw.get("barrier", False))
        )
        for k in _alias_keys(canonical, raw.get("aliases")):
            room_alias[k] = canonical

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
        )
        for k in _alias_keys(canonical, raw.get("aliases")):
            loc_alias[k] = canonical

    return rooms, locations, room_alias, loc_alias


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


def _default_map_path() -> Path:
    """$WALKIE_MAP_FILE -> $GPSR_WORLD_FILE -> the sibling GPSR world.toml.

    Defaulting to GPSR's file means one arena file serves every challenge and the
    existing teach_poses tool already feeds them all; point WALKIE_MAP_FILE
    elsewhere to swap in the map editor's output.
    """
    explicit = os.getenv("WALKIE_MAP_FILE") or os.getenv("GPSR_WORLD_FILE")
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parents[1] / "GPSR" / "world.toml"


def load_location_book(path: str | os.PathLike | None = None, *,
                       include_absent: bool = False) -> LocationBook:
    """Load a LocationBook from a world.toml-schema file.

    Resolution: explicit *path* -> $WALKIE_MAP_FILE -> $GPSR_WORLD_FILE -> the
    sibling GPSR ``world.toml``. **Returns an empty book (never raises) when the
    file is missing** so a box with no map still runs on env-var fallbacks.
    """
    p = Path(path) if path is not None else _default_map_path()
    if not p.exists():
        return LocationBook()
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    rooms, locations, room_alias, loc_alias = build_rooms_locations(
        data, include_absent=include_absent
    )
    return LocationBook(rooms=rooms, locations=locations,
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
