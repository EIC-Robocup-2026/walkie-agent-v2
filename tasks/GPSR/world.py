"""GPSR world model: the arena nouns and how to ground a parsed reference to them.

The model is the per-competition vocabulary (rooms, locations, objects, names,
gestures) loaded from a TOML file (`world.toml` by default; override with
`GPSR_WORLD_FILE`). It is *data, not code* — the EIC team fills in the announced
arena ~2h before the test.

This module is **pure and offline** — no robot, no LLM, no network. The parser
(`parse.py`) calls into it to ground the LLM's loose noun strings ("the kitchen
table", "a drink", "Charlie") onto canonical world entities, and the executor
(Phase 1) reads each entity's pose to navigate. Grounding is the offline-testable
core of Phase 0.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

Pose = tuple[float, float, float]


def _norm(s: str) -> str:
    """Canonical key for matching: lowercase, spaces/hyphens→underscore, trimmed.

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


@dataclass(frozen=True)
class Room:
    name: str
    pose: Pose = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Location:
    name: str
    room: str | None = None
    placement: bool = False
    category: str | None = None
    pose: Pose = (0.0, 0.0, 0.0)


@dataclass
class WorldModel:
    """Canonical arena vocabulary + alias→canonical lookup tables.

    Every public lookup returns a canonical name (the dict key) or None. Matching
    is alias-aware and article/whitespace tolerant so the LLM parser and STT can
    be loose; everything downstream speaks canonical names only.
    """

    rooms: dict[str, Room] = field(default_factory=dict)
    locations: dict[str, Location] = field(default_factory=dict)
    objects: dict[str, str] = field(default_factory=dict)        # object -> category
    categories: dict[str, list[str]] = field(default_factory=dict)  # category -> objects
    names: list[str] = field(default_factory=list)
    gestures: list[str] = field(default_factory=list)

    # alias key -> canonical, built at load (private)
    _room_alias: dict[str, str] = field(default_factory=dict)
    _loc_alias: dict[str, str] = field(default_factory=dict)
    _obj_alias: dict[str, str] = field(default_factory=dict)
    _name_alias: dict[str, str] = field(default_factory=dict)
    _gesture_alias: dict[str, str] = field(default_factory=dict)

    # --- grounding lookups (alias + article tolerant) -------------------

    def _lookup(self, table: dict[str, str], text: str | None) -> str | None:
        if not text:
            return None
        key = _norm(text)
        return table.get(key) or table.get(_strip_article(key))

    def room(self, text: str | None) -> str | None:
        return self._lookup(self._room_alias, text)

    def location(self, text: str | None) -> str | None:
        return self._lookup(self._loc_alias, text)

    def obj(self, text: str | None) -> str | None:
        """Ground an object reference: an exact item, else a category's stand-in.

        "the cola" -> "cola"; "a drink"/"drinks" -> the first object of that
        category (so a category reference still yields a concrete pickable item).
        """
        if not text:
            return None
        hit = self._lookup(self._obj_alias, text)
        if hit:
            return hit
        cat = self.category(text)
        if cat and self.categories.get(cat):
            return self.categories[cat][0]
        return None

    def category(self, text: str | None) -> str | None:
        """Ground a category reference, tolerating singular/plural ("drink")."""
        if not text:
            return None
        key = _strip_article(_norm(text))
        if key in self.categories:
            return key
        for cat in self.categories:
            if key == cat or key + "s" == cat or key == cat.rstrip("s"):
                return cat
        return None

    def name(self, text: str | None) -> str | None:
        return self._lookup(self._name_alias, text)

    def gesture(self, text: str | None) -> str | None:
        return self._lookup(self._gesture_alias, text)

    def location_pose(self, canonical: str) -> Pose | None:
        loc = self.locations.get(canonical)
        if loc is not None:
            return loc.pose
        room = self.rooms.get(canonical)
        return room.pose if room is not None else None


def _alias_keys(canonical: str, aliases: list[str] | None) -> list[str]:
    keys = {_norm(canonical), _norm(canonical.replace("_", " "))}
    for a in aliases or []:
        keys.add(_norm(a))
    return list(keys)


def _pose_of(raw: dict) -> Pose:
    p = raw.get("pose", [0.0, 0.0, 0.0])
    return (float(p[0]), float(p[1]), float(p[2]))


def load_world(path: str | os.PathLike | None = None) -> WorldModel:
    """Load the world model from a TOML file.

    Resolution order: explicit *path* arg, else $GPSR_WORLD_FILE, else the
    sibling `world.toml`. Raises FileNotFoundError if none exists (a missing
    arena is a setup error worth failing loudly on, unlike runtime perception).
    """
    if path is None:
        path = os.getenv("GPSR_WORLD_FILE") or Path(__file__).with_name("world.toml")
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    wm = WorldModel()

    for name, raw in (data.get("rooms") or {}).items():
        canonical = _norm(name)
        wm.rooms[canonical] = Room(name=canonical, pose=_pose_of(raw))
        for k in _alias_keys(canonical, raw.get("aliases")):
            wm._room_alias[k] = canonical

    for name, raw in (data.get("locations") or {}).items():
        canonical = _norm(name)
        wm.locations[canonical] = Location(
            name=canonical,
            room=_norm(raw["room"]) if raw.get("room") else None,
            placement=bool(raw.get("placement", False)),
            category=_norm(raw["category"]) if raw.get("category") else None,
            pose=_pose_of(raw),
        )
        for k in _alias_keys(canonical, raw.get("aliases")):
            wm._loc_alias[k] = canonical

    for category, items in (data.get("object_categories") or {}).items():
        cat = _norm(category)
        wm.categories[cat] = [_norm(o) for o in items]
        for o in items:
            obj = _norm(o)
            wm.objects[obj] = cat
            for k in _alias_keys(obj, [o.replace("_", " ")]):
                wm._obj_alias[k] = obj

    for n in data.get("names") or []:
        wm.names.append(n)
        wm._name_alias[_norm(n)] = n

    for g, raw in (data.get("gestures") or {}).items():
        canonical = _norm(g)
        wm.gestures.append(canonical)
        for k in _alias_keys(canonical, (raw or {}).get("aliases")):
            wm._gesture_alias[k] = canonical

    return wm
