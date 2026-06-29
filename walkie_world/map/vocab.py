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
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Location primitives live in the shared map layer; the vocab model is built on
# them (correct dependency direction). `_fuzzy_match` is re-exported here because
# tests/test_gpsr_hardening.py imports it (via the tasks.GPSR.world shim).
from walkie_world.map.locations import (  # noqa: F401  (re-export _fuzzy_match)
    Location,
    Pose,
    Room,
    _alias_keys,
    _fuzzy_cutoff,
    _fuzzy_match,
    _lookup,
    _norm,
    _pose_of,
    _strip_article,
    build_rooms_locations,
)


def _singulars(key: str) -> list[str]:
    """Candidate singular forms of a (already normalized) plural noun.

    Conservative English de-pluralization — callers only accept a candidate that
    is actually a known entity, so over-generating here is harmless: "cups"->
    "cup", "boxes"/"dishes"->"box"/"dish", "candies"->"candy".
    """
    cands: list[str] = []
    if key.endswith("ies") and len(key) > 3:
        cands.append(key[:-3] + "y")
    if key.endswith("es") and len(key) > 2:
        cands.append(key[:-2])
    if key.endswith("s") and len(key) > 1:
        cands.append(key[:-1])
    return cands


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
        # Shared alias/article/fuzzy lookup (tasks.skills.locations._lookup).
        return _lookup(table, text)

    def room(self, text: str | None) -> str | None:
        return self._lookup(self._room_alias, text)

    def location(self, text: str | None) -> str | None:
        return self._lookup(self._loc_alias, text)

    def obj(self, text: str | None) -> str | None:
        """Ground an object reference: an exact item, else a category's stand-in.

        "the cola" -> "cola"; "the cups" -> "cup" (counting commands are plural by
        nature); "a drink"/"drinks" -> the first object of that category (so a
        category reference still yields a concrete pickable item).
        """
        if not text:
            return None
        hit = self._lookup(self._obj_alias, text)
        if hit:
            return hit
        key = _strip_article(_norm(text))
        for cand in _singulars(key):  # tolerate a plural ("cups"->"cup", "boxes"->"box")
            if cand in self._obj_alias:
                return self._obj_alias[cand]
        cat = self.category(text)
        if cat and self.categories.get(cat):
            return self.categories[cat][0]
        return None

    def category(self, text: str | None) -> str | None:
        """Ground a category reference, tolerating singular/plural.

        Handles irregular plurals via the shared de-pluralizer: "drink"->drinks,
        "dish"->dishes (es), "cleaning supply"->cleaning_supplies (ies). The
        generator emits the singular category form ("the heaviest dish", "a
        cleaning supply"), so missing these would forfeit object-category commands.
        """
        if not text:
            return None
        key = _strip_article(_norm(text))
        if key in self.categories:
            return key
        for cat in self.categories:
            if key == cat or key in _singulars(cat):
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

    def is_barrier(self, canonical: str | None) -> bool:
        """True if a human-operated door/partition blocks the route to this place.

        Set ``barrier = true`` on the room/location in world.toml. Navigation then
        asks for it to be opened on a nav block even when the depth check reads the
        doorway "open" (it can't see a too-narrow gap) — see go_to_named.
        """
        if not canonical:
            return False
        loc = self.locations.get(canonical)
        if loc is not None:
            return loc.barrier
        room = self.rooms.get(canonical)
        return room.barrier if room is not None else False

    def vocab_prompt(self) -> str:
        """Compact listing of the arena nouns for the parser's system prompt.

        Giving the LLM the canonical vocabulary lets it normalize synonyms / STT
        slips to the exact terms ("coke"→cola, "fridge"→refrigerator) at parse
        time, so grounding hits directly — the robustness layer for §11. Nouns
        are shown with spaces; grounding is underscore/space tolerant anyway.
        """
        def _spaced(items):
            return ", ".join(sorted(i.replace("_", " ") for i in items))

        objs_by_cat = "; ".join(
            f"{cat.replace('_', ' ')}: {_spaced(items)}"
            for cat, items in sorted(self.categories.items())
        )
        return (
            "Arena vocabulary — map each reference to the CLOSEST term below and "
            "use that exact spelling (e.g. 'coke'/'soda' -> 'cola', 'fridge' -> "
            "'refrigerator', 'couch' -> 'sofa'). If something genuinely isn't "
            "listed, use the operator's words.\n"
            f"Rooms: {_spaced(self.rooms)}\n"
            f"Locations: {_spaced(self.locations)}\n"
            f"Objects by category: {objs_by_cat}\n"
            f"Names: {', '.join(sorted(self.names))}\n"
            f"Gestures/poses: {_spaced(self.gestures)}"
        )


def load_world(
    path: str | os.PathLike | None = None, *, include_absent: bool = False
) -> WorldModel:
    """Load the world model from a TOML file.

    Resolution order: explicit *path* arg, else $GPSR_WORLD_FILE, else the repo's
    ``tasks/GPSR/world.toml``. Raises FileNotFoundError if none exists (a missing
    arena is a setup error worth failing loudly on, unlike runtime perception).

    ``include_absent=True`` loads every room/location regardless of its
    ``present`` flag — the *full* CompetitionTemplate vocabulary. The default
    (False) drops ``present = false`` entries, which is what the robot wants (it
    must not navigate to an un-surveyed practice-arena place). The full vocabulary
    is for offline parser-coverage measurement against the whole grammar, where
    `present` is irrelevant — see tasks/GPSR/tools/gen_corpus.py.
    """
    if path is None:
        # walkie_world/map/vocab.py -> parents[2] is the repo root; the arena file
        # stays at tasks/GPSR/world.toml (override via $GPSR_WORLD_FILE).
        path = os.getenv("GPSR_WORLD_FILE") or (
            Path(__file__).resolve().parents[2] / "tasks" / "GPSR" / "world.toml"
        )
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    wm = WorldModel()

    # Rooms + locations (with present/cascade-drop) come from the shared map layer
    # so the schema lives in one place; GPSR layers its vocabulary on top below.
    wm.rooms, wm.locations, wm._room_alias, wm._loc_alias = build_rooms_locations(
        data, include_absent=include_absent
    )

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
