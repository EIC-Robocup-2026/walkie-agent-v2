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

import difflib
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

Pose = tuple[float, float, float]


def _fuzzy_cutoff() -> float:
    """difflib ratio an STT/LLM near-miss must clear to be accepted as a match
    (GPSR_GROUNDING_FUZZY_CUTOFF, default 0.8). 0 / unparsable disables fuzzy
    grounding entirely (exact-alias behaviour only)."""
    try:
        return float(os.getenv("GPSR_GROUNDING_FUZZY_CUTOFF", "0.8") or 0)
    except ValueError:
        return 0.0


def _fuzzy_match(key: str, candidates, cutoff: float) -> str | None:
    """Closest candidate to *key* at/above *cutoff* (difflib ratio), or None.

    Belt-and-suspenders for STT/LLM near-misses ("kitchen tabel" -> kitchen_table,
    "couch" -> couches) that survive the parser's vocab-grounded extraction and so
    would otherwise be an ungrounded gap. Only consulted on an exact-lookup miss,
    so it can never override a correct exact match. ``cutoff <= 0`` disables it.
    Pure (no robot/LLM/network) -> unit-tested directly."""
    if not key or cutoff <= 0:
        return None
    hits = difflib.get_close_matches(key, list(candidates), n=1, cutoff=cutoff)
    return hits[0] if hits else None


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
        hit = table.get(key) or table.get(_strip_article(key))
        if hit is not None:
            return hit
        # Exact-alias miss: fall back to a conservative fuzzy match so a mis-heard
        # noun ("kitchen tabel") still grounds instead of forfeiting the command.
        cand = _fuzzy_match(_strip_article(key), table.keys(), _fuzzy_cutoff())
        return table.get(cand) if cand else None

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


def _alias_keys(canonical: str, aliases: list[str] | None) -> list[str]:
    keys = {_norm(canonical), _norm(canonical.replace("_", " "))}
    for a in aliases or []:
        keys.add(_norm(a))
    return list(keys)


def _pose_of(raw: dict) -> Pose:
    p = raw.get("pose", [0.0, 0.0, 0.0])
    return (float(p[0]), float(p[1]), float(p[2]))


def load_world(
    path: str | os.PathLike | None = None, *, include_absent: bool = False
) -> WorldModel:
    """Load the world model from a TOML file.

    Resolution order: explicit *path* arg, else $GPSR_WORLD_FILE, else the
    sibling `world.toml`. Raises FileNotFoundError if none exists (a missing
    arena is a setup error worth failing loudly on, unlike runtime perception).

    ``include_absent=True`` loads every room/location regardless of its
    ``present`` flag — the *full* CompetitionTemplate vocabulary. The default
    (False) drops ``present = false`` entries, which is what the robot wants (it
    must not navigate to an un-surveyed practice-arena place). The full vocabulary
    is for offline parser-coverage measurement against the whole grammar, where
    `present` is irrelevant — see tasks/GPSR/tools/gen_corpus.py.
    """
    if path is None:
        path = os.getenv("GPSR_WORLD_FILE") or Path(__file__).with_name("world.toml")
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    wm = WorldModel()

    for name, raw in (data.get("rooms") or {}).items():
        if not include_absent and not raw.get("present", True):
            continue  # in the template but ABSENT from this arena — drop it entirely
        canonical = _norm(name)
        wm.rooms[canonical] = Room(name=canonical, pose=_pose_of(raw))
        for k in _alias_keys(canonical, raw.get("aliases")):
            wm._room_alias[k] = canonical

    for name, raw in (data.get("locations") or {}).items():
        if not include_absent and not raw.get("present", True):
            continue  # absent from this arena — drop so nothing grounds/navigates to it
        room = _norm(raw["room"]) if raw.get("room") else None
        if room is not None and room not in wm.rooms:
            continue  # its room was dropped (present=false / unlisted) — cascade-drop it too
        canonical = _norm(name)
        wm.locations[canonical] = Location(
            name=canonical,
            room=room,
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
