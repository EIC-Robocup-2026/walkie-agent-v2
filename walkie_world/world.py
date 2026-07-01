"""WalkieWorld — the unified world model reached as ``ctx.world``.

One facade over everything the robot knows about its world:

* the **static map** — rooms / locations / doors (waypoints + shapes) and the
  object/name/gesture **vocabulary** the parser grounds against
  (:mod:`walkie_world.map`);
* the **dynamic object scene graph** — the numpy :class:`SceneStore` of fused 3D
  objects + relations, plus the world-editor's surveyed objects seeded as
  bounding-box placeholders that perception later promotes to point clouds
  (:mod:`walkie_world.scene`);
* **people** — face + appearance memory for re-ID, built lazily so ``chromadb``
  loads only when a people method is first used (:mod:`walkie_world.people`).

Construction is cheap and import-light (numpy only); pass ``embed_text`` (bound to
``walkieAI.image.embed_text``) to enable CLIP text search for objects and semantic
attire re-ID for people. The perception producer (``services/realtime_explore``)
holds a reference to the SAME instance and feeds observations via
:meth:`observe_objects`; tasks and agents query it. There must be exactly ONE
WalkieWorld per process so the scene lock and the people DB stay single-owner.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from walkie_world.config import relation_kwargs, scene_store_kwargs
from walkie_world.map import locations as _loc
from walkie_world.map.locations import (
    Door,
    LocationBook,
    MapObject,
    Pose,
    Room,
    get_location_book,
    load_location_book,
)
from walkie_world.map.vocab import WorldModel, load_world
from walkie_world.scene.relations import derive_relations
from walkie_world.scene.store import ObjectNode, Relation, SceneStore


def _pose_surveyed(pose: Optional[Pose]) -> bool:
    """A pose is 'surveyed' unless it's missing or the all-zero [0,0,0] placeholder.

    world.toml uses ``[0,0,0]`` for "not yet surveyed"; navigating there would drive
    to the map origin, so such a location is treated as having no usable pose.
    """
    if pose is None:
        return False
    return not (abs(pose[0]) < 1e-6 and abs(pose[1]) < 1e-6 and abs(pose[2]) < 1e-6)


_LEAD_VERBS = (
    "go to ", "goto ", "drive to ", "navigate to ", "move to ", "come to ",
    "head to ", "walk to ", "get to ", "go ", "go to the ",
)


def _strip_lead_verbs(text: str) -> str:
    """Drop a leading navigation verb ("go to the cabinet" -> "the cabinet").

    The LLM usually passes a bare place name, but a delegated phrase may include the
    verb; stripping it keeps the resolver robust. Longest prefixes first.
    """
    low = text.lower()
    for v in sorted(_LEAD_VERBS, key=len, reverse=True):
        if low.startswith(v):
            return text[len(v):].strip()
    return text


def _split_room_qualifier(text: str) -> tuple[str, Optional[str]]:
    """Split a "<target> in [the] <room>" phrase into (target, room); else (text, None).

    Lets the resolver handle "the table in the kitchen" without an LLM round-trip
    (the agent can also pass ``room=`` explicitly).
    """
    low = text.lower()
    for sep in (" in the ", " in "):
        i = low.find(sep)
        if i != -1:
            return text[:i].strip(), text[i + len(sep):].strip()
    return text.strip(), None


@dataclass
class PlaceMatch:
    """A resolved place reference (see :meth:`WalkieWorld.resolve_place`).

    Exactly one navigation anchor is set: ``pose`` (a surveyed map location/room —
    drive there with heading) or ``point`` (an observed scene object's XY — approach
    it). ``candidates`` holds the other places that also matched when several did and
    the nearest was auto-picked (so the caller can announce "the nearest table").
    """

    kind: str                                  # "location" | "room" | "object"
    label: str                                 # human-readable, e.g. "kitchen table"
    name: Optional[str] = None                 # canonical location/room name, or scene node id
    pose: Optional[Pose] = None                # full nav pose (x, y, heading) for a map place
    point: Optional[tuple] = None              # (x, y) only, for an observed scene object
    room: Optional[str] = None                 # resolved room, if any
    candidates: list = field(default_factory=list)  # other matched labels (auto-picked nearest)
    source: str = "map"                        # "map" | "scene"


class WalkieWorld:
    """Unified query engine + context for rooms, objects and people (``ctx.world``)."""

    def __init__(
        self,
        *,
        map_path: str | os.PathLike | None = None,
        scene_dir: str | None = None,
        people_persist_dir: str | None = None,
        people_frames_dir: str | None = None,
        embed_text: Optional[Callable[[str], list[float]]] = None,
        enable_people: bool = True,
        include_absent: bool = False,
        seed_locations: bool | None = None,
    ) -> None:
        self._embed_text = embed_text
        self._include_absent = include_absent
        # Seed the static map's [locations] (furniture) as queryable scene nodes too,
        # not just [object_instances]. Default OFF (env override) because it changes
        # what every scene consumer sees (all_objects / recently_seen / the Database
        # agent's dumps), so it's opt-in. See _location_map_objects.
        self._seed_locations = (
            seed_locations
            if seed_locations is not None
            else os.getenv("WALKIE_EXPLORE_SEED_LOCATIONS", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )

        # --- map + vocab (lazy; pure, light) ---
        self._map_path = map_path
        self._book: Optional[LocationBook] = None
        self._vocab: Optional[WorldModel] = None

        # --- scene store (eager; numpy, cheap) ---
        kwargs = scene_store_kwargs()
        if scene_dir is not None:
            kwargs["store_dir"] = scene_dir
        self._scene = SceneStore(embed_text=embed_text, **kwargs)
        self._scene_lock = threading.RLock()  # serializes observe/seed installs
        self._relation_kwargs = relation_kwargs()

        # --- people (lazy; chromadb loads only on first use) ---
        self._enable_people = enable_people
        self._people = None
        self._people_persist_dir = people_persist_dir
        self._people_frames_dir = people_frames_dir

        # Seed world-editor object shapes as bounding-box placeholder nodes.
        self._seed_map_objects()

    # ==================================================================
    # Map / rooms / vocab
    # ==================================================================
    @property
    def map(self) -> LocationBook:
        """The :class:`LocationBook` (rooms/locations/doors/objects + waypoints)."""
        if self._book is None:
            if self._map_path is None and not self._include_absent:
                # Reuse the process-wide cached book (one parse, shared _CACHE).
                self._book = get_location_book()
            else:
                self._book = load_location_book(
                    self._map_path, include_absent=self._include_absent
                )
        return self._book

    @property
    def vocab(self) -> WorldModel:
        """The grounding :class:`WorldModel` (objects/categories/names/gestures)."""
        if self._vocab is None:
            self._vocab = load_world(self._map_path, include_absent=self._include_absent)
        return self._vocab

    # vocab grounding (the WorldModel surface GPSR's parser + skills depend on)
    def room(self, text: str | None) -> str | None:
        return self.vocab.room(text)

    def location(self, text: str | None) -> str | None:
        return self.vocab.location(text)

    def obj(self, text: str | None) -> str | None:
        return self.vocab.obj(text)

    def category(self, text: str | None) -> str | None:
        return self.vocab.category(text)

    def name(self, text: str | None) -> str | None:
        return self.vocab.name(text)

    def gesture(self, text: str | None) -> str | None:
        return self.vocab.gesture(text)

    def location_pose(self, canonical: str) -> Pose | None:
        return self.vocab.location_pose(canonical)

    def is_barrier(self, canonical: str | None) -> bool:
        return self.vocab.is_barrier(canonical)

    def vocab_prompt(self) -> str:
        return self.vocab.vocab_prompt()

    @property
    def categories(self) -> dict[str, list[str]]:
        return self.vocab.categories

    @property
    def objects(self) -> dict[str, str]:
        return self.vocab.objects

    @property
    def names(self) -> list[str]:
        return self.vocab.names

    @property
    def gestures(self) -> list[str]:
        return self.vocab.gestures

    # map waypoints + geometry (the LocationBook surface)
    @property
    def rooms(self) -> dict[str, Room]:
        return self.map.rooms

    @property
    def locations(self) -> dict[str, "object"]:
        return self.map.locations

    @property
    def doors(self) -> dict[str, Door]:
        return self.map.doors

    def pose(self, name: str | None) -> Pose | None:
        return self.map.pose(name)

    def has(self, name: str | None) -> bool:
        return self.map.has(name)

    def resolve_pose(
        self, name: str | None, *, env_fallback: str | None = None, default: str = "0.0,0.0,0.0"
    ) -> Pose:
        return _loc.resolve_pose(name, env_fallback=env_fallback, default=default)

    # polygons / doors
    def room_at(self, x: float, y: float) -> str | None:
        """Canonical room whose boundary polygon contains (x, y), else None."""
        return self.map.room_at(x, y)

    def has_doors(self) -> bool:
        return self.map.has_doors()

    def is_near_door(self, x: float, y: float, *, radius: float | None = None) -> bool:
        """True if (x, y) is in any door's polygon region or trigger circle.

        ``radius`` overrides the per-door / default radius; defaults to
        ``WALKIE_DOOR_NEAR_RADIUS_M`` (1.5 m).
        """
        if radius is None:
            radius = float(os.getenv("WALKIE_DOOR_NEAR_RADIUS_M", "1.5"))
        return self.map.is_near_door(x, y, default_radius=radius)

    def nearest_door(self, x: float, y: float):
        return self.map.nearest_door(x, y)

    def map_objects(self) -> list[MapObject]:
        """The world-editor's surveyed object shapes (XY footprint + Z height)."""
        return list(self.map.map_objects)

    def default_location_for(self, name: str | None) -> tuple[str, Pose | None] | None:
        """The canonical placement (name, pose) where an object/category belongs.

        Resolves object -> category -> the location whose ``category`` matches it
        (the map's "the drinks live in the cabinet" encoding). Accepts either an
        object name ("cola") or a category ("drinks"). Returns ``(location_name,
        pose)`` — ``pose`` may be ``None`` for an unsurveyed location — or ``None``
        when nothing grounds. This is the lookup the Finals "return a misplaced
        object to its default location" problem (and the Database agent's
        ``get_default_location`` tool) is built on.
        """
        obj = self.obj(name)
        cat = self.objects.get(obj) if obj else self.category(name)
        if cat is None:
            return None
        for loc_name, loc in self.locations.items():
            if getattr(loc, "category", None) == cat:
                return loc_name, self.location_pose(loc_name)
        return None

    def objects_in_room(self, room: str | None) -> list[ObjectNode]:
        """Confirmed scene objects whose centroid falls inside *room*'s polygon.

        Grounds *room* to a canonical name, then filters ``all_objects()`` by
        ``room_at`` of each centroid. Empty when the room is unknown, has no
        boundary polygon, or nothing has been catalogued there.
        """
        canon = self.room(room)
        if canon is None:
            return []
        out: list[ObjectNode] = []
        for n in self.all_objects():
            cx, cy = float(n.centroid[0]), float(n.centroid[1])
            if self.room_at(cx, cy) == canon:
                out.append(n)
        return out

    # --- natural-language place resolution ----------------------------------
    def _match_locations(self, target: str, canon_room: Optional[str]) -> list[str]:
        """Canonical locations matching *target*, optionally scoped to *canon_room*.

        Matches by name token ("table" ∈ kitchen_table), the vocab's 1-best alias/fuzzy
        grounding ("couch" → sofa), the category a location holds ("drinks" → cabinet),
        or a fuzzy hit on a name token. Room-scoping filters by ``Location.room``.
        """
        t = _loc._strip_article(_loc._norm(target)) if target else ""  # norm first, then strip
        if not t:
            return []
        global_hit = self.location(target)          # 1-best alias/fuzzy (may be another room)
        target_cat = self.category(target)          # e.g. "drinks"
        cutoff = _loc._fuzzy_cutoff()
        out: list[str] = []
        for name, loc in self.locations.items():
            if canon_room is not None and getattr(loc, "room", None) != canon_room:
                continue
            tokens = name.split("_")
            if (
                t in tokens
                or global_hit == name
                or (target_cat is not None and getattr(loc, "category", None) == target_cat)
                or bool(_loc._fuzzy_match(t, tokens, cutoff))
            ):
                out.append(name)
        return out

    @staticmethod
    def _nearest(surveyed: list[tuple[str, Pose]], near) -> tuple[str, Pose]:
        """The (name, pose) whose XY is closest to *near*; first entry if near is None."""
        if near is None:
            return surveyed[0]
        return min(surveyed, key=lambda it: math.hypot(it[1][0] - near[0], it[1][1] - near[1]))

    def _room_center_radius(self, canon_room: Optional[str]):
        """(center_xy, radius_m) for a room, for spatially scoping a scene query, or (None, None)."""
        room = self.rooms.get(canon_room) if canon_room else None
        if room is None:
            return None, None
        poly = getattr(room, "polygon", ()) or ()
        if poly:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            r = max(0.5, 0.5 * math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
            return (cx, cy), r
        if _pose_surveyed(room.pose):
            return (room.pose[0], room.pose[1]), float(os.getenv("WALKIE_ROOM_QUERY_RADIUS_M", "3.0"))
        return None, None

    def resolve_place(self, text: str | None, *, room: str | None = None, near=None) -> "PlaceMatch | None":
        """Resolve a natural-language place ("the table in the kitchen") to a nav target.

        Composes the whole world model: direct vocab grounding, then **room-scoped**
        location matching (disambiguating "the table" by room), then a **scene-graph**
        fallback to an observed object's position. Returns a :class:`PlaceMatch` whose
        ``pose`` (map place) or ``point`` (observed object) is the navigation anchor, or
        ``None`` when nothing resolves.

        - ``room`` scopes the search to a room (the agent can pass it explicitly; a
          "... in the <room>" phrase is also parsed out of *text*).
        - ``near`` is the robot's ``(x, y)``: when several map places match, the nearest
          is auto-picked and the rest are returned in ``candidates``.
        - The scene fallback (``WALKIE_RESOLVE_SCENE_FALLBACK=1``, default on) uses
          ``query_text`` scoped to the room's center/radius.
        """
        if not text or not str(text).strip():
            canon_room = self.room(room) if room else None
            if canon_room and _pose_surveyed(self.location_pose(canon_room)):
                return PlaceMatch(kind="room", label=canon_room.replace("_", " "),
                                  name=canon_room, pose=self.location_pose(canon_room),
                                  room=canon_room)
            return None

        text = _strip_lead_verbs(str(text).strip())
        target, room_from_text = _split_room_qualifier(text)
        room_hint = room or room_from_text
        canon_room = self.room(room_hint) if room_hint else None

        # (1) direct location grounding (full phrase, then the target), room-consistent.
        for probe in (text, target):
            canon = self.location(probe)
            if canon:
                loc = self.locations.get(canon)
                loc_room = getattr(loc, "room", None) if loc else None
                if canon_room is None or loc_room == canon_room:
                    pose = self.location_pose(canon)
                    if _pose_surveyed(pose):
                        return PlaceMatch(kind="location", label=canon.replace("_", " "),
                                          name=canon, pose=pose, room=loc_room or canon_room)

        # (1b) the whole reference IS a room ("the kitchen", "living room").
        canon_txt_room = self.room(text)
        if canon_txt_room and _pose_surveyed(self.location_pose(canon_txt_room)):
            return PlaceMatch(kind="room", label=canon_txt_room.replace("_", " "),
                              name=canon_txt_room, pose=self.location_pose(canon_txt_room),
                              room=canon_txt_room)

        # (2) room-scoped / ambiguous location collect -> single, else nearest to `near`.
        cands = self._match_locations(target, canon_room)
        surveyed = [(n, self.location_pose(n)) for n in cands]
        surveyed = [(n, p) for n, p in surveyed if _pose_surveyed(p)]
        if surveyed:
            name, pose = self._nearest(surveyed, near) if len(surveyed) > 1 else surveyed[0]
            others = [n.replace("_", " ") for n, _ in surveyed if n != name]
            return PlaceMatch(kind="location", label=name.replace("_", " "), name=name,
                              pose=pose, room=getattr(self.locations.get(name), "room", None),
                              candidates=others)

        # (3) scene-graph fallback: an observed object (optionally scoped to the room).
        if os.getenv("WALKIE_RESOLVE_SCENE_FALLBACK", "1").lower() in ("1", "true", "yes"):
            center, radius = self._room_center_radius(canon_room)
            try:
                hits = self.query_text(target, k=1, near=center, radius=radius)
            except Exception:  # noqa: BLE001 — degrade to "not found"
                hits = []
            if hits:
                n = hits[0]
                return PlaceMatch(kind="object", label=(n.best_caption or n.class_name),
                                  name=n.id, point=(float(n.centroid[0]), float(n.centroid[1])),
                                  room=canon_room, source="scene")
        return None

    def query_text_in_room(self, query: str, room: str | None, k: int = 5) -> list[ObjectNode]:
        """CLIP scene-graph search for *query* limited to *room* (its center + radius).

        Grounds *room*, then spatially filters ``query_text`` to the room's geometry
        (polygon centroid + half-diagonal, or the room pose + ``WALKIE_ROOM_QUERY_RADIUS_M``
        when there is no polygon). Empty when the room is unknown; unscoped when the room
        has no usable geometry. This answers "where is the <object> in the <room>".
        """
        canon = self.room(room)
        if canon is None:
            return []
        center, radius = self._room_center_radius(canon)
        if center is None:
            return self.query_text(query, k=k)
        return self.query_text(query, k=k, near=center, radius=radius)

    # ==================================================================
    # Objects / scene graph
    # ==================================================================
    def query_text(self, query: str, k: int = 5, *, near=None, radius: float | None = None) -> list[ObjectNode]:
        return self._scene.query_text(query, k, near=near, radius=radius)

    def query_near(self, center, radius: float) -> list[ObjectNode]:
        return self._scene.query_near(center, radius)

    def recently_seen(self, limit: int = 5) -> list[ObjectNode]:
        return self._scene.recently_seen(limit)

    def all_objects(self) -> list[ObjectNode]:
        return self._scene.all_objects()

    def get(self, node_id: str) -> ObjectNode | None:
        return self._scene.get(node_id)

    def relations_of(self, node_id: str) -> list[Relation]:
        return self._scene.relations_of(node_id)

    def to_text_description(self) -> str:
        return self._scene.to_text_description()

    def count(self) -> int:
        return self._scene.count()

    @property
    def scene(self) -> SceneStore:
        """The underlying scene store (for advanced/producer use)."""
        return self._scene

    # ingest (producer -> model)
    def observe_objects(self, observations) -> None:
        """Fold a batch of fused object observations into the scene graph.

        Merge-into-persisted (never shrinks; promotes map placeholders) -> derive
        relations -> atomic install, all under the world's scene lock so the
        producer's build thread and query callers never tear state. This is the
        single ingest entry point the perception producer calls each build.
        """
        with self._scene_lock:
            nodes = self._scene.merge(observations, now=time.time())
            rels = derive_relations(nodes, **self._relation_kwargs)
            self._scene.install(nodes, rels)

    def _seed_map_objects(self) -> None:
        """Seed surveyed world.toml shapes as ``source="map"`` placeholder nodes.

        Always seeds the world-editor object instances (``[object_instances]``). When
        ``seed_locations`` is on (env ``WALKIE_EXPLORE_SEED_LOCATIONS``), the surveyed
        furniture (``[locations]``: table/sofa/sink/…) is seeded too, so ``query_text``
        / ``query_near`` can return world.toml entries — at the cost of those nodes also
        showing up in ``all_objects`` / ``recently_seen`` / the Database agent's dumps.
        """
        mos = list(self.map.map_objects)
        if self._seed_locations:
            mos += self._location_map_objects()
        if not mos:
            return
        with self._scene_lock:
            nodes = self._scene.merge_map_objects(mos, now=time.time())
            rels = derive_relations(nodes, **self._relation_kwargs)
            self._scene.install(nodes, rels)

    def _location_map_objects(self) -> list[MapObject]:
        """Surveyed ``[locations]`` as :class:`MapObject` placeholders (query_text seed).

        A :class:`Location` carries no Z extent (only an XY footprint + a nav pose), so
        the node is a flat footprint at ``z=0`` — enough for text + XY-proximity recall,
        which is all ``query_text``/``query_near`` need. The node id is ``map:<location>``
        (distinct from the ``map:<class>.<i>`` object-instance ids), and ``class_name`` is
        the location name so the keyword matcher substring-hits it (``"table"`` ⊂
        ``"table_2"``). Honours the ``present``/``include_absent`` drop already applied by
        the LocationBook, so a ``present = false`` location is never seeded.
        """
        out: list[MapObject] = []
        for name, loc in self.map.locations.items():
            out.append(
                MapObject(
                    name=name,
                    class_name=name,
                    footprint_polygon=tuple(loc.polygon) if loc.polygon else (),
                    z_min=0.0,
                    z_max=0.0,
                    pose=loc.pose,
                    on=loc.room,
                )
            )
        return out

    # ==================================================================
    # People
    # ==================================================================
    @property
    def people(self):
        """The :class:`PeopleStore` (lazy; loads chromadb on first access)."""
        if not self._enable_people:
            raise RuntimeError(
                "people memory is disabled for this WalkieWorld (enable_people=False)"
            )
        if self._people is None:
            from walkie_world.people.store import PeopleStore  # lazy: pulls chromadb

            if self._people_persist_dir is not None or self._people_frames_dir is not None:
                self._people = PeopleStore(
                    persist_dir=self._people_persist_dir,
                    frames_dir=self._people_frames_dir,
                )
            else:
                self._people = PeopleStore.from_env()
        return self._people

    def enroll_person(self, name: str, drink: str, face_embedding, **kwargs):
        """Enroll/refresh a person. See :meth:`PeopleStore.enroll` for kwargs."""
        return self.people.enroll(name, drink, face_embedding, **kwargs)

    def recognize_person(self, face_embedding, *, max_distance: float | None = None):
        md = self.people.face_match_max_distance() if max_distance is None else max_distance
        return self.people.recognize(face_embedding, max_distance=md)

    def recognize_person_fused(self, face_embedding=None, app_embedding=None, **kwargs):
        return self.people.recognize_fused(face_embedding, app_embedding, **kwargs)

    def fused_person_scores(self, face_embedding=None, app_embedding=None, **kwargs):
        return self.people.fused_scores(face_embedding, app_embedding, **kwargs)

    def find_person_by_caption_embedding(self, embedding, *, max_distance: float | None = None):
        """Vector-first attire re-ID: nearest stored caption embedding, or None.

        The caption analogue of :meth:`recognize_person` — pass a precomputed CLIP-text
        vector (no ``embed_text`` needed, no lexical fallback). ``max_distance`` defaults
        to ``WORLD_PEOPLE_CAPTION_MATCH_THRESHOLD``.
        """
        return self.people.find_by_caption_embedding(embedding, max_distance=max_distance)

    def find_person_by_caption(self, query: str):
        """Semantic attire re-ID from TEXT: embed via ``embed_text`` → vector match
        (:meth:`find_person_by_caption_embedding`), then lexical fallback. Use the
        ``_embedding`` variant directly when you already hold the caption vector."""
        people = self.people
        if self._embed_text is not None:
            try:
                vec = self._embed_text(query)
            except Exception:
                vec = None
            if vec:
                hit = self.find_person_by_caption_embedding(vec)
                if hit is not None:
                    return hit
        return people.find_by_caption_lexical(query)

    def get_person(self, name_or_id: str):
        return self.people.get(name_or_id)

    def list_people(self) -> list:
        return self.people.list_people()

    def people_count(self) -> int:
        return self.people.count() if self._enable_people else 0

    def clear_people(self) -> None:
        if self._enable_people:
            self.people.clear()

    def fused_min_score(self) -> float:
        return self.people.fused_min_score()

    def face_match_max_distance(self) -> float:
        return self.people.face_match_max_distance()

    # ==================================================================
    # Lifecycle
    # ==================================================================
    def reset_map_cache(self) -> None:
        """Drop the cached map (for tests swapping WALKIE_MAP_FILE)."""
        self._book = None
        self._vocab = None
        _loc._reset_cache()

    def persist(self) -> None:
        """Flush the scene store to disk (producer calls this on shutdown).

        ``install`` already persists after every ingest, so this is a belt-and-braces
        final write; a no-op when the store has no on-disk directory.
        """
        self._scene._persist()
