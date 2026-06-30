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

import os
import threading
import time
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
