"""walkie_world — Walkie's unified world model + query engine (``ctx.world``).

One package that owns ALL world knowledge: the static arena map (rooms / locations
/ doors / object shapes + the grounding vocabulary), the dynamic 3D object scene
graph (numpy SceneStore + relations), and people memory (face + appearance re-ID).
The perception producer (:mod:`services.realtime_explore`) feeds it observations;
tasks and agents query it through :class:`~walkie_world.world.WalkieWorld`.

Public surface::

    from walkie_world import WalkieWorld
    world = WalkieWorld(embed_text=lambda q: walkieAI.image.embed_text(q))
    world.is_near_door(x, y)          # door proximity (polygon region or radius)
    world.room_at(x, y)               # which room am I in (point-in-polygon)
    world.query_text("red mug")       # CLIP object search over the scene graph
    world.find_person_by_caption(...)  # semantic attire re-ID

Intentionally **import-light** (PEP 562 lazy attributes): ``import walkie_world``
must not eagerly pull chromadb (people), Open3D, the camera, or the SDK. ``chromadb``
loads only when a people method is first used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "WalkieWorld",
    "PlaceMatch",
    "ObjectNode",
    "Relation",
    "ObjectObservation",
    "PersonRecord",
    "Room",
    "Location",
    "Door",
    "MapObject",
    "Pose",
]

_LAZY = {
    "WalkieWorld": ("walkie_world.world", "WalkieWorld"),
    "PlaceMatch": ("walkie_world.world", "PlaceMatch"),
    "ObjectNode": ("walkie_world.scene.store", "ObjectNode"),
    "Relation": ("walkie_world.scene.store", "Relation"),
    "ObjectObservation": ("walkie_world.scene.ingest", "ObjectObservation"),
    "PersonRecord": ("walkie_world.people.store", "PersonRecord"),
    "Room": ("walkie_world.map.locations", "Room"),
    "Location": ("walkie_world.map.locations", "Location"),
    "Door": ("walkie_world.map.locations", "Door"),
    "MapObject": ("walkie_world.map.locations", "MapObject"),
    "Pose": ("walkie_world.map.locations", "Pose"),
}


def __getattr__(name: str):  # PEP 562 — resolve public names lazily
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # static type hints only — never imported at runtime
    from walkie_world.map.locations import Door, Location, MapObject, Pose, Room
    from walkie_world.people.store import PersonRecord
    from walkie_world.scene.ingest import ObjectObservation
    from walkie_world.scene.store import ObjectNode, Relation
    from walkie_world.world import PlaceMatch, WalkieWorld
