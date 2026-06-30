"""Dynamic object scene graph: the numpy SceneStore + relation derivation + the
producer→model ingest DTO. All pure numpy/scipy — no camera, Open3D, or chromadb.
"""

from walkie_world.scene.ingest import ObjectObservation
from walkie_world.scene.relations import derive_relations
from walkie_world.scene.store import (
    BuiltScene,
    ObjectNode,
    Relation,
    SceneStore,
    aabb_of,
    cosine,
    l2,
)

__all__ = [
    "BuiltScene",
    "ObjectNode",
    "ObjectObservation",
    "Relation",
    "SceneStore",
    "aabb_of",
    "cosine",
    "derive_relations",
    "l2",
]
