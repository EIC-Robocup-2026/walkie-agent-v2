"""realtime_explore — Walkie's perception PRODUCER (capture + batch build).

A cheap CAPTURE thread streams RGB-D frames + detections into an on-disk snapshot
ring buffer and writes the live ``perception.json``; an occasional BATCH BUILD worker
fuses a window of snapshots (refine poses → lift masks → constrained-agglomerative
association) and pushes the resulting observations into the shared
:class:`~walkie_world.world.WalkieWorld` via ``world.observe_objects``. The world model
(:mod:`walkie_world`) owns the scene store, relations and all queries.

Public surface::

    from services.realtime_explore import RealtimeExplore
    explore = RealtimeExplore(model=model, walkieAI=walkieAI, walkie=walkie,
                              world=world, snapshot_path=...)
    explore.start()   # capture thread + batch build worker
    explore.stop()

Intentionally **import-light** (PEP 562 lazy attributes): importing the package must
not eagerly pull Open3D or the camera/SDK stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["RealtimeExplore", "ObjectNode", "Relation", "CameraSnapshot", "geometry"]

_LAZY = {
    "RealtimeExplore": ("services.realtime_explore.service", "RealtimeExplore"),
    "ObjectNode": ("walkie_world.scene.store", "ObjectNode"),
    "Relation": ("walkie_world.scene.store", "Relation"),
    "CameraSnapshot": ("interfaces.devices.camera", "CameraSnapshot"),
    "geometry": ("interfaces.perception", "geometry"),
}


def __getattr__(name: str):  # PEP 562 — resolve public names lazily
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # static type hints only — never imported at runtime
    from interfaces.devices.camera import CameraSnapshot
    from interfaces.perception import geometry
    from services.realtime_explore.service import RealtimeExplore
    from walkie_world.scene.store import ObjectNode, Relation
