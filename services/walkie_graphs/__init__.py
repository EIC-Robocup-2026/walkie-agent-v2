"""Compatibility shim — walkie_graphs was split into walkie_world (the model) +
services.realtime_explore (the producer).

Kept during the migration so existing imports keep working:
``from services.walkie_graphs import WalkieGraphs`` resolves to the renamed
:class:`~services.realtime_explore.service.RealtimeExplore`; ``ObjectNode`` /
``Relation`` resolve to :mod:`walkie_world.scene.store`. The submodule shims
``services/walkie_graphs/scene.py`` and ``relations.py`` alias the walkie_world
homes. Removed in the final migration phase.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["WalkieGraphs", "ObjectNode", "Relation", "CameraSnapshot", "geometry", "pcd_ops"]

_LAZY = {
    # WalkieGraphs is the old name for the producer (RealtimeExplore).
    "WalkieGraphs": ("services.realtime_explore.service", "RealtimeExplore"),
    "ObjectNode": ("walkie_world.scene.store", "ObjectNode"),
    "Relation": ("walkie_world.scene.store", "Relation"),
    "CameraSnapshot": ("interfaces.devices.camera", "CameraSnapshot"),
    "geometry": ("interfaces.perception", "geometry"),
    "pcd_ops": ("services.realtime_explore", "pcd_ops"),
}


def __getattr__(name: str):  # PEP 562 — resolve public names lazily
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # static type hints only — never imported at runtime
    from interfaces.devices.camera import CameraSnapshot
    from services.realtime_explore.service import RealtimeExplore as WalkieGraphs
    from walkie_world.scene.store import ObjectNode, Relation
