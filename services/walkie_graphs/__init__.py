"""walkie_graphs — Walkie's 3D scene-graph spatial memory (batch-snapshot backend).

Cheap continuous **capture** into an on-disk snapshot ring buffer (`graph_buffer/`) +
occasional offline **batch builds** with globally-consistent poses → a lean numpy
:class:`~services.walkie_graphs.scene.SceneStore` (`graph_scene/`, no ChromaDB for the
scene) populated by batch constrained-agglomerative association. The full pipeline and
every tuning knob are documented in ``docs/WALKIE_GRAPHS.md``.

Public surface (what the Database agent + GPSR depend on)::

    from services.walkie_graphs import WalkieGraphs

    graphs = WalkieGraphs(model=model, walkieAI=walkieAI, walkie=walkie, snapshot_path=...)
    graphs.start()                       # capture thread + batch build worker
    hits = graphs.query_text("where is the mug?")
    print(graphs.to_text_description())
    graphs.stop()

This module is intentionally **import-light** (PEP 562 lazy attributes): importing the
package — or any submodule — must not eagerly pull Open3D or the camera/SDK stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["WalkieGraphs", "ObjectNode", "Relation", "CameraSnapshot", "geometry"]

_LAZY = {
    "WalkieGraphs": ("services.walkie_graphs.service", "WalkieGraphs"),
    "ObjectNode": ("services.walkie_graphs.scene", "ObjectNode"),
    "Relation": ("services.walkie_graphs.scene", "Relation"),
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
    from .scene import ObjectNode, Relation
    from .service import WalkieGraphs
