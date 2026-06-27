"""walkie_graphs — Walkie's 3D scene-graph spatial memory.

Two backends live behind the **same** public facade, selected by
``WALKIE_GRAPHS_BACKEND`` (default ``v1``):

- **v1** (``facade.WalkieGraphs`` over :mod:`~services.walkie_graphs.memory`) — the
  original real-time ConceptGraphs-style incremental pipeline (ChromaDB + capture
  segments + per-frame ICP + staggered maintenance).
- **v2** (:mod:`~services.walkie_graphs.service_v2`) — the overhaul: cheap continuous
  capture into an on-disk snapshot ring buffer + occasional **offline batch builds**
  with globally-consistent poses (pose-graph + TSDF, Open3D), a lean numpy
  :class:`~services.walkie_graphs.scene.SceneStore` (no ChromaDB for the scene), and
  batch constrained-agglomerative association. See ``docs/WALKIE_GRAPHS.md``.

Both expose the identical query contract the Database agent and GPSR depend on:
``query_text / query_near / recently_seen / all_objects / get / relations_of /
to_text_description`` plus ``start() / stop() / observe()``.

This module is intentionally **import-light** (PEP 562 lazy attributes): importing the
package, or any submodule, must not eagerly pull ChromaDB / Open3D / the camera stack.
Names resolve on first access to the backend selected by the env var.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

__all__ = [
    "WalkieGraphs",
    "CameraSnapshot",
    "ObjectNode",
    "Relation",
    "geometry",
    # v1-only re-exports (kept for back-compat / the v1 test suite)
    "Capture",
    "CaptureStore",
    "Segment",
    "GraphMemory",
    "WalkieGraphsService",
    "Detection3D",
]


def backend() -> str:
    """The active scene-memory backend: ``"v1"`` (default) or ``"v2"``."""
    return os.getenv("WALKIE_GRAPHS_BACKEND", "v1").strip().lower()


# Shared (backend-independent) lazy targets.
_SHARED = {
    "CameraSnapshot": ("interfaces.devices.camera", "CameraSnapshot"),
    "geometry": ("interfaces.perception", "geometry"),
}

# v1-only lazy targets (the original store/service/capture types).
_V1_ONLY = {
    "Capture": ("services.walkie_graphs.capture", "Capture"),
    "CaptureStore": ("services.walkie_graphs.capture", "CaptureStore"),
    "Segment": ("services.walkie_graphs.capture", "Segment"),
    "GraphMemory": ("services.walkie_graphs.memory", "GraphMemory"),
    "WalkieGraphsService": ("services.walkie_graphs.service", "WalkieGraphsService"),
    "Detection3D": ("services.walkie_graphs.memory", "Detection3D"),
}


def _resolve_facade():
    if backend() == "v2":
        return importlib.import_module("services.walkie_graphs.service_v2"), "WalkieGraphs"
    return importlib.import_module("services.walkie_graphs.facade"), "WalkieGraphs"


def __getattr__(name: str):  # PEP 562 — resolve public names lazily
    if name == "WalkieGraphs":
        mod, attr = _resolve_facade()
        return getattr(mod, attr)
    if name in ("ObjectNode", "Relation"):
        # Bind the node/relation types of whichever backend is active so
        # ``isinstance`` checks and re-exports match the running store.
        pkg = "services.walkie_graphs.scene" if backend() == "v2" else "services.walkie_graphs.memory"
        return getattr(importlib.import_module(pkg), name)
    if name in _SHARED:
        mod_name, attr = _SHARED[name]
        return getattr(importlib.import_module(mod_name), attr)
    if name in _V1_ONLY:
        mod_name, attr = _V1_ONLY[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # static type hints only — never imported at runtime
    from interfaces.devices.camera import CameraSnapshot
    from .facade import WalkieGraphs
    from .memory import Capture, CaptureStore, Detection3D, GraphMemory, ObjectNode, Relation, Segment
    from .service import WalkieGraphsService
