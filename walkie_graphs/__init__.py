"""walkie_graphs — small open-vocabulary 3D scene-graph spatial memory for Walkie.

Segment (object detection + mask) → lift masked depth to 3D world points → fuse
across views into object nodes → derive distance-based relations → store
(ChromaDB + .npz point clouds + NetworkX edges) → visualize (Rerun) → export to
text for the LLM.

The package is built from small focused modules:

- :mod:`walkie_graphs.geometry` — camera intrinsics, world-pose composition, and
  masked-depth → world-point deprojection (pure numpy).
- :mod:`walkie_graphs.memory` — :class:`GraphMemory`, the persistent node/edge store.
- :mod:`walkie_graphs.service` — :class:`WalkieGraphsService`, the background thread.
- :mod:`walkie_graphs.viz` — optional Rerun real-time visualization.

The :class:`WalkieGraphs` facade (constructed with ``model``, ``walkieAI``,
``walkie``) ties them together; it is defined in :mod:`walkie_graphs.service` and
re-exported here once that module lands.
"""

from __future__ import annotations

__all__ = [
    "geometry",
]
