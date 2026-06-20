"""walkie_graphs — small open-vocabulary 3D scene-graph spatial memory for Walkie.

Capture-centric (ConceptGraphs-style): each frame becomes one **Capture** —
detected segmentations lifted to per-detection world-point segments plus the
classless background remainder (walls/floor) — registered against the map with
ONE rigid ICP correction (pose error is per-capture, not per-object), then
fused across views into object nodes whose clouds derive from their captures'
segments. The map is **self-correcting**: periodic maintenance rescues ghost
duplicates (ICP-aligned re-merge), refines each object's shape (co-registering
its segments), and carves free space (removing geometry a trusted later view
sees straight through — moved-object ghosts, edge-shadow trails). Distance-based
relations are derived on a cadence; everything is stored (ChromaDB + .npz point
clouds/segments + NetworkX edges), visualized (Rerun, background included), and
exportable to text for the LLM.

Typical use (constructed with a LangChain model, the AI client, and the hardware
interface — the same trio the agents take)::

    from walkie_graphs import WalkieGraphs

    graphs = WalkieGraphs(model=model, walkieAI=walkieAI, walkie=walkie)
    graphs.start()                       # background observer thread
    ...
    hits = graphs.query_text("where is the mug?")
    print(graphs.to_text_description())
    graphs.stop()

The heavy lifting lives in :mod:`interfaces.perception.geometry` (camera math,
re-exported here as ``walkie_graphs.geometry`` for back-compat),
:mod:`walkie_graphs.memory` (the node/edge store), :mod:`walkie_graphs.service`
(the background thread), and :mod:`walkie_graphs.viz` (optional Rerun).
"""

from __future__ import annotations

import os
from typing import Optional

from interfaces.devices.camera import CameraSnapshot
from interfaces.perception import geometry  # back-compat: ``walkie_graphs.geometry``

from .capture import Capture, CaptureStore, Segment
from .memory import Detection3D, GraphMemory, ObjectNode, Relation
from .service import WalkieGraphsService

__all__ = [
    "WalkieGraphs",
    "CameraSnapshot",
    "Capture",
    "CaptureStore",
    "Segment",
    "GraphMemory",
    "WalkieGraphsService",
    "ObjectNode",
    "Relation",
    "Detection3D",
    "geometry",
]


def _build_viz():
    """Construct the configured visualizer, or None (lazy: rerun is optional)."""
    backend = os.getenv("WALKIE_GRAPHS_VIZ", "none").lower()
    if backend in ("", "none"):
        return None
    try:
        from .viz import build_viz

        return build_viz(backend)
    except Exception as e:  # noqa: BLE001 — viz is best-effort
        print(f"[graphs] visualizer '{backend}' unavailable: {e}")
        return None


class WalkieGraphs:
    """Facade tying the store + observer + visualizer together.

    Args:
        model: A LangChain chat model (accepted for forward-compat; the current
            server-caption + geometric-edge pipeline does not use an LLM).
        walkieAI: :class:`client.WalkieAIClient` for detection/caption/embedding.
        walkie: :class:`interfaces.walkie_interface.WalkieInterface` for the camera,
            pose, lift, and head tilt.
        memory: Override the store (mainly for tests); built from env otherwise.
        viz: Override the visualizer; built from ``WALKIE_GRAPHS_VIZ`` otherwise.
        snapshot_path: Where the observer loop writes the live ``perception.json`` snapshot
            the agents read each turn. ``None`` (default) writes no snapshot.
    """

    def __init__(
        self,
        model=None,
        walkieAI=None,
        walkie=None,
        *,
        memory: Optional[GraphMemory] = None,
        viz=None,
        snapshot_path=None,
    ) -> None:
        self.model = model
        self.walkieAI = walkieAI
        self.walkie = walkie
        self.snapshot_path = snapshot_path

        embed_text = None
        if walkieAI is not None:
            def embed_text(query: str, _ai=walkieAI):
                return _ai.image.embed_text(query)

        self.memory = memory if memory is not None else GraphMemory.from_env(embed_text=embed_text)
        self.viz = viz if viz is not None else _build_viz()
        self._service: Optional[WalkieGraphsService] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _ensure_service(self) -> WalkieGraphsService:
        if self._service is None:
            self._service = WalkieGraphsService(
                self.walkieAI,
                self.walkie,
                self.memory,
                model=self.model,
                viz=self.viz,
                snapshot_path=self.snapshot_path,
            )
        return self._service

    def start(self) -> None:
        """Start the background observer thread (no-op if already running)."""
        svc = self._ensure_service()
        if not svc.is_alive():
            svc.start()

    def stop(self) -> None:
        """Stop the background observer thread."""
        if self._service is not None:
            self._service.stop_and_join(timeout=5)
            self._service = None

    def observe(self) -> list[ObjectNode]:
        """Process a single live RGB-D frame (manual path; use instead of start())."""
        touched = self._ensure_service()._observe_once()
        self.memory.derive_relations()
        return touched

    # ------------------------------------------------------------------
    # Query passthroughs (used by the database agent)
    # ------------------------------------------------------------------
    def query_text(self, query: str, k: int = 5, *, near=None, radius=None) -> list[ObjectNode]:
        return self.memory.query_text(query, k, near=near, radius=radius)

    def query_near(self, center, radius: float) -> list[ObjectNode]:
        return self.memory.query_near(center, radius)

    def recently_seen(self, limit: int = 5) -> list[ObjectNode]:
        return self.memory.recently_seen(limit)

    def all_objects(self) -> list[ObjectNode]:
        return self.memory.all_objects()

    def get(self, node_id: str) -> Optional[ObjectNode]:
        return self.memory.get(node_id)

    def relations_of(self, node_id: str) -> list[Relation]:
        return self.memory.relations_of(node_id)

    def to_text_description(self) -> str:
        return self.memory.to_text_description()

    # ------------------------------------------------------------------
    # Optional LLM refinement (Tier 3) — on-demand triggers using the wired model
    # ------------------------------------------------------------------
    def refine_captions(self, *, limit: Optional[int] = None, use_images: bool = False) -> int:
        """Condense each object's view captions into one coherent caption (needs ``model``)."""
        return self.memory.refine_captions(self.model, limit=limit, use_images=use_images)

    def infer_edges(self, *, max_pairs: Optional[int] = None) -> int:
        """Label spatial relations between nearby objects with the LLM (needs ``model``)."""
        return self.memory.infer_edges_llm(self.model, max_pairs=max_pairs)

    def visualize(self) -> None:
        if self.viz is not None:
            pose = None
            if self.walkie is not None:
                try:
                    pose = self.walkie.status.get_position()
                except Exception:  # noqa: BLE001
                    pose = None
            self.viz.update(self.memory, robot_pose=pose)
