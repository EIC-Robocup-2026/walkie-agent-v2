"""walkie_graphs — small open-vocabulary 3D scene-graph spatial memory for Walkie.

Segment (object detection + mask) → lift masked depth to 3D world points → fuse
across views into object nodes → derive distance-based relations → store
(ChromaDB + .npz point clouds + NetworkX edges) → visualize (Rerun) → export to
text for the LLM.

Typical use (constructed with a LangChain model, the AI client, and the hardware
interface — the same trio the agents take)::

    from walkie_graphs import WalkieGraphs

    graphs = WalkieGraphs(model=model, walkieAI=walkieAI, walkie=walkie)
    graphs.start()                       # background observer thread
    ...
    hits = graphs.query_text("where is the mug?")
    print(graphs.to_text_description())
    graphs.stop()

The heavy lifting lives in :mod:`walkie_graphs.geometry` (camera math),
:mod:`walkie_graphs.memory` (the node/edge store), :mod:`walkie_graphs.service`
(the background thread), and :mod:`walkie_graphs.viz` (optional Rerun).
"""

from __future__ import annotations

import os
from typing import Optional

from . import geometry
from .memory import Detection3D, GraphMemory, ObjectNode, Relation
from .service import WalkieGraphsService

__all__ = [
    "WalkieGraphs",
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
    """

    def __init__(
        self,
        model=None,
        walkieAI=None,
        walkie=None,
        *,
        memory: Optional[GraphMemory] = None,
        viz=None,
    ) -> None:
        self.model = model
        self.walkieAI = walkieAI
        self.walkie = walkie

        embed_text = None
        if walkieAI is not None:
            def embed_text(query: str, _ai=walkieAI):
                return _ai.image_embed.embed_text(query)

        self.memory = memory if memory is not None else GraphMemory.from_env(embed_text=embed_text)
        self.viz = viz if viz is not None else _build_viz()
        self._service: Optional[WalkieGraphsService] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _ensure_service(self) -> WalkieGraphsService:
        if self._service is None:
            self._service = WalkieGraphsService(
                self.walkieAI, self.walkie, self.memory, model=self.model, viz=self.viz
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
    # Externally-driven ingestion (perception owns capture+detect in production)
    # ------------------------------------------------------------------
    def detection_prompts(self) -> Optional[list[str]]:
        """The open-vocabulary prompt list (interested classes) for the shared detector.

        Perception detects once per frame with these prompts, then feeds the result to
        :meth:`ingest_frame` — so the detector runs once instead of once here and once in
        perception. ``None`` when no interested classes are configured."""
        return self._ensure_service().interested or None

    def ingest_frame(self, img, detections, depth, *, tick: bool = True) -> dict[int, dict]:
        """Fold one externally-captured frame into the graph; return per-detection
        ``{"centroid", "caption"}`` (see :meth:`WalkieGraphsService.ingest_frame`).

        Builds the service lazily but does **not** start its thread — perception drives it.
        """
        return self._ensure_service().ingest_frame(img, detections, depth, tick=tick)

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

    def visualize(self) -> None:
        if self.viz is not None:
            pose = None
            if self.walkie is not None:
                try:
                    pose = self.walkie.status.get_position()
                except Exception:  # noqa: BLE001
                    pose = None
            self.viz.update(self.memory, robot_pose=pose)
