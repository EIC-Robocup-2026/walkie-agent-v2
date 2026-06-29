"""The producer→model ingest contract: one fused object observation.

:class:`ObjectObservation` is what :meth:`walkie_world.world.WalkieWorld.observe_objects`
(and :meth:`walkie_world.scene.store.SceneStore.merge`) consume. The perception
producer (``services/realtime_explore``) builds these from a window of snapshots
(lift → associate) and hands them over; the model folds them into the persisted
scene graph. Defining the DTO here — in the model — keeps it the single source of
truth so the producer's ``associate`` cannot drift from what ``merge`` reads.

``SceneStore.merge`` reads fields via ``getattr``, so any duck-compatible object
works; this dataclass documents the canonical shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ObjectObservation:
    """One associated object cluster ready to merge into the scene store.

    The denoised fused ``points`` cloud (``(N, 3)`` world frame) and its AABB
    summary, a union of member captions, the L2-normalised mean CLIP embedding, and
    provenance counters (``n_obs`` = member count; ``ts_first``/``ts_last``).
    """

    class_name: str
    class_id: Optional[int]
    conf: float
    captions: list[str]
    clip_emb: list[float]
    ts_first: float
    ts_last: float
    n_obs: int
    points: np.ndarray
    centroid: tuple[float, float, float]
    extent: tuple[float, float, float]
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
