"""Shared long-term object-memory lookup for agent tools.

The main Walkie agent and the Walkie Database sub-agent share this helper so
the lookup logic and the text format stay identical across agents.

Backend preference:
  1. The CLIP-backed :class:`perception.SceneStore`. We query the **caption
     text** index first (``text_query``) — "where is the mug?" matching a
     record captioned "a white coffee mug" is far more reliable than
     comparing the query against image vectors. If the caption index is empty
     (e.g. data collected before it existed, not yet reindexed) we fall back
     to the CLIP image search (``semantic_query``).
  2. The legacy :class:`db.walkie_db.WalkieVectorDB` (``query_objects``) —
     fallback for runs built by the older explore stage.

Either way the result is a human-readable string listing map-frame
coordinates the actuator can navigate to, plus the stored caption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid importing heavy deps at agent-build time
    from db.walkie_db import WalkieVectorDB
    from perception import SceneStore


def lookup_object_in_memory(
    object_name: str,
    *,
    scene_store: "Optional[SceneStore]" = None,
    db: "Optional[WalkieVectorDB]" = None,
    n_results: int = 5,
) -> str:
    """Search long-term memory for ``object_name`` and format the matches.

    Prefers ``scene_store`` (caption text search, then CLIP image search)
    when supplied, otherwise falls back to ``db``. Returns a ready-to-speak
    summary string.
    """
    if scene_store is not None:
        try:
            # Caption-first: text→text against the stored descriptions.
            entries = scene_store.text_query(object_name, n_results=n_results)
            if not entries:
                # Caption index empty or no caption hit — fall back to the
                # CLIP image-embedding search so we still answer.
                entries = scene_store.semantic_query(object_name, n_results=n_results)
        except Exception as e:  # noqa: BLE001 — surface, don't crash the agent
            return f"Scene memory lookup failed: {e}"
        if not entries:
            return f"No record of '{object_name}' in scene memory."
        lines = [f"Top matches for '{object_name}':"]
        for e in entries:
            x, y, z = e.position
            lines.append(
                f"- {e.class_name} @ ({x:+.2f}, {y:+.2f}, {z:+.2f}) "
                f"conf={e.position_conf:.2f} sightings={e.sightings} "
                f"caption={e.caption!r}"
            )
        return "\n".join(lines)

    if db is not None:
        hits = db.query_objects(object_name, n_results=n_results)
        if not hits:
            return f"No record of '{object_name}' in memory."
        lines = [f"Top matches for '{object_name}':"]
        for h in hits:
            x, y, z = h["position"]
            lines.append(
                f"- {h.get('class_name', '?')} @ ({x:+.2f}, {y:+.2f}, {z:+.2f}) "
                f"conf={h.get('confidence', 0):.2f} "
                f"sightings={h.get('sightings', '?')} "
                f"caption={h.get('caption', '')!r}"
            )
        return "\n".join(lines)

    return f"No memory backend available to look up '{object_name}'."
