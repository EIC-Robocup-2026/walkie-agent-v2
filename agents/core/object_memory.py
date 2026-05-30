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

Quality filters (scene backend only): callers can restrict matches to a
spatial ball (``within_radius_of`` + ``max_distance_m``), to recent sightings
(``min_last_seen_ts``), and to confidently-positioned records
(``min_position_conf``). These keep "where is X?" answers grounded — the robot
shouldn't be sent to a low-confidence or stale memory when a better one exists.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid importing heavy deps at agent-build time
    from db.walkie_db import WalkieVectorDB
    from perception import SceneEntry, SceneStore


def query_min_conf() -> float:
    """Confidence floor for long-term lookups (drops weakly-positioned records).

    Code default 0.0 (no filter, what tests assume); config.toml sets the
    production floor. See the "code default + config.toml" convention.
    """
    try:
        return float(os.getenv("SCENE_QUERY_MIN_CONF", "0.0"))
    except ValueError:
        return 0.0


def robot_xy(walkie) -> "Optional[tuple[float, float, float]]":
    """The robot's current planar map pose ``(x, y, 0)``, or None if unknown.

    Duck-typed on ``walkie.status.get_position()`` so this module stays free of
    a hard ``WalkieInterface`` import. Used to anchor "near me" lookups.
    """
    try:
        pose = walkie.status.get_position()
    except Exception:  # noqa: BLE001 — telemetry hiccup shouldn't break a lookup
        return None
    if not pose:
        return None
    return (float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), 0.0)


def _fmt_scene_entries(object_name: str, entries: "list[SceneEntry]") -> str:
    lines = [f"Top matches for '{object_name}':"]
    for e in entries:
        x, y, z = e.position
        lines.append(
            f"- {e.class_name} @ ({x:+.2f}, {y:+.2f}, {z:+.2f}) "
            f"conf={e.position_conf:.2f} sightings={e.sightings} "
            f"caption={e.caption!r}"
        )
    return "\n".join(lines)


def lookup_object_in_memory(
    object_name: str,
    *,
    scene_store: "Optional[SceneStore]" = None,
    db: "Optional[WalkieVectorDB]" = None,
    n_results: int = 5,
    within_radius_of: "Optional[tuple[float, float, float]]" = None,
    max_distance_m: Optional[float] = None,
    min_last_seen_ts: Optional[float] = None,
    min_position_conf: float = 0.0,
) -> str:
    """Search long-term memory for ``object_name`` and format the matches.

    Prefers ``scene_store`` (caption text search, then CLIP image search)
    when supplied, otherwise falls back to ``db``. Returns a ready-to-speak
    summary string.

    ``within_radius_of`` + ``max_distance_m`` restrict to a map-frame ball
    (e.g. "near me"); ``min_last_seen_ts`` drops sightings older than a cutoff;
    ``min_position_conf`` drops low-confidence positions. All filters apply to
    the scene backend only — the legacy ``db`` path ignores them.
    """
    if scene_store is not None:
        try:
            # Over-fetch so the confidence post-filter still has candidates to
            # return after pruning weak ones.
            fetch = max(n_results * 3, n_results)
            common = dict(
                n_results=fetch,
                within_radius_of=within_radius_of,
                max_distance_m=max_distance_m,
                min_last_seen_ts=min_last_seen_ts,
            )
            # Caption-first: text→text against the stored descriptions.
            entries = scene_store.text_query(object_name, **common)
            if not entries:
                # Caption index empty or no caption hit — fall back to the
                # CLIP image-embedding search so we still answer.
                entries = scene_store.semantic_query(object_name, **common)
        except Exception as e:  # noqa: BLE001 — surface, don't crash the agent
            return f"Scene memory lookup failed: {e}"

        n_before = len(entries)
        if min_position_conf > 0.0:
            entries = [e for e in entries if e.position_conf >= min_position_conf]
        entries = entries[:n_results]

        if not entries:
            extra = ""
            if n_before and min_position_conf > 0.0:
                extra = (
                    f" ({n_before} match(es) were below the "
                    f"{min_position_conf:.2f} confidence floor)"
                )
            return f"No confident record of '{object_name}' in scene memory.{extra}"
        return _fmt_scene_entries(object_name, entries)

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
