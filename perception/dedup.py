"""Pure-function dedup logic.

Given a fresh :class:`Detection` and a list of nearby SceneEntries from the
store, decide whether to INSERT a new record or UPDATE an existing one.

Decision tree (see ``docs/scene_perception_design.md``):

    1. If no candidates → INSERT.
    2. For each candidate (closest first):
         - cos_sim ≥ EMB_SIM_HIGH                          → UPDATE (visual match)
         - cos_sim ≥ EMB_SIM_LOW and L2 ≤ TIGHT_RADIUS    → UPDATE (failsafe)
    3. Otherwise → INSERT.

All four numeric thresholds are env-var-overridable so they can be tuned
without code changes:

    SCENE_DEDUP_RADIUS_M      default 0.5   used by the store's find_nearby
    SCENE_DEDUP_TIGHT_M       default 0.2   inner gate for the failsafe
    SCENE_EMB_SIM_HIGH        default 0.85
    SCENE_EMB_SIM_LOW         default 0.65

Reasoning for the defaults is in the design doc.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

from .types import DedupDecision, Detection, SceneEntry


def get_dedup_radius_m() -> float:
    return float(os.getenv("SCENE_DEDUP_RADIUS_M", "0.5"))


def get_tight_radius_m() -> float:
    return float(os.getenv("SCENE_DEDUP_TIGHT_M", "0.2"))


def get_emb_sim_high() -> float:
    return float(os.getenv("SCENE_EMB_SIM_HIGH", "0.85"))


def get_emb_sim_low() -> float:
    return float(os.getenv("SCENE_EMB_SIM_LOW", "0.65"))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity assuming both vectors are L2-normalized.

    Falls back to manual normalization if either vector isn't unit length.
    Returns 0.0 when a vector is degenerate (all-zero).
    """
    if len(a) != len(b):
        raise ValueError(f"Embedding dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    # If both already unit-normalized, sqrt(1)*sqrt(1)=1 and this is dot.
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def l2_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def classify(
    new: Detection,
    candidates: Sequence[SceneEntry],
) -> DedupDecision:
    """Decide whether ``new`` should INSERT or UPDATE one of ``candidates``.

    Candidates are assumed to already be the result of
    :meth:`SceneStore.find_nearby` with matching class — we never merge
    across classes here, so it's a precondition violation if the caller
    hands us a cross-class candidate.

    Iteration order: closest-first. The first candidate that passes any
    merge gate wins — we don't keep scanning for a "better" match. This
    matches the design doc's "closest wins" rule (test case #9).
    """
    if not candidates:
        return DedupDecision(action="insert", reason="no candidates within radius")

    high = get_emb_sim_high()
    low = get_emb_sim_low()
    tight = get_tight_radius_m()

    # Sort by L2 distance ascending — the caller may have already done this
    # but we re-sort to be defensive.
    sorted_candidates = sorted(
        candidates,
        key=lambda c: l2_distance(c.position, new.position),
    )

    for c in sorted_candidates:
        if c.class_name != new.class_name:
            raise ValueError(
                "classify() received a cross-class candidate "
                f"({c.class_name!r} vs {new.class_name!r}); "
                "callers must pre-filter by class."
            )

        sim = cosine_similarity(c.embedding, new.embedding)
        dist = l2_distance(c.position, new.position)

        if sim >= high:
            return DedupDecision(
                action="update",
                target_id=c.id,
                reason=f"cosine {sim:.3f} ≥ EMB_SIM_HIGH ({high}); dist {dist:.2f}m",
            )
        if sim >= low and dist <= tight:
            return DedupDecision(
                action="update",
                target_id=c.id,
                reason=(
                    f"cosine {sim:.3f} ≥ EMB_SIM_LOW ({low}) "
                    f"and dist {dist:.2f}m ≤ TIGHT_RADIUS ({tight}m)"
                ),
            )

    return DedupDecision(
        action="insert",
        reason=f"{len(sorted_candidates)} candidate(s) failed both merge gates",
    )


def merged_position(
    old_pos: tuple[float, float, float],
    old_count: int,
    new_pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Running mean of the position, weighted by previous sighting count."""
    n = max(1, old_count)
    return (
        (old_pos[0] * n + new_pos[0]) / (n + 1),
        (old_pos[1] * n + new_pos[1]) / (n + 1),
        (old_pos[2] * n + new_pos[2]) / (n + 1),
    )


def merged_confidence(old_conf: float, old_count: int, new_conf: float) -> float:
    """Running mean of detection confidence."""
    n = max(1, old_count)
    return (old_conf * n + new_conf) / (n + 1)
