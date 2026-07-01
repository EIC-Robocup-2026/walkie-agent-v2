"""Interleave scheduler: merge several command plans into one room-batched order.

The Interleaved Task Bonus (rulebook 5.3, 200 pts) is awarded only when all three
commands are taken at once AND the interleaving is *meaningful* — "saving time or
reducing unnecessary movements". The cheapest meaningful interleave is to visit
each room ONCE: while the robot is in a room, do every command's pending step that
belongs there (plus location-agnostic steps), then move on. Combined with the
shared nav-dedup in dispatch.execute_interleaved, that removes the back-and-forth
a strictly-serial run makes when two commands touch the same room.

This module is PURE (no robot, no LLM) and offline-tested: it takes the typed
plans + the world model and returns the execution order. It deliberately is NOT
an LLM scheduler — a deterministic room-batching greedy is reliable, explainable,
and testable, which matters more than cleverness for a bonus that must not break
the serial path it falls back to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .plan import Plan, step_location

if TYPE_CHECKING:
    from walkie_world.world import WalkieWorld  # ctx.world; only its .locations is used here


def _region(world: WalkieWorld, location: str | None) -> str | None:
    """The room a step's location belongs to (a placement -> its room; a room ->
    itself; None/unknown -> as-is), so 'kitchen_table' and 'kitchen' batch together."""
    if location is None:
        return None
    loc = world.locations.get(location)
    if loc is not None and loc.room:
        return loc.room
    return location  # a room name, or an unknown location -> treat the name as a region


def interleave(indexed: list[tuple[int, Plan]], world: WalkieWorld) -> list[tuple[int, int]]:
    """Return the interleaved execution order as ``[(command_id, step_index), ...]``.

    Greedy room batching that PRESERVES each command's internal step order (a
    command's step *i* always precedes its step *i+1*): stay in the current room
    and drain every command's pending step whose region is the current room (or is
    location-agnostic) before travelling; when nothing remains here, advance the
    lowest-id command's next step (which sets the new room). Deterministic — ties
    break by ascending command id. Every step appears exactly once.
    """
    plans = {cid: plan for cid, plan in indexed}
    cmd_ids = [cid for cid, _ in indexed]
    ptr = {cid: 0 for cid in cmd_ids}
    total = sum(len(p.steps) for p in plans.values())

    def region_of(cid: int, idx: int) -> str | None:
        return _region(world, step_location(plans[cid].steps[idx]))

    order: list[tuple[int, int]] = []
    room: str | None = None
    while len(order) < total:
        cands = [(cid, ptr[cid]) for cid in cmd_ids if ptr[cid] < len(plans[cid].steps)]
        # Doable without travel: location-agnostic (None) or already in this room.
        here = [(cid, i) for (cid, i) in cands if region_of(cid, i) in (None, room)]
        cid, idx = here[0] if here else cands[0]
        new_room = region_of(cid, idx)
        if new_room is not None:
            room = new_room
        order.append((cid, idx))
        ptr[cid] += 1
    return order
