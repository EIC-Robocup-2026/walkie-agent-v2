"""Offline tests for the HRI slice builders (no robot/LLM/network).

The HRI runner selects a slice via ``HRI_SLICE`` (seats / greet / follow_host /
full) instead of commenting steps in and out of the flow. These lock the
slice → Task wiring: the ``full`` slice is the 12-step Receptionist sequence
(guest 1 fully handled before guest 2), and each isolated slice maps to its
bring-up harness. Construction is pure — per-run state lives in ``prepare_run``.
"""

from __future__ import annotations

from tasks.HRI.subtasks import (
    build_follow_host_slice,
    build_greet_slice,
    build_hri_task,
    build_seats_slice,
    prepare_run,
)


class FakeCtx:
    """Minimal TaskContext stand-in: the builders only need .data / .people."""

    def __init__(self):
        self.data = {}
        self.people = None


def _names(task):
    return [type(s).__name__ for s in task.subtasks]


def _guest_order(task):
    """Guest index of each step that targets a specific guest, in flow order."""
    return [int(s.name.split("guest ")[1].rstrip(")"))
            for s in task.subtasks if "guest " in s.name]


def test_full_flow_is_the_twelve_step_receptionist_sequence():
    task = build_hri_task(FakeCtx())
    assert task.name == "HRI"
    assert _names(task) == [
        "GoToDoor", "GreetAndLearn", "GuideToLivingRoom", "OfferSeat",
        "GoToDoor", "GreetAndLearn", "ReceiveBag", "GuideToLivingRoom",
        "OfferSeat", "AuditIdentities", "IntroduceGuests", "FollowHostAndDropBag",
    ]


def test_full_flow_finishes_guest_one_before_greeting_guest_two():
    # Each guest is greeted → guided → seated before the next; never interleaved.
    assert _guest_order(build_hri_task(FakeCtx())) == [1, 1, 1, 1, 2, 2, 2, 2]


def test_slice_builders_map_to_their_harness():
    assert _names(build_seats_slice(FakeCtx())) == ["TestScanSeats"]
    assert _names(build_greet_slice(FakeCtx())) == ["GreetAndLearn"]
    assert _names(build_follow_host_slice(FakeCtx())) == ["TestRememberAndFollowHost"]


def test_greet_slice_targets_guest_one():
    [greet] = build_greet_slice(FakeCtx()).subtasks
    assert greet.guest == 1


def test_prepare_run_is_safe_without_people_store():
    # people=None -> skips clear(); reset_people_positions must still no-op cleanly.
    prepare_run(FakeCtx())  # must not raise


def test_every_hri_score_key_exists_in_the_sheet():
    """Every ctx.score("...") in the HRI subtasks must be a real HRI_SHEET key.

    The award calls run only on the robot, so a typo'd key would otherwise only
    surface live (where ctx.score swallows it). This grep-and-check catches it
    offline — the runtime cost of a wrong key is a silently-missed point.
    """
    import pathlib
    import re

    import tasks.HRI.subtasks as subtasks_mod
    from tasks.HRI.scoring import HRI_SHEET

    src = pathlib.Path(subtasks_mod.__file__).read_text()
    keys = set(re.findall(r'ctx\.score\(\s*"([^"]+)"', src))
    assert keys, "expected ctx.score() award calls in the HRI subtasks"
    sheet_keys = {ln.key for ln in HRI_SHEET.lines}
    assert keys <= sheet_keys, f"HRI score keys not in HRI_SHEET: {sorted(keys - sheet_keys)}"
