"""Offline unit tests for the object-detection confidence floor (skills._above_floor
/ skills._conf_floor). Pure logic — no robot/LLM — so it is tested here directly.

The floor protects two scored paths: `count` (objects) does ``len(dets)`` and
`find_object` calls the single best detection "found"; without a floor a stray
low-confidence box inflates the count or produces a false "found" claim. Default 0
must keep the pre-gate behaviour exactly (drop nothing).
"""

from __future__ import annotations

from tasks.GPSR.skills import _above_floor, _conf_floor


class _Det:
    """Minimal stand-in for client.image.DetectedObject (only .confidence is read)."""

    def __init__(self, confidence):
        self.confidence = confidence


def test_floor_zero_keeps_everything():
    dets = [_Det(0.9), _Det(0.1), _Det(0.0), _Det(None)]
    assert _above_floor(dets, 0.0) == dets  # default path: nothing dropped


def test_floor_drops_below_and_keeps_at_or_above():
    dets = [_Det(0.9), _Det(0.30), _Det(0.29), _Det(0.0)]
    kept = _above_floor(dets, 0.30)
    assert [d.confidence for d in kept] == [0.9, 0.30]  # boundary is inclusive


def test_floor_treats_missing_confidence_as_zero():
    dets = [_Det(None), _Det(0.5)]
    assert [d.confidence for d in _above_floor(dets, 0.3)] == [0.5]


def test_floor_zero_is_a_noop_copy_not_the_same_list():
    dets = [_Det(0.5)]
    out = _above_floor(dets, 0.0)
    assert out == dets and out is not dets  # safe to mutate without touching caller's list


def test_empty_in_empty_out():
    assert _above_floor([], 0.5) == []


def test_conf_floor_default_is_zero(monkeypatch):
    monkeypatch.delenv("GPSR_DETECT_CONF_MIN", raising=False)
    assert _conf_floor() == 0.0


def test_conf_floor_reads_env(monkeypatch):
    monkeypatch.setenv("GPSR_DETECT_CONF_MIN", "0.35")
    assert _conf_floor() == 0.35


def test_conf_floor_tolerates_garbage(monkeypatch):
    monkeypatch.setenv("GPSR_DETECT_CONF_MIN", "not_a_number")
    assert _conf_floor() == 0.0  # never raise into the detect path
