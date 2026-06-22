"""Offline unit tests for the Restaurant task's pure logic — no robot/LLM/network.

Restaurant's in-suite coverage was zero (the dry-runs live in manual_tests/, which
pytest deliberately does not collect), yet the pure helpers below carry the whole
no-arm tier (~960 pts): caller detection (160), distinct-customer accounting (the
rulebook's ">= 2 customers" gate), order confirmation (320), and the map->base /
reach geometry the manipulation path is built on. These lock that logic against
regression the way tests/test_gpsr_* do for GPSR.
"""

from __future__ import annotations

import pytest

from client import PersonPose, PoseKeypoint
from tasks.Restaurant.skills import (
    Caller,
    _cxcywh_to_xyxy,
    _dedup_callers,
    _said_no,
    _scan_offsets,
    exclude_handled,
    is_calling,
)

_COCO = {"left_shoulder": 5, "right_shoulder": 6, "left_wrist": 9, "right_wrist": 10}


def _person(*, bbox=(100.0, 200.0, 80.0, 400.0), conf=0.9, **kp_y):
    """A PersonPose with the named keypoints at a given y (x fixed, conf high).

    kp_y maps a keypoint name -> (y, confidence); confidence defaults to 0.9.
    """
    kps = []
    for name, val in kp_y.items():
        y, c = val if isinstance(val, tuple) else (val, 0.9)
        kps.append(PoseKeypoint(x=100.0, y=float(y), confidence=c, name=name, index=_COCO[name]))
    return PersonPose(bbox=bbox, confidence=conf, keypoints=kps)


# --- is_calling (the 160-pt caller-detection heuristic) --------------------

def test_is_calling_true_when_a_wrist_is_clearly_above_its_shoulder():
    # image y grows downward, so a raised hand has the smaller y
    p = _person(left_shoulder=200, left_wrist=80)  # bbox h=400, margin=0.05*400=20
    assert is_calling(p, margin_frac=0.05, kp_conf=0.3) is True


def test_is_calling_false_when_hand_is_down():
    p = _person(left_shoulder=200, left_wrist=300)
    assert is_calling(p, margin_frac=0.05, kp_conf=0.3) is False


def test_is_calling_false_within_margin():
    # wrist 10px above shoulder but margin is 20px -> a resting arm, not a call
    p = _person(left_shoulder=200, left_wrist=190)
    assert is_calling(p, margin_frac=0.05, kp_conf=0.3) is False


def test_is_calling_either_arm_counts():
    p = _person(right_shoulder=200, right_wrist=60)
    assert is_calling(p, margin_frac=0.05, kp_conf=0.3) is True


def test_is_calling_ignores_low_confidence_keypoints():
    # the raised wrist is below the confidence gate -> not trusted
    p = _person(left_shoulder=(200, 0.9), left_wrist=(60, 0.1))
    assert is_calling(p, margin_frac=0.05, kp_conf=0.3) is False


def test_is_calling_needs_both_shoulder_and_wrist():
    p = _person(left_wrist=60)  # no shoulder to compare against
    assert is_calling(p, margin_frac=0.05, kp_conf=0.3) is False


def test_is_calling_margin_scales_with_bbox_height():
    # same 30px raise: a call for a small person (h=200 -> margin 10), not a tall
    # one (h=2000 -> margin 100).
    near = _person(bbox=(0, 0, 80, 200), left_shoulder=200, left_wrist=170)
    far = _person(bbox=(0, 0, 80, 2000), left_shoulder=200, left_wrist=170)
    assert is_calling(near, margin_frac=0.05, kp_conf=0.3) is True
    assert is_calling(far, margin_frac=0.05, kp_conf=0.3) is False


# --- caller dedup + distinct-customer accounting ---------------------------

def _caller(xy, conf=0.5):
    return Caller(world_xy=xy, bearing=0.0, bbox_xyxy=(0, 0, 1, 1), confidence=conf)


def test_dedup_callers_collapses_within_radius_keeping_the_confident_one():
    a = _caller((0.0, 0.0), conf=0.4)
    b = _caller((0.3, 0.0), conf=0.8)  # 0.3 m away -> same person, two views
    out = _dedup_callers([a, b], radius_m=0.6)
    assert len(out) == 1 and out[0].confidence == 0.8


def test_dedup_callers_keeps_distinct_people():
    a = _caller((0.0, 0.0))
    b = _caller((2.0, 0.0))
    assert len(_dedup_callers([a, b], radius_m=0.6)) == 2


def test_exclude_handled_drops_already_served_customer():
    callers = [_caller((0.0, 0.0)), _caller((3.0, 0.0))]
    out = exclude_handled(callers, handled_xys=[(0.2, 0.0)], radius_m=1.0)
    assert [c.world_xy for c in out] == [(3.0, 0.0)]


def test_exclude_handled_empty_handled_keeps_all():
    callers = [_caller((0.0, 0.0)), _caller((3.0, 0.0))]
    assert exclude_handled(callers, handled_xys=[], radius_m=1.0) == callers


# --- scan sweep + bbox conversion ------------------------------------------

def test_scan_offsets_cover_the_arc_symmetrically(monkeypatch):
    monkeypatch.setenv("RESTAURANT_SCAN_ARC_DEG", "120")
    monkeypatch.setenv("RESTAURANT_SCAN_STEP_DEG", "30")
    assert _scan_offsets() == [-60.0, -30.0, 0.0, 30.0, 60.0]


def test_scan_offsets_step_has_a_floor(monkeypatch):
    monkeypatch.setenv("RESTAURANT_SCAN_ARC_DEG", "10")
    monkeypatch.setenv("RESTAURANT_SCAN_STEP_DEG", "0")  # floored to 5
    assert _scan_offsets() == [-5.0, 0.0, 5.0]


def test_cxcywh_to_xyxy():
    assert _cxcywh_to_xyxy((10, 20, 4, 6)) == (8, 17, 12, 23)


# --- order confirmation (_said_no, biased to accept) -----------------------

@pytest.mark.parametrize("text", ["no", "Nope", "nah", "that's wrong", "no it isn't"])
def test_said_no_detects_explicit_rejection(text):
    assert _said_no(text) is True


@pytest.mark.parametrize("text", ["", "yes that's right", "correct, thanks", "sure"])
def test_said_no_accepts_silence_and_affirmation(text):
    assert _said_no(text) is False
