"""Pure unit tests for the HRI seat-occupancy / sofa-cushion parsing.

No robot, no AI server — synthetic person boxes + COCO hip keypoints fed to the
pure geometry helpers in tasks/HRI/skills.py. These cover the two bugs the
cushion parser fixes: a person on part of a sofa wrongly hiding its free
cushions, and a neighbour's box clipping a free seat wrongly marking it taken.
"""

from client import PersonPose, PoseKeypoint
from tasks.HRI.skills import (
    SeatCandidate,
    find_seated_person_bbox,
    is_sofa_class,
    parse_sofa_parts,
    person_hip_anchor,
    person_seat_anchor,
    resolve_free_part,
    split_seat_regions,
)

_LEFT_HIP, _RIGHT_HIP = 11, 12


def _hips(x, y, conf=0.9):
    return [
        PoseKeypoint(x=x, y=y, confidence=conf, name="left_hip", index=_LEFT_HIP),
        PoseKeypoint(x=x, y=y, confidence=conf, name="right_hip", index=_RIGHT_HIP),
    ]


def _person(cx, cy, w, h, keypoints=None):
    return PersonPose(bbox=(cx, cy, w, h), confidence=0.9, keypoints=keypoints or [])


# --- anchors -----------------------------------------------------------------


def test_hip_anchor_is_midpoint():
    p = _person(0, 0, 10, 10, _hips(40, 60))
    assert person_hip_anchor(p) == (40, 60)


def test_hip_anchor_none_when_low_confidence():
    p = _person(0, 0, 10, 10, _hips(40, 60, conf=0.1))
    assert person_hip_anchor(p) is None


def test_seat_anchor_falls_back_to_bbox_lower_center():
    p = _person(100, 50, 40, 80)  # no keypoints
    assert person_hip_anchor(p) is None
    # lower-centre: (cx, cy + 0.25*h)
    assert person_seat_anchor(p) == (100, 50 + 20)


# --- region split ------------------------------------------------------------


def test_split_three_cushions_left_to_right():
    regions = split_seat_regions((0, 0, 300, 100), has_middle=True)
    assert [label for label, _ in regions] == ["LEFT", "MIDDLE", "RIGHT"]
    assert [box for _, box in regions] == [
        (0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100)
    ]


def test_split_two_cushions_flag():
    regions = split_seat_regions((0, 0, 200, 100), has_middle=False)
    assert [label for label, _ in regions] == ["LEFT", "RIGHT"]
    assert [box for _, box in regions] == [(0, 0, 100, 100), (100, 0, 200, 100)]


# --- the two bugs ------------------------------------------------------------


def test_one_person_on_sofa_leaves_other_cushions_free():
    # Person sitting on the LEFT cushion, box sprawling into MIDDLE — only LEFT
    # should be taken, not the whole sofa.
    sofa = (0, 0, 300, 100)
    p = _person(60, 50, 120, 100, _hips(50, 50))  # box spans x 0..120
    parts = parse_sofa_parts(sofa, [p], has_middle=True)
    taken = {p.label: p.occupied for p in parts}
    assert taken == {"LEFT": True, "MIDDLE": False, "RIGHT": False}


def test_neighbor_box_clipping_free_seat_does_not_occupy_it():
    # A person on a chair at x~150 whose box edges right to x=220, clipping a
    # FREE chair at 200..300. Their hip anchor (150) is on their own chair, so
    # the free chair must stay free.
    own_chair = SeatCandidate((100, 0, 200, 100), "chair", 0.9, (150, 50), False)
    free_chair = (200, 0, 300, 100)
    person = _person(160, 50, 120, 100, _hips(150, 50))  # box 100..220

    # The free chair, parsed as a single region, is not occupied by the neighbour.
    parts = parse_sofa_parts(free_chair, [person], has_middle=False)
    # (single-region check via the same machinery: split into 2 just confirms the
    #  neighbour's anchor lands in neither half)
    assert all(not pt.occupied for pt in parts)
    # ...and they ARE found as seated on their own chair.
    assert find_seated_person_bbox([person], [own_chair]) is not None


def test_anchorless_person_uses_overlap_fallback():
    # No hip keypoints: a box covering most of the LEFT cushion still occupies it.
    sofa = (0, 0, 300, 100)
    big = _person(50, 50, 100, 100)  # box 0..100 == LEFT cushion, no keypoints
    parts = parse_sofa_parts(sofa, [big], has_middle=True, hard_overlap=0.5)
    assert {pt.label: pt.occupied for pt in parts} == {
        "LEFT": True, "MIDDLE": False, "RIGHT": False
    }


def test_anchorless_edge_clip_below_hard_overlap_is_free():
    sofa = (0, 0, 300, 100)
    # Box 80..120 straddles LEFT/MIDDLE boundary: 20% of LEFT, 20% of MIDDLE —
    # both under the 0.5 hard floor, so with no hips nobody is seated.
    edge = _person(100, 50, 40, 100)
    parts = parse_sofa_parts(sofa, [edge], has_middle=True, hard_overlap=0.5)
    assert all(not pt.occupied for pt in parts)


# --- cushion selection -------------------------------------------------------


def _sofa_candidate(occupied_labels):
    parts = parse_sofa_parts((0, 0, 300, 100), [], has_middle=True)
    for pt in parts:
        pt.occupied = pt.label in occupied_labels
    occupied = all(pt.occupied for pt in parts)
    return SeatCandidate((0, 0, 300, 100), "sofa", 0.9, (150, 50), occupied, parts)


def test_resolve_free_part_honors_label_when_free():
    sofa = _sofa_candidate({"LEFT"})
    assert resolve_free_part(sofa, "RIGHT").label == "RIGHT"


def test_resolve_free_part_falls_back_when_label_taken():
    sofa = _sofa_candidate({"LEFT"})
    # Asked for LEFT (taken) -> first free cushion instead.
    assert resolve_free_part(sofa, "LEFT").label == "MIDDLE"


def test_resolve_free_part_none_for_plain_seat():
    chair = SeatCandidate((0, 0, 100, 100), "chair", 0.9, (50, 50), False, None)
    assert resolve_free_part(chair, None) is None


def test_resolve_free_part_none_when_all_taken():
    sofa = _sofa_candidate({"LEFT", "MIDDLE", "RIGHT"})
    assert resolve_free_part(sofa, None) is None


def test_is_sofa_class():
    assert is_sofa_class("sofa")
    assert is_sofa_class("Couch")
    assert not is_sofa_class("chair")
    assert not is_sofa_class(None)
