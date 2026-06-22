"""Global, prompt-free perception / geometry / navigation / people primitives.

Themed submodules (geometry, lift, seating, navigation, people) hold the plain
functions split out of the per-task skills.py files; this package re-exports
them all flat so callers do `from tasks.skills import <name>`.
"""

from __future__ import annotations

from .geometry import (
    BBox,
    cxcywh_to_xyxy,
    overlap_fraction,
    parse_pose,
    person_hip_anchor,
    person_seat_anchor,
)
from .listening import CommandListener
from .lift import (
    bboxes_world_position,
    lift_bbox_world_xy,
    recall_person_xy,
    remember_located_positions,
    remember_person_xy,
    reset_people_positions,
)
from .seating import (
    SeatCandidate,
    SeatPart,
    find_persons,
    find_seated_person_bbox,
    is_sofa_class,
    match_people_to_seats,
    parse_sofa_parts,
    pick_free_seat,
    resolve_free_part,
    scan_seats,
    split_seat_regions,
)
from .door import door_open_from_depth, go_to_through_door, is_door_open, request_open_door
from .grasp import (
    GraspCandidate,
    align_arm_to_object,
    approach_object,
    execute_grasp,
    face_object,
    face_object_with_arm,
    get_object_grasp_pos,
    in_arm_deadzone,
    look_at_object,
    pick_object,
)
from .held import (
    HeldObject,
    clear_held_object,
    held_arms,
    recall_held_object,
    record_held_object,
)
from .place import (
    detect_surfaces,
    execute_place,
    place_object,
)
from interfaces.perception.surfaces import (
    SurfacePlane,
    assign_objects_to_surfaces,
    support_surface_for,
)
from .navigation import (
    MotionPredictor,
    approach_point,
    face_point,
    follow_person,
    heading_to_point,
    move_base_relative,
    rotate_by,
    side_relative_to_listener,
    sweep_snapshots,
    tilt_head,
)
from .people import (
    FaceTracker,
    biggest_face,
    is_calling_gesture,
    nearest_person_bbox,
    person_bboxes,
    select_largest_person,
    wait_for_person,
)

__all__ = [
    "CommandListener",
    "request_open_door",
    "go_to_through_door",
    "is_door_open",
    "door_open_from_depth",
    "GraspCandidate",
    "get_object_grasp_pos",
    "look_at_object",
    "approach_object",
    "in_arm_deadzone",
    "face_object",
    "face_object_with_arm",
    "align_arm_to_object",
    "execute_grasp",
    "pick_object",
    "HeldObject",
    "record_held_object",
    "recall_held_object",
    "held_arms",
    "clear_held_object",
    "SurfacePlane",
    "support_surface_for",
    "assign_objects_to_surfaces",
    "detect_surfaces",
    "execute_place",
    "place_object",
    "BBox",
    "cxcywh_to_xyxy",
    "overlap_fraction",
    "parse_pose",
    "person_hip_anchor",
    "person_seat_anchor",
    "bboxes_world_position",
    "lift_bbox_world_xy",
    "recall_person_xy",
    "remember_located_positions",
    "remember_person_xy",
    "reset_people_positions",
    "SeatCandidate",
    "SeatPart",
    "find_persons",
    "find_seated_person_bbox",
    "is_sofa_class",
    "match_people_to_seats",
    "parse_sofa_parts",
    "pick_free_seat",
    "resolve_free_part",
    "scan_seats",
    "split_seat_regions",
    "MotionPredictor",
    "approach_point",
    "face_point",
    "follow_person",
    "heading_to_point",
    "move_base_relative",
    "rotate_by",
    "side_relative_to_listener",
    "sweep_snapshots",
    "tilt_head",
    "FaceTracker",
    "biggest_face",
    "is_calling_gesture",
    "nearest_person_bbox",
    "person_bboxes",
    "select_largest_person",
    "wait_for_person",
]
