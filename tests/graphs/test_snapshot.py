"""Unit tests for services.walkie_graphs.snapshot — the live perception.json builder.

No robot/server: detections are plain stubs and the ingest result is a hand-built dict.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from services.realtime_explore.snapshot import build_object_records, write_atomic


def _det(class_name, bbox, conf=0.9):
    return SimpleNamespace(class_name=class_name, bbox=bbox, confidence=conf)


def test_build_object_records_maps_centroid_and_caption():
    dets = [_det("cup", (10, 20, 30, 40), conf=0.8)]
    result = {0: {"centroid": (1.0, 2.0, 3.0), "caption": "a red cup"}}
    recs = build_object_records(dets, result, image_size=(640, 480), robot_heading=0.0)
    assert len(recs) == 1
    r = recs[0]
    assert r["class"] == "cup"
    assert r["bbox"] == [10, 20, 30, 40]
    assert r["conf"] == 0.8
    assert r["position_3d"] == [1.0, 2.0, 3.0]
    assert r["caption"] == "a red cup"
    # in-frame placement is filled for every detection
    assert r["frame_h"] in {"left", "slightly left", "center", "slightly right", "right"}
    assert r["frame_v"] in {"top", "slightly top", "center", "slightly bottom", "bottom"}
    assert isinstance(r["heading"], float)


def test_build_object_records_position_none_when_no_centroid():
    dets = [_det("table", (0, 0, 100, 100))]
    # ingest produced no 3D for this detection (sparse / no depth)
    result = {0: {"centroid": None, "caption": ""}}
    recs = build_object_records(dets, result, image_size=(640, 480), robot_heading=0.0)
    assert recs[0]["position_3d"] is None
    assert recs[0]["caption"] == ""


def test_build_object_records_missing_index_degrades_gracefully():
    # ingest_result has no entry for index 0 (e.g. ingest aborted) → unknown pos, empty caption
    dets = [_det("bowl", (5, 5, 15, 15))]
    recs = build_object_records(dets, {}, image_size=(320, 240), robot_heading=1.0)
    assert recs[0]["position_3d"] is None
    assert recs[0]["caption"] == ""
    assert recs[0]["class"] == "bowl"


def test_frame_position_center_object_is_center():
    dets = [_det("cup", (300, 220, 340, 260))]  # centred in a 640x480 frame
    recs = build_object_records(dets, {}, image_size=(640, 480), robot_heading=0.0)
    assert recs[0]["frame_h"] == "center"
    assert recs[0]["frame_v"] == "center"


def test_write_atomic_writes_valid_json_no_tmp_left(tmp_path):
    out = tmp_path / "perception.json"
    snap = {"ts": 123.0, "objects": [{"class": "cup", "position_3d": [1.0, 2.0, 3.0]}]}
    write_atomic(out, snap)
    assert json.loads(out.read_text()) == snap
    # the .tmp scratch file is renamed away, not left behind
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_write_atomic_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "dir" / "perception.json"
    write_atomic(out, {"ts": 1.0, "objects": []})
    assert out.exists()
    assert json.loads(out.read_text())["objects"] == []
