"""End-to-end v2 pipeline: synthetic snapshots -> build_scene -> SceneStore -> query.

Exercises the real lift (deproject_mask) + batch association + store merge/query without
the camera, server, or Open3D — by using baseline poses (cam_R=I), no TSDF, and masks
the same resolution as depth (so deproject_mask never needs cv2). This is the seam the
robot will run; the camera/server/Open3D edges are covered separately on-robot.
"""

from __future__ import annotations

import numpy as np

from services.walkie_graphs.builder import build_scene
from services.walkie_graphs.buffer import Detection, Snapshot, SnapshotBuffer
from services.walkie_graphs.relations import derive_relations
from services.walkie_graphs.scene import SceneStore

H = W = 64
FX = FY = 500.0
CX = CY = 32.0
INTR = (FX, FY, CX, CY, W, H)


def _object_frame(ts, *, region, depth0, clip_class, caption, label):
    """One synthetic snapshot with a single planar-ish object in `region` (y0,y1,x0,x1)."""
    y0, y1, x0, x1 = region
    depth = np.zeros((H, W), dtype=np.float32)  # 0 == invalid elsewhere
    mask = np.zeros((H, W), dtype=np.uint8)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    # a depth ramp across the patch so voxelization keeps a real 3D cloud (not one plane)
    depth[y0:y1, x0:x1] = depth0 + 0.003 * (xx - x0) + 0.003 * (yy - y0)
    mask[y0:y1, x0:x1] = 1
    emb = [0.0] * 8
    emb[clip_class] = 1.0
    det = Detection(
        class_name=label, class_id=clip_class, conf=0.9,
        bbox=(x0, y0, x1, y1), caption=caption, clip_emb=emb, mask=mask,
    )
    return Snapshot(
        ts=ts, depth=depth, intr=INTR,
        cam_R=np.eye(3), cam_t=np.zeros(3), robot_pose={"x": 0, "y": 0, "heading": 0},
        detections=[det],
    )


def _build_kwargs():
    return dict(pose_mode="baseline", do_tsdf=False, voxel_m=0.01, min_points=5,
               erode_px=0, edge_thresh=0.0, sor_k=0, log=lambda *_a: None)


def test_same_object_three_frames_fuses_to_one():
    snaps = [
        _object_frame(t, region=(20, 44, 20, 44), depth0=1.0, clip_class=0,
                      caption="a red coke can", label="coke")
        for t in (1.0, 2.0, 3.0)
    ]
    res = build_scene(snaps, **_build_kwargs())
    assert len(res.observations) == 1
    obs = res.observations[0]
    assert obs.n_obs == 3
    assert obs.class_name == "coke"
    # centroid is in front of the camera (+Z), near the patch depth
    assert 0.9 < obs.centroid[2] < 1.3


def test_two_distinct_objects_stay_separate():
    snaps = []
    for t in (1.0, 2.0):
        snaps.append(_object_frame(t, region=(8, 28, 8, 28), depth0=1.0, clip_class=0,
                                    caption="a coke", label="coke"))
        snaps.append(_object_frame(t + 0.1, region=(40, 60, 40, 60), depth0=2.0, clip_class=1,
                                    caption="a chair", label="chair"))
    res = build_scene(snaps, **_build_kwargs())
    labels = sorted(o.class_name for o in res.observations)
    assert labels == ["chair", "coke"], labels
    assert all(o.n_obs == 2 for o in res.observations)


def test_pipeline_into_store_and_query():
    snaps = [
        _object_frame(t, region=(20, 44, 20, 44), depth0=1.0, clip_class=0,
                      caption="a red coke can", label="coke")
        for t in (1.0, 2.0)
    ]
    res = build_scene(snaps, **_build_kwargs())
    store = SceneStore(min_obs_confirm=2, require_confirmation=True)  # no embed -> keyword
    nodes = store.merge(res.observations, now=10.0)
    store.install(nodes, derive_relations(nodes))
    # confirmed (n_obs==2) and findable by keyword
    hits = store.query_text("coke", k=5)
    assert len(hits) == 1
    assert hits[0].class_name == "coke"
    assert hits[0].centroid[:2]  # navigable XY exists
    # a second build of the SAME object must MERGE (not duplicate) and bump n_obs
    res2 = build_scene(snaps, **_build_kwargs())
    nodes2 = store.merge(res2.observations, now=20.0)
    store.install(nodes2, derive_relations(nodes2))
    assert store.count() == 1
    assert store.get(nodes2[0].id).n_obs == 4


def test_through_snapshot_buffer(tmp_path):
    buf = SnapshotBuffer(tmp_path / "buf", cap=10)
    for t in (1.0, 2.0, 3.0):
        buf.append(_object_frame(t, region=(20, 44, 20, 44), depth0=1.0, clip_class=0,
                                 caption="a coke", label="coke"))
    snaps = buf.load_window(None)
    assert len(snaps) == 3
    res = build_scene(snaps, **_build_kwargs())
    assert len(res.observations) == 1
    assert res.observations[0].n_obs == 3
