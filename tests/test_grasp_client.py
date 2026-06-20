"""Unit tests for GraspClient.infer() — no server.

A stub overrides ``_post_files`` to capture the multipart payload the real client
would send and to return a canned response in the ``/grasp`` envelope. The tests
assert the client (a) ships the cloud losslessly as ``.npy`` bytes, (b) puts the
options in the JSON ``spec``, and (c) parses the response into ``GraspPose`` objects.
"""

import io
import json

import numpy as np
import pytest

from client.grasp import GraspClient, GraspPose


class _StubClient(GraspClient):
    """GraspClient whose HTTP POST is replaced by a canned response."""

    def __init__(self, grasps):
        super().__init__(base_url="http://stub")
        self._grasps = grasps
        self.sent_cloud: np.ndarray | None = None
        self.sent_form: dict | None = None

    def _post_files(self, path, files, data):  # type: ignore[override]
        self.sent_cloud = np.load(io.BytesIO(files["cloud"][1]), allow_pickle=False)
        self.sent_form = dict(data)
        return {"grasps": self._grasps, "count": len(self._grasps)}


def test_infer_round_trips_cloud_and_parses_grasps():
    grasps = [
        {
            "translation": [0.1, 0.2, 0.4],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "width": 0.05,
            "score": 0.9,
            "antipodal_score": 0.7,
        }
    ]
    client = _StubClient(grasps)
    cloud = np.random.rand(2000, 3).astype(np.float32)

    out = client.infer(cloud, max_grasps=5, antipodal=True, score_threshold=0.1)

    # cloud round-trips losslessly as .npy
    assert client.sent_cloud is not None
    assert client.sent_cloud.shape == (2000, 3)
    assert np.allclose(client.sent_cloud, cloud)

    # options carried in the spec
    spec = json.loads(client.sent_form["spec"])
    assert spec["max_grasps"] == 5
    assert spec["antipodal"] is True
    assert spec["score_threshold"] == 0.1

    # response parses into a GraspPose
    assert len(out) == 1
    g = out[0]
    assert isinstance(g, GraspPose)
    assert g.translation == (0.1, 0.2, 0.4)
    assert g.rotation.shape == (3, 3)
    assert g.width == 0.05 and g.score == 0.9 and g.antipodal_score == 0.7
    assert np.allclose(g.approach, [1, 0, 0])   # rotation col-0
    assert np.allclose(g.closing, [0, 1, 0])    # rotation col-1


def test_infer_defaults_omit_optional_overrides():
    client = _StubClient([])
    client.infer(np.random.rand(500, 3).astype(np.float32))
    spec = json.loads(client.sent_form["spec"])
    # voxel_size / num_point only appear when explicitly set
    assert "voxel_size" not in spec
    assert "num_point" not in spec
    assert spec["antipodal"] is False


def test_infer_empty_grasps_returns_empty_list():
    client = _StubClient([])
    out = client.infer(np.random.rand(500, 3).astype(np.float32))
    assert out == []


def test_infer_rejects_bad_shape():
    client = _StubClient([])
    with pytest.raises(ValueError):
        client.infer(np.zeros((10, 2), dtype=np.float32))
