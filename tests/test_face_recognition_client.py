"""Unit tests for the FaceRecognitionClient — mocks the HTTP transport.

Verifies the client deserializes the walkie-ai-server /face-recognition/*
contract correctly, without a running server.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from client import FaceEmbedding, FaceRecognitionClient
from client.base import WalkieAPIError


def _resp(payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


@pytest.fixture
def img():
    return Image.new("RGB", (8, 8), (10, 20, 30))


def test_embed_deserializes_faces(img):
    data = [
        {"bbox_xyxy": [10, 20, 110, 220], "embedding": [0.6, 0.8], "det_score": 0.97},
        {"bbox_xyxy": [0, 0, 5, 5], "embedding": [1.0, 0.0], "det_score": 0.41},
    ]
    client = FaceRecognitionClient()
    with patch.object(client._session, "post", return_value=_resp({"success": True, "data": data})):
        faces = client.embed(img)
    assert len(faces) == 2
    assert isinstance(faces[0], FaceEmbedding)
    assert faces[0].bbox_xyxy == (10, 20, 110, 220)
    assert faces[0].embedding == [0.6, 0.8]
    assert faces[0].det_score == pytest.approx(0.97)
    # largest-face selection (used by enroll) picks the up-front person
    assert max(faces, key=lambda f: f.area()) is faces[0]


def test_embed_empty_when_no_face(img):
    client = FaceRecognitionClient()
    with patch.object(client._session, "post", return_value=_resp({"success": True, "data": []})):
        assert client.embed(img) == []


def test_embed_raises_on_server_error(img):
    client = FaceRecognitionClient()
    with patch.object(
        client._session, "post", return_value=_resp({"success": False, "error": "bad image"})
    ):
        with pytest.raises(WalkieAPIError, match="bad image"):
            client.embed(img)


def test_info_is_cached():
    client = FaceRecognitionClient()
    get = MagicMock(return_value=_resp({"success": True, "data": {"model_name": "insightface-buffalo_l", "dim": 512}}))
    with patch.object(client._session, "get", get):
        assert client.get_embedding_dim() == 512
        assert client.get_model_name() == "insightface-buffalo_l"
    # two reads, one HTTP call (cached)
    assert get.call_count == 1


def test_face_embedding_area():
    f = FaceEmbedding(bbox_xyxy=(0, 0, 10, 20), embedding=[], det_score=0.5)
    assert f.area() == 200
    assert FaceEmbedding(bbox_xyxy=(10, 10, 0, 0)).area() == 0  # degenerate → clamped
