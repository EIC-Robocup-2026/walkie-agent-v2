"""Unit tests for the AppearanceClient — mocks the HTTP transport.

Verifies the client deserializes the walkie-ai-server /appearance/* contract
(pipeline design by Chalk, EIC team) correctly, without a running server.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from client import AppearanceClient
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


def test_embed_deserializes_embedding(img):
    client = AppearanceClient()
    with patch.object(
        client._session,
        "post",
        return_value=_resp({"success": True, "data": {"embedding": [0.6, 0.8]}}),
    ):
        emb = client.embed(img)
    assert emb == [0.6, 0.8]
    assert all(isinstance(x, float) for x in emb)


def test_embed_raises_on_server_error(img):
    client = AppearanceClient()
    with patch.object(
        client._session, "post", return_value=_resp({"success": False, "error": "bad image"})
    ):
        with pytest.raises(WalkieAPIError, match="bad image"):
            client.embed(img)


def test_info_is_cached():
    client = AppearanceClient()
    get = MagicMock(
        return_value=_resp({"success": True, "data": {"model_name": "osnet_x1_0", "dim": 512}})
    )
    with patch.object(client._session, "get", get):
        assert client.get_embedding_dim() == 512
        assert client.get_model_name() == "osnet_x1_0"
    assert get.call_count == 1
