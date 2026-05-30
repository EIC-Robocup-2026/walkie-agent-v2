"""Unit tests for the embedder adapters.

The local CLIP backend's heavy deps (torch/transformers) are an optional extra,
so these tests cover only what's checkable without loading a model.
"""

from __future__ import annotations

import pytest

from perception import LocalCLIPEmbedder, RemoteCLIPEmbedder


def test_local_clip_records_same_short_model_name_as_remote():
    """Both backends must record the same model_name so they're interchangeable
    over one store (switching backends mustn't invalidate the catalogue)."""
    emb = LocalCLIPEmbedder(model_name="openai/clip-vit-base-patch16")
    assert emb.model_name == "clip-vit-base-patch16"
    assert emb.model_name == RemoteCLIPEmbedder.DEFAULT_MODEL_NAME


def test_local_clip_without_extra_raises_actionable_error():
    """When torch/transformers aren't installed, embedding fails with a message
    telling the user how to fix it — not an opaque ImportError."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch+transformers installed; the missing-extra path is moot")

    with pytest.raises(RuntimeError, match=r"uv sync --extra clip"):
        LocalCLIPEmbedder().embed_text("a coffee mug")
