"""Concrete ``Embedder`` implementations.

This file adapts the HTTP-backed CLIP service (``client.ImageEmbedClient``)
to the :class:`perception.types.Embedder` Protocol so the perception loop
can use it as a drop-in.

Why a separate adapter (rather than making ``ImageEmbedClient`` itself
satisfy the protocol): the HTTP client is a transport-level concern that
shouldn't know about ``model_name`` semantics; the Embedder Protocol is a
domain-level concern that shouldn't know about HTTP. Separating them keeps
the test layers crisp — the HTTP smoke test mocks the network, the
Embedder test mocks the client.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from client.image_embed import ImageEmbedClient


class RemoteCLIPEmbedder:
    """``Embedder`` Protocol implementation backed by ``/image-embed/*``.

    The walkie-ai-server's CLIP provider is ``openai/clip-vit-base-patch16``,
    so we hardcode that as ``model_name``. If the server provider ever
    changes, update this string — it lands in every record's
    ``embedding_model`` metadata field, used for forward-compat invalidation.
    """

    DEFAULT_MODEL_NAME = "clip-vit-base-patch16"

    def __init__(
        self,
        client: ImageEmbedClient,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._dim: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self._client.get_embedding_dim()
        return self._dim

    def embed_image(self, image: Image.Image) -> list[float]:
        return self._client.embed_image(image)

    def embed_text(self, text: str) -> list[float]:
        return self._client.embed_text(text)


class LocalCLIPEmbedder:
    """In-process CLIP — embeddings with **no walkie-ai-server dependency**.

    Loads the *same* checkpoint the server uses (``openai/clip-vit-base-patch16``
    by default) via HuggingFace ``transformers``, so its vectors live in the
    same space as anything the :class:`RemoteCLIPEmbedder` already stored —
    switching backends does not invalidate the catalogue (the recorded
    ``model_name`` matches too). Both towers run locally: ``embed_image`` (image
    tower) and ``embed_text`` (text tower), each L2-normalized to satisfy the
    :class:`~perception.types.Embedder` protocol.

    ``torch`` + ``transformers`` are an optional extra — install with
    ``uv sync --extra clip``. They're imported lazily (on first embed), so the
    rest of the app, the tests, and the dev machine don't need them unless this
    backend is actually selected (``SCENE_EMBED_BACKEND=local``).
    """

    def __init__(
        self,
        *,
        model_name: str = "openai/clip-vit-base-patch16",
        device: Optional[str] = None,
        fp16: Optional[bool] = None,
    ) -> None:
        self._hf_id = model_name
        # Short name recorded in each record's metadata. Kept identical to
        # RemoteCLIPEmbedder's so the two backends are interchangeable over the
        # same store.
        self._model_name = model_name.split("/")[-1]
        self._device = device
        # None → decide at load time (fp16 on CUDA, fp32 on CPU).
        self._fp16 = fp16
        self._model = None
        self._processor = None
        self._torch = None
        self._dtype = None
        self._dim: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:  # pragma: no cover — depends on optional extra
            raise RuntimeError(
                "LocalCLIPEmbedder needs torch + transformers. Install them with "
                "`uv sync --extra clip` (or set SCENE_EMBED_BACKEND=remote)."
            ) from e
        self._torch = torch
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        on_cuda = str(device).startswith("cuda")
        # Half precision on the GPU: CLIP ViT-B/16 is tiny, so on a modern card
        # this is essentially free accuracy-wise and noticeably faster + lighter
        # on VRAM. Default on for CUDA, off for CPU (CPU fp16 is slow).
        use_fp16 = self._fp16 if self._fp16 is not None else on_cuda
        self._dtype = torch.float16 if (use_fp16 and on_cuda) else torch.float32
        self._device = device
        self._model = (
            CLIPModel.from_pretrained(self._hf_id, torch_dtype=self._dtype)
            .to(device)
            .eval()
        )
        self._processor = CLIPProcessor.from_pretrained(self._hf_id)
        self._dim = int(self._model.config.projection_dim)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def _prep(self, inputs):
        """Move processor outputs to the model's device + dtype.

        Only floating tensors (e.g. ``pixel_values``) are cast to the model
        dtype; integer tensors (``input_ids``, ``attention_mask``) stay intact.
        """
        out = {}
        for k, v in inputs.items():
            if hasattr(v, "is_floating_point") and v.is_floating_point():
                out[k] = v.to(self._device, self._dtype)
            else:
                out[k] = v.to(self._device)
        return out

    def embed_image(self, image: Image.Image) -> list[float]:
        self._ensure_loaded()
        inputs = self._prep(self._processor(images=image, return_tensors="pt"))
        with self._torch.inference_mode():
            feats = self._model.get_image_features(**inputs)
        return self._normalize(feats)

    def embed_text(self, text: str) -> list[float]:
        self._ensure_loaded()
        inputs = self._prep(
            self._processor(
                text=[text], return_tensors="pt", padding=True, truncation=True
            )
        )
        with self._torch.inference_mode():
            feats = self._model.get_text_features(**inputs)
        return self._normalize(feats)

    def _normalize(self, feats) -> list[float]:
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        # float() so a half-precision tensor serializes to clean Python floats.
        return feats[0].float().detach().cpu().tolist()
