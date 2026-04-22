"""HTTP client for the Image Caption API — /image-caption/*"""

from __future__ import annotations

from PIL import Image
import numpy as np
from .base import WalkieBaseClient, _numpy_to_bytes, _pil_to_bytes


class ImageCaptionClient(WalkieBaseClient):
    """Client for the ``/image-caption`` blueprint.

    Mirrors the interface of :class:`services.image_caption.ImageCaption`.

    Example::

        client = ImageCaptionClient()

        # Single image
        caption = client.caption(pil_image)
        caption = client.caption(pil_image, prompt="What color is the sky?")

        # Batch
        captions = client.caption_batch([img_a, img_b])
        captions = client.caption_batch(
            [img_a, img_b],
            prompts=["Describe A.", "Describe B."],
        )
    """

    def caption(self, image: Image.Image | np.ndarray, prompt: str | None = None) -> str:
        """Generate a caption for a single *image*.

        Args:
            image: A PIL Image (RGB) or numpy array (BGR).
            prompt: Optional prompt to guide the captioning model.

        Returns:
            Caption string.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        if isinstance(image, np.ndarray):
            files = {"image": ("image.jpg", _numpy_to_bytes(image, fmt="JPEG"), "image/jpeg")}
        else:
            files = {"image": ("image.png", _pil_to_bytes(image), "image/png")}
        form_data = {"prompt": prompt} if prompt is not None else {}
        data = self._post_files("/image-caption/caption", files=files, data=form_data)
        return data["caption"]

    def caption_batch(
        self,
        images: list[Image.Image | np.ndarray],
        prompts: list[str] | None = None,
    ) -> list[str]:
        """Generate captions for multiple images.

        Args:
            images: List of PIL Images (RGB) or numpy arrays (BGR).
            prompts: Optional list of prompts, one per image.  Must be the
                same length as *images* if provided.

        Returns:
            List of caption strings in the same order as *images*.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        files = []
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                files.append(("images", (f"image_{i}.jpg", _numpy_to_bytes(img, fmt="JPEG"), "image/jpeg")))
            else:
                files.append(("images", (f"image_{i}.png", _pil_to_bytes(img), "image/png")))
        form_data = (
            [("prompts", p) for p in prompts]
            if prompts is not None
            else []
        )
        data = self._post_files("/image-caption/caption-batch", files=files, data=form_data)
        return data["captions"]

    def available_providers(self) -> list[str]:
        """List all providers registered on the server."""
        return self._get("/image-caption/providers")
