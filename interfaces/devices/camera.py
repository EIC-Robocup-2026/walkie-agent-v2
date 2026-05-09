"""Camera input for capturing images and video frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from walkie_sdk.robot import WalkieRobot


class Camera:
    """Camera interface for capturing images.

    Supports two mutually-exclusive sources:
      - **Robot mode**: frames come from a ``WalkieRobot`` instance.
      - **Local mode**: frames come from a local webcam via ``cv2.VideoCapture``.

    Provides methods for capturing single frames and returning images
    in various formats (numpy array, PIL Image, bytes).
    """

    def __init__(
        self,
        robot: "WalkieRobot | None" = None,
        device: int | None = None,
    ) -> None:
        """Initialize camera.

        Exactly one of *robot* or *device* must be provided.

        Args:
            robot: WalkieRobot instance for robot camera access.
            device: Local webcam device index (e.g. 0 for the default laptop camera).

        Raises:
            ValueError: If both or neither source is provided.
        """
        if robot is not None and device is not None:
            raise ValueError("Provide either 'robot' or 'device', not both.")
        if robot is None and device is None:
            raise ValueError("Provide either 'robot' or 'device'.")

        self._bot = robot
        self._device = device
        self._cap: cv2.VideoCapture | None = None  # only used in local mode

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera source.

        - Local mode: opens the ``cv2.VideoCapture`` for the configured device.
        - Robot mode: no-op (the robot manages its own camera lifecycle).

        Raises:
            RuntimeError: If the local camera cannot be opened.
        """
        if self._device is not None:
            self._cap = cv2.VideoCapture(self._device)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Failed to open local camera (device={self._device})."
                )

    def close(self) -> None:
        """Release the camera source.

        - Local mode: releases the ``cv2.VideoCapture``.
        - Robot mode: no-op.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Capture helpers
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Capture a single frame from the camera.

        Returns:
            Frame as numpy array in BGR format.

        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        if self._bot is not None:
            frame = self._bot.camera.get_frame()
            if frame is None:
                raise RuntimeError("Failed to get frame from robot camera.")
        else:
            if self._cap is None or not self._cap.isOpened():
                raise RuntimeError("Local camera is not open. Call open() first.")
            ret, frame = self._cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read frame from local camera.")
        return frame

    def capture_rgb(self) -> np.ndarray:
        """Capture a single frame in RGB format.

        Returns:
            Frame as numpy array in RGB format.

        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        frame = self.capture()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def capture_pil(self) -> Image.Image:
        """Capture a single frame as PIL Image.

        Returns:
            Frame as PIL Image in RGB format.

        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        frame_rgb = self.capture_rgb()
        return Image.fromarray(frame_rgb)

    # ------------------------------------------------------------------
    # Context manager / destructor
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        """Context manager entry - opens the camera."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes the camera."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures camera connection is released."""
        self.close()
