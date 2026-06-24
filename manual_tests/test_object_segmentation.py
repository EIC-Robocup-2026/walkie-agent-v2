"""Manual webcam demo: open-vocab detect + segment on the local camera, draw masks.

Needs walkie-ai-server up at WALKIE_AI_BASE_URL and a webcam (cv2.VideoCapture(0)).
Run: uv run python -m manual_tests.test_object_segmentation
"""

from client import WalkieAIClient
import cv2
import numpy as np
import time

# Open-vocabulary prompts (noun phrases) for concept providers (SAM3 / YOLOE).
# Leave empty to let the provider use its default vocabulary (YOLO ignores this).
PROMPTS: list[str] | None = None

MASK_ALPHA = 0.45            # blend weight of the mask overlay
BBOX_COLOR = (0, 0, 255)     # BGR
TEXT_COLOR = (255, 255, 255)

# A fixed palette so each detection in a frame gets a distinct colour (BGR).
PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (128, 255, 0),
]


def draw_detections(frame, detections):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for i, det in enumerate(detections):
        color = PALETTE[i % len(PALETTE)]

        # Paint the segmentation mask (if the provider returned one).
        mask = det.mask
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[mask.astype(bool)] = color

        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = det.class_name or "object"
        if det.confidence is not None:
            label = f"{label} {det.confidence:.2f}"
        cv2.putText(
            frame, label, (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    # Blend the filled masks back over the frame (with bboxes/labels on top).
    cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
    for i, det in enumerate(detections):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def main():
    walkieAI = WalkieAIClient()
    print("Walkie Client initialized")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        start_time = time.time()
        detections = walkieAI.image.detect(
            frame, prompts=PROMPTS, return_mask=True
        )
        end_time = time.time()
        print(
            f"Time taken: {end_time - start_time} seconds. "
            f"(FPS: {1 / (end_time - start_time):.2f}, detections: {len(detections)})"
        )
        draw_detections(frame, detections)

        cv2.imshow("Object Segmentation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
