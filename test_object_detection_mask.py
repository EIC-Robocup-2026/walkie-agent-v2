from client import WalkieAIClient
import cv2
import numpy as np
import time


# Stable per-class colors so the same class keeps the same overlay color
# across frames (BGR). New classes get a deterministic random color.
_class_colors: dict[str, tuple[int, int, int]] = {}


def _color_for(class_name: str | None) -> tuple[int, int, int]:
    key = class_name or "object"
    if key not in _class_colors:
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        _class_colors[key] = tuple(int(c) for c in rng.integers(64, 256, size=3))
    return _class_colors[key]


def test_object_detection_mask():
    walkieAI = WalkieAIClient(base_url="http://localhost:5000")
    print("Walkie Client initialized")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        start_time = time.time()
        # 2. Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not captured correctly, ret will be False
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Request a segmentation mask per detection.
        detections = walkieAI.object_detection.detect(frame, prompts=["chair", "table", "stool", "shirt"], return_mask=True)

        print(f"Frame shape: {frame.shape}, Detections: {len(detections)}")

        # Blend all masks onto a copy, then draw bboxes/labels on top.
        overlay = frame.copy()
        for detection in detections:
            color = _color_for(detection.class_name)

            mask = detection.mask
            if mask is not None:
                # Mask is a 2D uint8 {0,1} array; resize to the frame if the
                # server returned it at a different resolution.
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(
                        mask, (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                sel = mask.astype(bool)
                # Tint the masked pixels with the class color.
                overlay[sel] = (0.5 * overlay[sel] + 0.5 * np.array(color)).astype(np.uint8)

        # Composite the tinted overlay back with the original frame.
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw bounding boxes and class-name labels on top of the masks.
        for detection in detections:
            color = _color_for(detection.class_name)
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = detection.class_name or "object"
            if detection.confidence is not None:
                label = f"{label} {detection.confidence:.2f}"

            # Filled label background for readability.
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            ly = max(y1, th + 4)
            cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw, ly + baseline - 2), color, -1)
            cv2.putText(
                frame, label, (x1, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        cv2.imshow("Object Detection (mask)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds. (FPS: {1 / (end_time - start_time)})")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_object_detection_mask()
