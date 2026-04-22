from client import WalkieAIClient
import cv2
import numpy as np
from PIL import Image
import time

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 6),                                 # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 11), (6, 12), (11, 12),            # torso
    (11, 13), (13, 15), (12, 14), (14, 16) # legs
]

KEYPOINT_COLOR = (0, 255, 0)
SKELETON_COLOR = (255, 180, 0)
BBOX_COLOR = (0, 0, 255)
KP_CONFIDENCE_THRESHOLD = 0.3


def draw_poses(frame, poses):
    for person in poses:
        cx, cy, w, h = person.bbox
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        cv2.putText(
            frame,
            f"person {person.confidence:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLOR, 1,
        )

        kps = person.keypoints
        for kp in kps:
            if kp.confidence < KP_CONFIDENCE_THRESHOLD:
                continue
            cv2.circle(frame, (int(kp.x), int(kp.y)), 4, KEYPOINT_COLOR, -1)

        for i, j in SKELETON:
            if i >= len(kps) or j >= len(kps):
                continue
            a, b = kps[i], kps[j]
            if a.confidence < KP_CONFIDENCE_THRESHOLD or b.confidence < KP_CONFIDENCE_THRESHOLD:
                continue
            cv2.line(frame, (int(a.x), int(a.y)), (int(b.x), int(b.y)), SKELETON_COLOR, 2)


def test_pose_estimation():
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
        poses = walkieAI.pose_estimation.estimate(frame)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds. (FPS: {1 / (end_time - start_time)})")
        draw_poses(frame, poses)

        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_pose_estimation()
