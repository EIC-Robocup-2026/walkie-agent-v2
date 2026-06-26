from client import WalkieAIClient
import cv2
import time
import os
from dotenv import load_dotenv

from walkie_sdk import WalkieRobot
from interfaces.walkie_interface import WalkieInterface

ZENOH_PORT = 7447
ROBOT_IP = "127.0.0.1"


def get_robot() -> WalkieRobot:
    return WalkieRobot(
        ip=ROBOT_IP,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )

def test_object_detection():
    load_dotenv()
    robot = get_robot()
    walkie = WalkieInterface(robot)
    walkieAI = WalkieAIClient(
        base_url=os.getenv("WALKIE_AI_BASE_URL", "http://10.0.0.202:5000"),
    )
    print("Walkie Client initialized")

    # Over SSH there's usually no X display, so cv2.imshow can't open a window
    # (it fails with "could not connect to display" / Qt xcb plugin errors).
    # Fall back to writing the annotated frame to a file the user can scp/view.
    # Force headless with HEADLESS=1.
    headless = os.getenv("HEADLESS") == "1" or not os.environ.get("DISPLAY")
    out_path = os.path.join(os.path.dirname(__file__), "object_detection_output.jpg")
    if headless:
        print(f"No display detected — writing annotated frames to {out_path} (Ctrl+C to stop)")

    while True:
        start_time = time.time()

        # capture() returns a BGR numpy array, which both detect() and the
        # cv2 draw/show calls below accept (capture_pil() returns a PIL Image
        # that cv2.imshow/rectangle/putText can't handle).
        frame = walkie.camera.capture()

        detections = walkieAI.image.detect(frame, prompts=["toothpaste"])
        # Show the detections
        for detection in detections:
            cv2.rectangle(frame, (detection.bbox[0], detection.bbox[1]), (detection.bbox[2], detection.bbox[3]), (0, 0, 255), 2)
            label = detection.class_name or "object"
            cv2.putText(frame, label, (detection.bbox[0], detection.bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if headless:
            print(f"Detections: {len(detections)} -> {[d.class_name for d in detections]}")
            cv2.imwrite(out_path, frame)
        else:
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds. (FPS: {1 / (end_time - start_time)})")
        time.sleep(1)

    if not headless:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_object_detection()
