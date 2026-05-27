from client import WalkieAIClient
import cv2
import time
def test_object_detection():
    walkieAI = WalkieAIClient()
    print("Walkie Client initialized")

    cap = cv2.VideoCapture(0)

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

        detections = walkieAI.object_detection.detect(frame)
        # Show the detections
        for detection in detections:
            cv2.rectangle(frame, (detection.bbox[0], detection.bbox[1]), (detection.bbox[2], detection.bbox[3]), (0, 0, 255), 2)
            cv2.putText(frame, detection.class_name, (detection.bbox[0], detection.bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds. (FPS: {1 / (end_time - start_time)})")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_object_detection()