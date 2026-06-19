from client import WalkieAIClient
import cv2
import time

def test_captioning():
    walkieAI = WalkieAIClient()
    print("Walkie Client initialized")

    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    ret, frame = cap.read()

    # If the frame was not captured correctly, ret will be False
    if not ret:
        print("Error: Can't receive frame. Exiting...")

    start_time = time.time()
    result = walkieAI.image.caption(frame)
    end_time = time.time()
    print(result)
    print(f"Time taken: {end_time - start_time} seconds")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    test_captioning()