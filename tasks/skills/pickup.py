from interfaces.walkie_interface import WalkieInterface
from walkie_sdk import WalkieRobot
from client import WalkieAIClient

from walkie_sdk.utils import converters

from typing import Optional, Dict, Any
import os
import warnings

ZENOH_PORT = 7447

class Manipulate():
    def __init__(self, walkieAIClient, walkieInterface,debug=False): 
        '''
        Grasp instance to connect to WalkieAIClient(obj-detection) and WalkieInterface(GraspNet). 
        Pass in None in both fields to instantiate class without a server instance (for isolated tests)
        '''

        self.waklieAIClient = walkieAIClient
        self.walkieInterface = walkieInterface
        self.debug = debug

    def _manual_init(self): #for isolated tests, this can be used to init walkieAIClient and walkieInterface
        robot = self._get_robot()
        self.waklieAIClient = WalkieAIClient(
            base_url=os.getenv("WALKIE_AI_BASE_URL", "http://10.0.0.202:5000"),
        )
        self.walkieInterface = WalkieInterface(robot)

    def _get_robot(self) -> WalkieRobot:
        ros_protocol = os.getenv("WALKIE_ROS_PROTOCOL", "rosbridge")
        ros_port = int(os.getenv("WALKIE_ROS_PORT", str(ZENOH_PORT if ros_protocol == "zenoh" else 9090)))
        # 127.0.0.1 is correct when running on the robot itself (SSH'd in); set
        # WALKIE_ROBOT_IP to walkie's LAN address when running from a developer PC.
        robot_ip = os.getenv("WALKIE_ROBOT_IP", "10.0.0.201")
        return WalkieRobot(
            ip=robot_ip,
            ros_protocol=ros_protocol,
            ros_port=ros_port,
            camera_protocol="zenoh",
            camera_port=ZENOH_PORT,
        )

    def get_grasp_from_prompt(self,prompt:str, threshold:float = 0.7) -> Optional[Dict[str, Any]] :
        '''
        Interface to call Graspnet on WalkieAIClient, returning End effector pose.
        Input:
            prompt: str, the prompt to detect object, sent to WalkieAIClient's object detection (currently YOLOE)
            threshold: float, the confidence threshold for detection
        Output:
            dict: whatever the GrasphNet on the ROS node returns 
        '''
        retry = 3
        while retry > 0:
            try:
                image = self.walkieInterface.camera.capture()
            except RuntimeError as e:
                if self.debug:
                    print(f"Camera capture failed: {e}. Retrying...")
                retry -= 1
                continue

        detection = self.waklieAIClient.image.detect(image,return_mask=True,prompts = [prompt])

        filteredDetection = [d for d in detection 
                             if not d.confidence is None 
                             and d.confidence >= threshold]
        filteredDetection = sorted(filteredDetection, key=lambda d: d.confidence, reverse=True)

        if len(filteredDetection) == 0:
            warnings.warn("No detection met the confidence threshold")
            if self.debug:
                print(f"All detections: {[[x.bbox, x.confidence, x.class_id, x.class_name] for x in detection]}")
                print(f"Filtered detections: {filteredDetection}")
            return None

        if len(filteredDetection) > 1 : 
            if self.debug:
                print(f"All detections: \n {[[x.bbox, x.confidence, x.class_id, x.class_name] for x in detection]}")
                print(f"Filtered detections: \n {filteredDetection}")
            warnings.warn("YOLO has detected multiple objects from provided prompt, please provided a more detailed prompt or higher threshold. Defaulting to Highest confidence")
        
        mask = detection[0].mask
        encodedMask = converters.numpy_to_mono8_image(mask)
        return self.walkieInterface.robot.grasp.from_mask(encodedMask)
    

if __name__ == "__main__":
    manipulator = Manipulate(None,None,debug=True) #provide the proper WalkieAIClient and WalkieRobot instances for actual implementation
    manipulator._manual_init() # DONOT USE IN PRODUCTION, this is only for isolated testing
    results = manipulator.get_grasp_from_prompt("Red Bottle", threshold=0.4)
    print(results)




        