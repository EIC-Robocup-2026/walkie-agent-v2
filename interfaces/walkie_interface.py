from .devices.speaker import Speaker
from .devices.microphone import Microphone
from .devices.camera import Camera
from walkie_sdk.robot import WalkieRobot

class WalkieInterface:
    def __init__(self, robot: WalkieRobot):
        self._speaker = Speaker()
        self._microphone = Microphone()
        self._camera = Camera(robot)
        self._nav = robot.nav
        self._status = robot.status
    
    @property
    def speaker(self) -> Speaker:
        return self._speaker
    
    @property
    def microphone(self) -> Microphone:
        return self._microphone
    
    @property
    def camera(self) -> Camera:
        return self._camera
    
    @property
    def nav(self):
        return self._nav
    
    @property
    def status(self):
        return self._status
        