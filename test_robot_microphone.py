from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from walkie_sdk import WalkieRobot

from walkie_config import load_config
from client import WalkieAIClient
from interfaces.walkie_interface import WalkieInterface
from interfaces.devices.microphone import list_audio_devices


ZENOH_PORT = 7447

def get_robot() -> WalkieRobot:
    load_dotenv()
    # Tuning knobs (perception/scene/explore/viewer) live in config.toml; .env
    # holds only secrets/endpoints/transport. setdefault means .env + real env
    # still win over config.toml.
    load_config()
    ros_protocol = os.getenv("WALKIE_ROS_PROTOCOL", "rosbridge")
    ros_port = int(os.getenv("WALKIE_ROS_PORT", str(ZENOH_PORT if ros_protocol == "zenoh" else 9090)))
    # 127.0.0.1 is correct when running on the robot itself (SSH'd in); set
    # WALKIE_ROBOT_IP to walkie's LAN address when running from a developer PC.
    robot_ip = os.getenv("WALKIE_ROBOT_IP", "127.0.0.1")
    return WalkieRobot(
        ip=robot_ip,
        ros_protocol=ros_protocol,
        ros_port=ros_port,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )



robot = get_robot()
walkieAI = WalkieAIClient(
base_url=os.getenv("WALKIE_AI_BASE_URL", "http://10.0.0.213:5000"),
)
print(f"List audio devices: {list_audio_devices()}")
walkie = WalkieInterface(robot, microphone_device=9)

while True:
    print("Say something...")
    text = walkie.microphone.record_until_silence()
    print("Transcribed text:", text)
