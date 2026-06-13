"""Manual hardware poke: print the robot head tilt angle in a loop.

Needs the robot reachable (set WALKIE_ROBOT_IP from a dev PC; 127.0.0.1 on-robot).
Run: uv run python -m manual_tests.test_robot_head
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from walkie_sdk import WalkieRobot

from walkie_config import load_config

ZENOH_PORT = 7447


def get_robot() -> WalkieRobot:
    # ROS services go over rosbridge, camera over zenoh (the SDK's default split).
    # 127.0.0.1 is correct on the robot itself; set WALKIE_ROBOT_IP from a dev PC.
    ros_protocol = os.getenv("WALKIE_ROS_PROTOCOL", "rosbridge")
    ros_port = int(os.getenv("WALKIE_ROS_PORT", str(ZENOH_PORT if ros_protocol == "zenoh" else 9090)))
    robot_ip = os.getenv("WALKIE_ROBOT_IP", "127.0.0.1")
    return WalkieRobot(
        ip=robot_ip,
        ros_protocol=ros_protocol,
        ros_port=ros_port,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )


def main() -> None:
    load_dotenv()
    load_config()
    robot = get_robot()
    print("Robot connected:", robot.is_connected)
    while True:
        print("Tilt:", robot.head.get_angle())
        time.sleep(0.5)


if __name__ == "__main__":
    main()
