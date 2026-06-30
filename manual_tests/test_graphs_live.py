"""Live walkie_graphs demo — drive the robot camera into the 3D scene graph.

Run on the robot (or a dev PC with WALKIE_ROBOT_IP set) with walkie-ai-server up::

    WALKIE_VIZ=rerun uv run python -m manual_tests.test_graphs_live

A Rerun viewer shows the accumulating point clouds, per-object boxes, and relation
edges; the console prints the text description each tick. Set
WALKIE_EXPLORE_INTERESTED_CLASSES (e.g. "bottle,cup,chair") to scope what's mapped.
Ctrl+C to stop. This is a manual test (no robot in CI) — guarded by __main__.

To watch from ANOTHER computer on the LAN, add WALKIE_VIZ_SERVE=1 (the
startup log prints the URL). The servers bind 0.0.0.0, so if a remote machine can't
connect it's the robot's host firewall — open BOTH ports on the robot, e.g.
``sudo ufw allow 9090/tcp && sudo ufw allow 9876/tcp``.
"""

import os
import time

from dotenv import load_dotenv
from walkie_sdk import WalkieRobot

from client import WalkieAIClient
from interfaces.walkie_interface import WalkieInterface
from walkie_config import load_config
from services.realtime_explore import RealtimeExplore

ZENOH_PORT = 7447


def test_graphs_live() -> None:
    load_dotenv()
    load_config()
    os.environ.setdefault("WALKIE_VIZ", "rerun")

    robot = WalkieRobot(
        ip=os.getenv("WALKIE_ROBOT_IP", "127.0.0.1"),
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )
    walkie = WalkieInterface(robot)
    walkieAI = WalkieAIClient(
        base_url=os.getenv("WALKIE_AI_BASE_URL", "http://localhost:5000"),
    )
    graphs = RealtimeExplore(walkieAI=walkieAI, walkie=walkie)
    print("[graphs-live] starting observer — Ctrl+C to stop.")

    try:
        graphs.start()
        while True:
            time.sleep(float(os.getenv("WALKIE_EXPLORE_INTERVAL_SEC", "3.0")) + 0.5)
            print("\n" + graphs.to_text_description())
    except KeyboardInterrupt:
        print("\n[graphs-live] stopping.")
    finally:
        graphs.stop()
        walkie.close()


if __name__ == "__main__":
    test_graphs_live()
