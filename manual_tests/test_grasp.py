"""Manual test for the /grasp endpoint (GraspNet over HTTP).

Two modes, picked by the WALKIE_GRASP_TEST_MODE env var (default "offline"):

  offline  — build a synthetic graspable box cloud (no robot, no camera) and call
             walkieAI.grasp.infer. Smoke-tests the full client→server→GraspNet path.
                 WALKIE_GRASP_TEST_MODE=offline uv run python -m manual_tests.test_grasp

  robot    — capture an RGB-D snapshot, run masked detection, lift the first
             detection's mask to a camera-optical cloud, and infer grasps on it. Then
             map the top grasp back to the map frame as a sanity check. Needs the
             robot (ZED + transforms) and a running walkie-ai-server.
                 WALKIE_GRASP_TEST_MODE=robot uv run python -m manual_tests.test_grasp

Both need walkie-ai-server up with the /grasp endpoint (open3d + graspnetAPI + the
graspnet-baseline pointnet2/knn CUDA ops installed in its venv).
"""

import os

import numpy as np
from dotenv import load_dotenv

from client import WalkieAIClient

from walkie_config import load_config

ZENOH_PORT = 7447
ROBOT_IP = "127.0.0.1"

# Open-vocab detector prompts for the robot path — graspable tabletop objects.
PROMPTS = ["bottle", "cup", "mug", "can", "box", "cube", "toy", "object"]


def _print_grasps(grasps) -> None:
    if not grasps:
        print("No grasps returned.")
        return
    print(f"{len(grasps)} grasp(s) (best first):")
    for i, g in enumerate(grasps[:10]):
        print(g)
        # anti = "" if g.antipodal_score is None else f" antipodal={g.antipodal_score:.2f}"
        # tx, ty, tz = g.translation
        # print(
        #     f"  [{i}] pos=({tx:+.3f}, {ty:+.3f}, {tz:+.3f}) "
        #     f"score={g.score:.3f} width={g.width * 100:.1f}cm{anti}"
        # )


def _synthetic_box_cloud() -> np.ndarray:
    """Points on a 4 cm box ~0.4 m in front, in the optical frame (X-right, Y-down, Z-fwd)."""
    rng = np.random.default_rng(0)
    half = 0.02
    center = np.array([0.0, 0.0, 0.4], dtype=np.float32)
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            pts = rng.uniform(-half, half, size=(500, 3)).astype(np.float32)
            pts[:, axis] = sign * half
            faces.append(pts)
    return (np.concatenate(faces, axis=0) + center).astype(np.float32)


def run_offline(walkieAI: WalkieAIClient) -> None:
    cloud = _synthetic_box_cloud()
    print(f"Synthetic box cloud: {cloud.shape[0]} points (optical frame)")
    grasps = walkieAI.grasp.infer(cloud, antipodal=True, max_grasps=10)
    _print_grasps(grasps)


def run_robot(walkieAI: WalkieAIClient) -> None:
    from walkie_sdk import WalkieRobot

    from interfaces.devices.camera import CameraSnapshot
    from interfaces.walkie_interface import WalkieInterface

    robot = WalkieRobot(ip=ROBOT_IP, camera_protocol="zenoh", camera_port=ZENOH_PORT)
    walkie = WalkieInterface(robot)
    print("Robot + AI client initialized")

    for _ in range(100):

        snap = CameraSnapshot.capture(walkie, log=print)
        if snap is None or not snap.has_geometry:
            print("No snapshot geometry (depth/pose/intrinsics) — is the ZED running?")
            continue

        detections = walkieAI.image.detect(snap.img, prompts=["red can"], return_mask=True)
        detections = [d for d in detections if d.mask is not None]
        if not detections:
            print("No masked detections in view.")
            continue
        det = detections[0]
        print(f"Grasping '{det.class_name}' (conf={det.confidence}) bbox={tuple(det.bbox)}")

        # Lift the mask to a camera-optical cloud (what GraspNet expects).
        cloud = snap.mask_to_points(det.mask, voxel= 0.005, frame="optical", erode_px=5)
        print(f"Lifted {cloud.shape[0]} optical-frame points")
        if cloud.shape[0] < 50:
            print("Too few points lifted — object too far / occluded?")
            continue

        grasps = walkieAI.grasp.infer(cloud, antipodal=True, max_grasps=1)
        _print_grasps(grasps)

        if grasps:
            # Sanity check: map the top grasp's centre back to the map frame.
            p_opt = np.asarray(grasps[0].translation, dtype=float)
            p_map = snap.cam.R @ p_opt + snap.cam.t
            print(f"Top grasp in map frame: ({p_map[0]:+.3f}, {p_map[1]:+.3f}, {p_map[2]:+.3f})")


def main() -> None:
    load_dotenv()
    load_config()
    mode = os.getenv("WALKIE_GRASP_TEST_MODE", "offline").strip().lower()
    walkieAI = WalkieAIClient(
        base_url=os.getenv("WALKIE_AI_BASE_URL", "http://localhost:5000"),
    )
    print(f"Mode: {mode}")
    if mode == "robot":
        run_robot(walkieAI)
    else:
        run_offline(walkieAI)


if __name__ == "__main__":
    main()
