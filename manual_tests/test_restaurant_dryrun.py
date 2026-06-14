"""Off-robot dry run of the FULL Restaurant serve loop with stubbed hardware.

Exercises GoToStart -> ServeCustomers end to end (scan -> approach -> take order
-> relay -> pick/serve), so the orchestration + status transitions actually RUN
before the shared robot session — not just compile. Nav prints; the pose server
is faked to return a calling customer; detection/caption are faked; you type the
orders at the prompt. Manipulation stays fail-safe (uncalibrated -> logged, no
move). The only real dependency is the LLM (OPENROUTER_API_KEY) for order parsing.

    uv run python -m manual_tests.test_restaurant_dryrun
    RESTAURANT_BATCH=1 uv run python -m manual_tests.test_restaurant_dryrun   # batched loop

(Needs torch importable, i.e. the GPU box — tasks.base pulls the device stack in.)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from tasks.common import load_task_config

load_task_config(Path(__file__).resolve().parent.parent / "tasks" / "Restaurant")

from PIL import Image

from client.object_detection import DetectedObject
from client.pose_estimation import PersonPose, PoseKeypoint
from tasks.base import TaskContext
from tasks.common import initialize_llm_model
from tasks.Restaurant.subtasks import build_restaurant_task


class _Nav:
    def go_to(self, x, y, heading, blocking=True):
        print(f"  [stub nav] go_to({x:.2f}, {y:.2f}, {heading:.2f})")
        return "SUCCEEDED"


class _Status:
    def get_position(self):
        return {"x": 0.0, "y": 0.0, "heading": 0.0}


class _Snapshot:
    img = Image.new("RGB", (640, 480))
    has_geometry = True

    def bbox_world_xy(self, bbox_xyxy, **kw):
        return (2.0, 0.5)

    def bbox_world_point(self, bbox_xyxy, **kw):
        return (2.0, 0.5, 0.8)


class _Failing:
    def __getattr__(self, name):
        def fail(*a, **k):
            raise RuntimeError(f"stub {name}: not available in dry run")

        return fail


class _Walkie:
    nav = _Nav()
    status = _Status()
    speaker = _Failing()
    microphone = _Failing()
    arm = _Failing()

    def capture_snapshot(self):
        return _Snapshot()


def _calling_person() -> PersonPose:
    return PersonPose(
        bbox=(320, 240, 120, 320),
        confidence=0.9,
        keypoints=[
            PoseKeypoint(x=300, y=200, confidence=0.9, name="right_shoulder", index=6),
            PoseKeypoint(x=300, y=90, confidence=0.9, name="right_wrist", index=10),
        ],
    )


class _PoseEstimation:
    def estimate(self, img):
        return [_calling_person()]


class _ObjectDetection:
    def detect(self, img, prompts=None, return_mask=False):
        name = prompts[0] if prompts else "item"
        return [DetectedObject(mask=None, bbox=(260, 80, 380, 400), class_name=name, confidence=0.9)]


class _ImageCaption:
    def caption(self, image, prompt=None):
        return "a person in a blue shirt"


class _WalkieAI:
    pose_estimation = _PoseEstimation()
    object_detection = _ObjectDetection()
    image_caption = _ImageCaption()
    tts = _Failing()
    stt = _Failing()


def test_restaurant_dryrun():
    ctx = TaskContext(
        walkie=_Walkie(),
        walkieAI=_WalkieAI(),
        model=initialize_llm_model(),
        disable_listening=True,  # type the orders at the prompt
    )
    ok = build_restaurant_task(ctx).run()
    print("task returned:", ok)
    print("orders:", ctx.data.get("orders"))
    print("bar anchor:", ctx.data.get("bar_anchor"))


if __name__ == "__main__":
    test_restaurant_dryrun()
