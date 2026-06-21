"""Off-robot dry run of the Restaurant Phase 0 slice with stubbed hardware.

Exercises GoToStart -> ScanAndApproach end to end: nav calls are printed, the
pose server is faked to return one customer with a raised hand, and the snapshot
lift returns a fixed map point. Proves the scan -> detect -> approach control
flow without a robot. (Run on the GPU box — tasks.common pulls the device stack;
won't run on a CUDA-less laptop.)

    uv run python -m manual_tests.test_restaurant_phase0_dryrun
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from tasks.common import load_task_config

load_task_config(Path(__file__).resolve().parent.parent / "tasks" / "Restaurant")

from PIL import Image

from client import PersonPose, PoseKeypoint
from tasks.base import TaskContext
from tasks.common import initialize_llm_model
from tasks.Restaurant.subtasks import build_phase0_slice


class _Nav:
    def go_to(self, x, y, heading, blocking=True):
        print(f"  [stub nav] go_to({x:.2f}, {y:.2f}, {heading:.2f})")
        return "SUCCEEDED"


class _Status:
    def get_position(self):
        return {"x": 0.0, "y": 0.0, "heading": 0.0}


class _Snapshot:
    """Minimal CameraSnapshot stand-in: geometry present, lift returns a fixed point."""

    img = Image.new("RGB", (640, 480))
    has_geometry = True

    def bbox_world_xy(self, bbox_xyxy, **kw):
        return (2.0, 0.5)  # 2 m ahead, slightly left — the "customer's" map point


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
    """One person with the right hand clearly raised above the shoulder."""
    return PersonPose(
        bbox=(320, 240, 120, 320),
        confidence=0.9,
        keypoints=[
            PoseKeypoint(x=300, y=200, confidence=0.9, name="right_shoulder", index=6),
            PoseKeypoint(x=300, y=90, confidence=0.9, name="right_wrist", index=10),
        ],
    )


class _Image:
    """Unified image client stub (walkieAI.image.*)."""

    def estimate_poses(self, img):
        return [_calling_person()]

    def detect(self, img, prompts=None, return_mask=False):
        raise RuntimeError("stub detect: not used in the Phase 0 slice")

    def caption(self, image, prompt=None):
        raise RuntimeError("stub caption: not used in the Phase 0 slice")


class _WalkieAI:
    image = _Image()
    tts = _Failing()
    stt = _Failing()


def test_restaurant_phase0_dryrun():
    ctx = TaskContext(
        walkie=_Walkie(),
        walkieAI=_WalkieAI(),
        model=initialize_llm_model(),
        disable_listening=True,
    )
    ok = build_phase0_slice(ctx).run()
    print("task returned:", ok)
    print("target caller:", ctx.data.get("target"))
    print("bar anchor:", ctx.data.get("bar_anchor"))


if __name__ == "__main__":
    test_restaurant_phase0_dryrun()
