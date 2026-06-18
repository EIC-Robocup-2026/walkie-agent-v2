"""Off-robot dry run of the full HRI flow with stubbed hardware.

Exercises all 12 subtasks: nav calls are printed, camera/TTS/detector fail and
degrade, and you type the guests' answers at the prompt. The only real
dependency is the LLM (OPENROUTER_API_KEY) for name/drink extraction.

    uv run python -m manual_tests.test_hri_dryrun
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from tasks.common import load_task_config

load_task_config(Path(__file__).resolve().parent.parent / "tasks" / "HRI")

from tasks.base import TaskContext
from tasks.common import initialize_llm_model
from tasks.HRI.subtasks import build_hri_task


class _Nav:
    def go_to(self, x, y, heading, blocking=True):
        print(f"  [stub nav] go_to({x:.2f}, {y:.2f}, {heading:.2f})")


class _Status:
    def get_position(self):
        return {"x": 0.0, "y": 0.0, "heading": 0.0}


class _Failing:
    """Any method raises — exercises the degrade paths."""

    def __getattr__(self, name):
        def fail(*a, **k):
            raise RuntimeError(f"stub {name}: not available in dry run")

        return fail


class _Arm:
    """Non-raising arm stub (local hardware): every method prints. The top arm
    exposes ``left``/``right`` sub-arms; ``get_joint_states`` returns a plausible
    triple so the bag-handover step can read an effort without crashing."""

    def __init__(self, label="arm"):
        self._label = label
        if label == "arm":
            self.left = _Arm("arm.left")
            self.right = _Arm("arm.right")

    def __getattr__(self, name):
        def method(*a, **k):
            print(f"  [stub {self._label}] {name}()")
            if name == "get_joint_states":
                return ([], [], [0.0, 0.0, 0.0, 0.0])
            return None

        return method


class _Head:
    def tilt(self, angle_rad):
        print(f"  [stub head] tilt({angle_rad})")


class _Robot:
    """Mirrors WalkieInterface.robot — local arm/head, non-raising like _Nav."""

    arm = _Arm()
    head = _Head()


class _Walkie:
    robot = _Robot()
    nav = _Nav()
    status = _Status()
    camera = _Failing()
    speaker = _Failing()
    microphone = _Failing()
    arm = _Failing()


class _WalkieAI:
    tts = _Failing()
    stt = _Failing()
    object_detection = _Failing()
    pose_estimation = _Failing()
    image_caption = _Failing()
    face_recognition = _Failing()
    appearance = _Failing()


def test_hri_dryrun():
    from perception import PeopleStore

    ctx = TaskContext(
        walkie=_Walkie(),
        walkieAI=_WalkieAI(),
        model=initialize_llm_model(),
        disable_listening=True,  # type answers at the prompt
        people=PeopleStore(persist_dir=None),  # in-memory; enrollment will degrade
    )
    ok = build_hri_task(ctx).run()
    print("task returned:", ok)
    print("blackboard guests:", ctx.data.get("guests"))
    print("blackboard seats:", ctx.data.get("seats"))


if __name__ == "__main__":
    test_hri_dryrun()
