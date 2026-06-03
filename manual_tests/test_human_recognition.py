"""Interactive harness for the human-recognition tools (no LLM).

Drives the **real** human-agent tools (`agents/human_agent/tools.py`) against a
live webcam + a running walkie-ai-server, so you can test each feature
deterministically — press a key, the tool runs on the current frame, the result
prints. This bypasses the main agent so a flaky LLM delegation can't mask a
working (or broken) tool.

Run it as a module so ``from client import ...`` resolves, with the robot
*stopped* (it writes the same ``chroma_db_people`` dir main.py would):

    uv run python -m manual_tests.test_human_recognition
    uv run python -m manual_tests.test_human_recognition --reset   # forget people first
    uv run python -m manual_tests.test_human_recognition --device 2

Needs walkie-ai-server up at WALKIE_AI_BASE_URL with the
/face-recognition, /image-caption, /object-detection, /pose-estimation routes.

Keys (focus the OpenCV window):
    d  describe_person          c  count_people
    e  enroll_person (asks name/drink in the terminal)
    r  recognize_person         l  list_known_people
    s  find_empty_seat          g  locate_person (asks an optional name)
    q  quit
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import cv2
from dotenv import load_dotenv
from PIL import Image

# Load .env then config.toml so WALKIE_AI_BASE_URL and the [human] thresholds
# (FACE_MATCH_THRESHOLD, CAMERA_HFOV_DEG, …) apply exactly as in main.py.
load_dotenv()
from walkie_config import load_config  # noqa: E402

load_config()

from agents.human_agent.tools import make_human_tools  # noqa: E402
from client import WalkieAIClient  # noqa: E402
from perception import PeopleStore  # noqa: E402

HELP = (
    "[d]escribe  [c]ount  [e]nroll  [r]ecognize  "
    "[l]ist  find-[s]eat  [g]aze/locate  [q]uit"
)


def _build(device: int, reset: bool):
    walkieAI = WalkieAIClient(base_url=os.getenv("WALKIE_AI_BASE_URL", "http://localhost:5000"))

    model_name = ""
    try:
        model_name = walkieAI.face_recognition.get_model_name()
        print(f"[harness] face backend: {model_name}")
    except Exception as e:  # noqa: BLE001
        print(f"[harness] WARNING face route unreachable ({e!r}); enroll/recognize will error.")

    store = PeopleStore(
        persist_dir=os.getenv("PEOPLE_CHROMA_DIR", "chroma_db_people"),
        embedding_model=model_name,
        frames_dir=os.getenv("PEOPLE_FRAMES_DIR", "people_frames"),
    )
    if reset:
        store.clear()
        print("[harness] people memory cleared.")
    print(f"[harness] people memory: {store.count()} remembered")

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[harness] ERROR: could not open camera device {device}", file=sys.stderr)
        sys.exit(1)

    # The tools call walkie.camera.capture_pil() themselves; hand them the most
    # recent webcam frame (BGR → RGB PIL) so they see exactly what's on screen.
    latest = {"frame": None}

    class _Cam:
        def capture_pil(self) -> Image.Image:
            f = latest["frame"]
            if f is None:
                raise RuntimeError("no frame captured yet")
            return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    walkie = SimpleNamespace(camera=_Cam(), speaker=None)
    tools = {t.name: t for t in make_human_tools(walkie, walkieAI, people_store=store)}
    return cap, latest, tools


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _run(name: str, tools: dict, args: dict | None = None):
    print(f"\n=== {name}({args or ''}) ===")
    try:
        print(tools[name].invoke(args or {}))
    except Exception as e:  # noqa: BLE001 — surface, don't crash the loop
        print(f"[tool raised] {e!r}")
    print("-" * 60)


def main():
    ap = argparse.ArgumentParser(description="Manual test harness for human-recognition tools.")
    ap.add_argument("--device", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--reset", action="store_true", help="forget all remembered people first")
    opts = ap.parse_args()

    cap, latest, tools = _build(opts.device, opts.reset)
    print("\n" + HELP + "\n(focus the video window; enroll/locate prompt in this terminal)\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[harness] camera read failed; exiting.")
                break
            latest["frame"] = frame
            cv2.putText(frame, HELP, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("human-recognition harness", frame)
            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                break
            elif k == ord("d"):
                _run("describe_person", tools)
            elif k == ord("c"):
                _run("count_people", tools)
            elif k == ord("e"):
                name = _ask("  guest name: ")
                drink = _ask("  favorite drink: ")
                if name and drink:
                    _run("enroll_person", tools, {"name": name, "drink": drink})
                else:
                    print("  (skipped — need both name and drink)")
            elif k == ord("r"):
                _run("recognize_person", tools)
            elif k == ord("l"):
                _run("list_known_people", tools)
            elif k == ord("s"):
                _run("find_empty_seat", tools)
            elif k == ord("g"):
                who = _ask("  person name (blank = nearest): ")
                _run("locate_person", tools, {"name": who} if who else {})
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
