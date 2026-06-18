import os
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import (
    load_task_config,
    initialize_llm_model,
    initialize_robot,
)
from .subtasks import build_hri_task
from client import WalkieAIClient
from perception import PeopleStore


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    # Initialize
    walkie_interface = initialize_robot()
    walkie_interface.walkie