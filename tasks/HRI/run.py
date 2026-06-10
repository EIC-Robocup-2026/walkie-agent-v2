import os

from dotenv import load_dotenv

from ..common import (
    load_config,
    initialize_llm_model,
    initialize_robot,
    WalkieBrain,
)
from client import WalkieAIClient


def main() -> None:
    load_dotenv()
    load_config()

    # Initialize
    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()

    # Main agent and memory graph
    walkie_brain = WalkieBrain(walkie_ai, walkie_interface, model)

    # Flow start here
    walkie_brain.listen_and_act()