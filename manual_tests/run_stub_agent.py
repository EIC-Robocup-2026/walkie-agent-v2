"""Run the full Walkie agent brain offline — no robot, no walkie-ai-server.

Every leaf tool (movement, detection, captioning, memory, speak) is short-circuited
to a canned success by ``StubToolMiddleware``, and ``TraceMiddleware`` prints each
agent's reasoning + tool decisions live. Sub-agents really run, so you see all four
agents think. The ONLY real dependency is the LLM (OpenRouter) — that's the point:
watch the thinking process while iterating on prompts/behavior.

Run as a module so the repo root is on sys.path::

    uv run python -m manual_tests.run_stub_agent

Type instructions at the prompt; ``quit`` / ``exit`` / Ctrl-D to leave.
Needs OPENROUTER_API_KEY in .env (or LLM_USE_LOCAL=1 + a local endpoint).
"""

# BUG: Bash log script currently logs to currently open kernel, please fix to log to same dir as this script

from __future__ import annotations

import logging
import os
import types

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from walkie_config import load_config
import json

StREAMING = True  # set True to see the model's token stream in real time (debugging)


def _stub_walkie() -> types.SimpleNamespace:
    """A placeholder robot interface. Never actually called — all leaf tools are
    short-circuited — so the hardware handles are ``None``. A missed stub then
    surfaces loudly as an AttributeError instead of silently hitting hardware."""
    return types.SimpleNamespace(
        nav=None,
        status=None,
        arm=None,
        tools=None,
        camera=None,
        speaker=None,
        microphone=None,
        close=lambda: None,
    )


def main() -> None:
    load_dotenv()
    load_config()

    # Flip stub mode BEFORE building agents: the factory reads these env flags at
    # agent-creation time, and we want the boot to stay fully offline.
    os.environ["WALKIE_STUB_TOOLS"] = "1"
    os.environ["WALKIE_TRACE"] = "1"
    os.environ["WALKIE_GRAPHS_ENABLED"] = "0"  # no perception loop (needs camera)
    os.environ["DISABLE_LISTENING"] = "1"  # type prompts, no mic

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Imported here, after env is set, to keep intent obvious. build_model() is
    # module-level in main and main() is __main__-guarded, so this is side-effect-free.
    from main import build_model

    from agents.actuator_agent import create_actuator_agent
    from agents.core.robot_context import RobotContext
    from agents.database_agent import create_database_agent
    from agents.vision_agent import create_vision_agent
    from agents.walkie_agent import create_walkie_main_agent

    ctx = RobotContext.init(perception_path=os.getenv("PERCEPTION_PATH", "perception.json"))
    ctx.stage = "ready"

    model = build_model()
    walkie = _stub_walkie()
    walkieAI = types.SimpleNamespace()

    actuator = create_actuator_agent(model, walkieAI, walkie)
    vision = create_vision_agent(model, walkieAI, walkie)
    database = create_database_agent(model, walkieAI, walkie, graphs=None)
    walkie_agent = create_walkie_main_agent(model, walkieAI, walkie, actuator, vision, database)

    print("=" * 70)
    print("STUB MODE — no robot, no AI server. Every tool returns success.")
    print("Watch [THINK]/[STUB]/[SPEAK] lines for the agent's reasoning.")
    print("Type an instruction; 'quit'/'exit'/Ctrl-D to leave.")
    print("=" * 70)
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            print("bye.")
            break

        print(f"\n--- turn: {text!r} ---")
        try:
            if StREAMING:
                tokens = []
                for chunk in walkie_agent.stream(
                    {"messages": [HumanMessage(content=text)]},
                    config={"configurable": {"thread_id": "main"}},
                    stream_mode="messages",
                    version="v2",
                ):
                    if chunk["type"] == "messages":
                        token, metadata = chunk["data"]
                        if token.content_blocks:
                            tokens.append(token.content_blocks[-1])
                            if token.content_blocks[-1].get('text', ''):
                                print(f"{token.content_blocks[-1].get('text', '')}",end='', flush=True)
                                tokens.append(token.content_blocks[-1].get('text', ''))
                            else:
                                print(f"{token.content_blocks}")
                                tokens.append('\n')
                                tokens.append(json.dumps(token.content_blocks))
            else:
                events = walkie_agent.invoke(
                    {"messages": [HumanMessage(content=text)]},
                    config={"configurable": {"thread_id": "main"}},
                )
                print(f"Token used {events["messages"][-1].usage_metadata}")
        except Exception as exc:  # noqa: BLE001 — keep the REPL alive across failures
            print(f"[error] turn failed: {exc!r}")
        print("--- end turn ---")


if __name__ == "__main__":
    main()
