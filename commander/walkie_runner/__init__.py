"""Walkie Task Commander — a self-contained NiceGUI web UI that launches the
walkie-agent-v2 RoboCup challenges as subprocesses. No imports of the agent code:
it shells out to ``<repo>/.venv/bin/python -m tasks.<NAME>.run`` and reads back the
scorecard each task writes."""
