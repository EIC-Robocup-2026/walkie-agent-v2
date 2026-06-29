"""On-robot background services.

Currently houses :mod:`services.realtime_explore` — the 3D scene-graph spatial memory and
the robot's perception loop (detect → ingest → write ``perception.json``). Import it
directly, e.g. ``from services.realtime_explore import WalkieGraphs``; nothing is re-exported
here to keep package import lightweight (realtime_explore pulls in open3d lazily; chromadb loads only on people use).
"""
