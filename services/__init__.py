"""On-robot background services.

Currently houses :mod:`services.walkie_graphs` — the 3D scene-graph spatial memory and
the robot's perception loop (detect → ingest → write ``perception.json``). Import it
directly, e.g. ``from services.walkie_graphs import WalkieGraphs``; nothing is re-exported
here to keep package import lightweight (walkie_graphs pulls in chromadb/torch lazily).
"""
