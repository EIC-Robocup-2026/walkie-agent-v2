"""Scene-embedding perception subsystem.

A persistent spatial-semantic memory of objects the robot has seen:
where they are (3D map-frame), what they look like (caption + visual
embedding), and when they were last seen. Backed by ChromaDB.

Public surface:
    SceneStore     — read/write façade over the underlying vector DB
    SceneEntry     — a single record in the scene memory
    run_scene_perception — async background loop that ingests detections

See ``docs/scene_perception_design.md`` for the schema and dedup strategy.
"""

from .embedders import LocalCLIPEmbedder, RemoteCLIPEmbedder
from .lifters import RobotPoseLifter
from .types import (
    CameraSource,
    Captioner,
    Detection,
    DedupDecision,
    Detector,
    Embedder,
    PositionLifter,
    SceneDiff,
    SceneEntry,
    TickReport,
)
from .store import SceneStore
from .people_store import PeopleStore, PersonRecord
from .loop import run_scene_perception
from .pipeline import process_frame

__all__ = [
    "CameraSource",
    "Captioner",
    "DedupDecision",
    "Detection",
    "Detector",
    "Embedder",
    "LocalCLIPEmbedder",
    "PeopleStore",
    "PersonRecord",
    "PositionLifter",
    "RemoteCLIPEmbedder",
    "RobotPoseLifter",
    "SceneDiff",
    "SceneEntry",
    "SceneStore",
    "TickReport",
    "process_frame",
    "run_scene_perception",
]
