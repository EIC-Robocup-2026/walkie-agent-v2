from __future__ import annotations

import math
import time
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings


Position = tuple[float, float, float]


def _l2(a: Position, b: Position) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


class WalkieVectorDB:
    """Persistent ChromaDB store for objects the robot has seen.

    Each record has a class label, a 3D position (in map frame), a confidence,
    a sighting count, and a free-text caption. The embedded document is
    "<class_name>: <caption>" so semantic queries can match either.
    """

    OBJECTS_COLLECTION = "objects"

    def __init__(self, persist_dir: str | Path = "chroma_db") -> None:
        path = str(Path(persist_dir).resolve())
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._objects = self._client.get_or_create_collection(
            name=self.OBJECTS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_object(
        self,
        class_name: str,
        position: Position,
        confidence: float,
        caption: str = "",
        sightings: int = 1,
    ) -> str:
        obj_id = str(uuid.uuid4())
        x, y, z = position
        self._objects.add(
            ids=[obj_id],
            documents=[f"{class_name}: {caption}".strip(": ").strip()],
            metadatas=[
                {
                    "class_name": class_name,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "confidence": float(confidence),
                    "sightings": int(sightings),
                    "caption": caption,
                    "last_seen_ts": time.time(),
                }
            ],
        )
        return obj_id

    def update_object(
        self,
        obj_id: str,
        *,
        position: Position | None = None,
        confidence: float | None = None,
        caption: str | None = None,
        sightings: int | None = None,
    ) -> None:
        existing = self._objects.get(ids=[obj_id])
        if not existing["ids"]:
            return
        meta = dict(existing["metadatas"][0])
        if position is not None:
            meta["x"], meta["y"], meta["z"] = (float(c) for c in position)
        if confidence is not None:
            meta["confidence"] = float(confidence)
        if sightings is not None:
            meta["sightings"] = int(sightings)
        if caption is not None:
            meta["caption"] = caption
        meta["last_seen_ts"] = time.time()
        new_doc = f"{meta['class_name']}: {meta.get('caption', '')}".strip(": ").strip()
        self._objects.update(ids=[obj_id], documents=[new_doc], metadatas=[meta])

    def find_nearby(
        self, class_name: str, position: Position, radius: float
    ) -> list[dict[str, Any]]:
        """Return existing records of the same class within `radius` meters."""
        result = self._objects.get(where={"class_name": class_name})
        out: list[dict[str, Any]] = []
        for obj_id, meta in zip(result["ids"], result["metadatas"]):
            pos = (float(meta["x"]), float(meta["y"]), float(meta["z"]))
            if _l2(pos, position) <= radius:
                out.append({"id": obj_id, **meta, "position": pos})
        out.sort(key=lambda r: _l2(r["position"], position))
        return out

    def query_objects(
        self, query_text: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Semantic search across object docs (class + caption)."""
        n_results = max(1, min(n_results, max(1, self._objects.count())))
        if self._objects.count() == 0:
            return []
        result = self._objects.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        out: list[dict[str, Any]] = []
        ids = result["ids"][0]
        metas = result["metadatas"][0]
        docs = result["documents"][0]
        dists = result.get("distances", [[None] * len(ids)])[0]
        for obj_id, meta, doc, dist in zip(ids, metas, docs, dists):
            out.append(
                {
                    "id": obj_id,
                    "document": doc,
                    "distance": dist,
                    "position": (float(meta["x"]), float(meta["y"]), float(meta["z"])),
                    **meta,
                }
            )
        return out

    def list_all(self) -> list[dict[str, Any]]:
        result = self._objects.get()
        out: list[dict[str, Any]] = []
        for obj_id, meta, doc in zip(
            result["ids"], result["metadatas"], result["documents"]
        ):
            out.append(
                {
                    "id": obj_id,
                    "document": doc,
                    "position": (float(meta["x"]), float(meta["y"]), float(meta["z"])),
                    **meta,
                }
            )
        return out

    def clear(self) -> None:
        self._client.delete_collection(self.OBJECTS_COLLECTION)
        self._objects = self._client.get_or_create_collection(
            name=self.OBJECTS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._objects.count()
