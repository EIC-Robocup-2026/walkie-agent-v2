"""ChromaDB-backed scene store.

One collection (``scene_entries``), cosine space. Every record carries
the full set of metadata fields defined in the design doc; lists are
JSON-encoded into strings because Chroma metadata values must be scalars.

The store knows nothing about cameras, detectors, or async loops — it's
a pure read/write façade. ``classify`` from ``dedup.py`` is what makes
the merge decisions; the store just executes them.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import chromadb
from chromadb.config import Settings
from PIL import Image

from .dedup import (
    classify,
    cosine_similarity,
    get_dedup_radius_m,
    l2_distance,
    merged_confidence,
    merged_position,
)
from .types import DedupDecision, Detection, Embedder, SceneDiff, SceneEntry


def _embeddings_or_blanks(result: dict, n: int) -> list:
    """Chroma may return embeddings as a numpy array or omit them entirely.

    Standard truthy fallback ``result.get("embeddings") or [None]*n`` fails
    because ``bool(numpy.ndarray)`` raises. Handle both shapes explicitly.
    """
    embs = result.get("embeddings")
    if embs is None:
        return [None] * n
    try:
        length = len(embs)
    except TypeError:
        return [None] * n
    if length == 0:
        return [None] * n
    return list(embs)


_log = logging.getLogger("perception.store")


def _bucket(pos: tuple[float, float, float], radius: float) -> tuple[int, int, int]:
    r = max(radius, 1e-6)
    return (round(pos[0] / r), round(pos[1] / r), round(pos[2] / r))


class SceneStore:
    """Read/write façade over a ChromaDB persistent (or in-memory) collection.

    Pass ``persist_dir=None`` to get an ephemeral in-memory client — useful
    for tests. Otherwise the directory is created on first use.
    """

    COLLECTION = "scene_entries"

    def __init__(
        self,
        *,
        persist_dir: str | Path | None = "chroma_db_scene",
        embedder: Optional[Embedder] = None,
        frames_dir: str | Path | None = None,
    ) -> None:
        if persist_dir is None:
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        else:
            path = str(Path(persist_dir).resolve())
            self._client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder
        self._frames_dir = Path(frames_dir) if frames_dir else None
        if self._frames_dir:
            self._frames_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ writes

    def upsert(
        self,
        detection: Detection,
        *,
        source_frame: Optional[Image.Image] = None,
    ) -> tuple[str, DedupDecision]:
        """Insert or update a record for ``detection``.

        Returns ``(chroma_id, decision)`` — the id of the affected record
        and the dedup decision (for logging / test assertions).
        """
        radius = get_dedup_radius_m()
        candidates = self.find_nearby(
            class_name=detection.class_name,
            position=detection.position,
            radius=radius,
        )
        decision = classify(detection, candidates)

        # Pre-compute audit fields (closest candidate's dist + similarity)
        # so insert and update share a consistent, parsable log shape.
        if candidates:
            closest = min(
                candidates, key=lambda c: l2_distance(c.position, detection.position)
            )
            closest_id: Optional[str] = closest.id
            closest_dist: float = l2_distance(closest.position, detection.position)
            closest_sim: float = cosine_similarity(
                closest.embedding, detection.embedding
            )
        else:
            closest_id = None
            closest_dist = float("nan")
            closest_sim = float("nan")

        if decision.action == "update":
            assert decision.target_id is not None
            self._apply_update(decision.target_id, detection)
            _log.info(
                "scene.dedup action=UPDATE id=%s class=%s matched_id=%s "
                "dist=%.3f sim=%.3f reason=%s",
                decision.target_id,
                detection.class_name,
                decision.target_id,
                closest_dist,
                closest_sim,
                decision.reason,
            )
            return decision.target_id, decision

        # INSERT
        new_id = self._make_id(detection)
        frame_ref = self._maybe_archive(detection, source_frame, new_id)
        self._collection.add(
            ids=[new_id],
            documents=[self._format_document(detection.class_name, detection.caption)],
            embeddings=[list(detection.embedding)],
            metadatas=[self._metadata_for_insert(detection, frame_ref)],
        )
        _log.info(
            "scene.dedup action=INSERT id=%s class=%s matched_id=%s "
            "dist=%.3f sim=%.3f reason=%s",
            new_id,
            detection.class_name,
            closest_id if closest_id is not None else "null",
            closest_dist,
            closest_sim,
            decision.reason,
        )
        return new_id, decision

    def _apply_update(self, target_id: str, detection: Detection) -> None:
        existing = self._collection.get(
            ids=[target_id], include=["metadatas", "embeddings"]
        )
        if not existing["ids"]:
            # Race condition or stale id — fall back to insert by re-adding.
            _log.warning("update target %s vanished; inserting instead", target_id)
            self._collection.add(
                ids=[target_id],
                documents=[
                    self._format_document(detection.class_name, detection.caption)
                ],
                embeddings=[list(detection.embedding)],
                metadatas=[self._metadata_for_insert(detection, None)],
            )
            return

        meta = dict(existing["metadatas"][0])
        existing_embs = _embeddings_or_blanks(existing, 1)
        existing_emb = existing_embs[0]
        n = int(meta.get("sightings", 1))
        old_pos = (
            float(meta.get("x", 0.0)),
            float(meta.get("y", 0.0)),
            float(meta.get("z", 0.0)),
        )
        new_pos = merged_position(old_pos, n, detection.position)
        new_conf = merged_confidence(
            float(meta.get("position_conf", 0.0)), n, detection.confidence
        )

        meta.update(
            {
                "x": new_pos[0],
                "y": new_pos[1],
                "z": new_pos[2],
                "position_conf": new_conf,
                "sightings": n + 1,
                "last_seen_ts": detection.ts,
                "caption": detection.caption,
                "bbox_last": json.dumps(list(detection.bbox_xyxy)),
            }
        )
        # Keep frame_ref if a new one wasn't supplied (updates don't archive).
        # Keep embedding unchanged on update (see design doc rationale).

        # Pass embeddings through explicitly. If we supply `documents` to
        # Chroma's .update without `embeddings`, it re-embeds with its
        # default embedder (different dimension → InvalidArgumentError).
        # The design doc also says keep the original embedding on update.
        update_kwargs: dict[str, Any] = {
            "ids": [target_id],
            "documents": [self._format_document(detection.class_name, detection.caption)],
            "metadatas": [meta],
        }
        if existing_emb is not None:
            update_kwargs["embeddings"] = [list(existing_emb)]
        self._collection.update(**update_kwargs)

    # ------------------------------------------------------------------ reads

    def find_nearby(
        self,
        class_name: str,
        position: tuple[float, float, float],
        radius: float,
    ) -> list[SceneEntry]:
        """All entries of ``class_name`` within ``radius`` of ``position``.

        Sorted by L2 distance ascending. Used by :meth:`upsert` to feed
        dedup; also exposed for callers that want spatial filtering by
        class without a vector query.
        """
        result = self._collection.get(
            where={"class_name": class_name},
            include=["metadatas", "documents", "embeddings"],
        )
        out = []
        for cid, meta, doc, emb in zip(
            result["ids"],
            result["metadatas"],
            result["documents"],
            _embeddings_or_blanks(result, len(result["ids"])),
        ):
            entry = self._row_to_entry(cid, meta, doc, emb)
            if l2_distance(entry.position, position) <= radius:
                out.append(entry)
        out.sort(key=lambda e: l2_distance(e.position, position))
        return out

    def semantic_query(
        self,
        text: str,
        *,
        n_results: int = 5,
        min_last_seen_ts: Optional[float] = None,
        within_radius_of: Optional[tuple[float, float, float]] = None,
        max_distance_m: Optional[float] = None,
        class_name: Optional[str] = None,
    ) -> list[SceneEntry]:
        """KNN over text embedding, optionally filtered by recency / spatial / class."""
        if self._embedder is None:
            raise RuntimeError(
                "SceneStore was constructed without an embedder; semantic_query "
                "requires one. Pass `embedder=` to the constructor."
            )
        if self.count == 0:
            return []
        query_vec = self._embedder.embed_text(text)
        where = self._build_where(min_last_seen_ts, class_name)
        # Over-fetch so post-filter spatial doesn't starve.
        fetch = max(n_results * 4, n_results)
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(fetch, max(1, self.count)),
            where=where,
            include=["metadatas", "documents", "embeddings", "distances"],
        )
        entries = self._unpack_query(result)
        if within_radius_of is not None and max_distance_m is not None:
            entries = [
                e for e in entries
                if l2_distance(e.position, within_radius_of) <= max_distance_m
            ]
        return entries[:n_results]

    def visual_query(
        self,
        image: Image.Image,
        *,
        n_results: int = 5,
        min_last_seen_ts: Optional[float] = None,
        within_radius_of: Optional[tuple[float, float, float]] = None,
        max_distance_m: Optional[float] = None,
        class_name: Optional[str] = None,
    ) -> list[SceneEntry]:
        """KNN over image embedding (CLIP image-tower)."""
        if self._embedder is None:
            raise RuntimeError(
                "SceneStore was constructed without an embedder; visual_query "
                "requires one."
            )
        if self.count == 0:
            return []
        query_vec = self._embedder.embed_image(image)
        where = self._build_where(min_last_seen_ts, class_name)
        fetch = max(n_results * 4, n_results)
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(fetch, max(1, self.count)),
            where=where,
            include=["metadatas", "documents", "embeddings", "distances"],
        )
        entries = self._unpack_query(result)
        if within_radius_of is not None and max_distance_m is not None:
            entries = [
                e for e in entries
                if l2_distance(e.position, within_radius_of) <= max_distance_m
            ]
        return entries[:n_results]

    def spatial_query(
        self,
        center: tuple[float, float, float],
        radius_m: float,
        *,
        class_name: Optional[str] = None,
        n_results: Optional[int] = None,
    ) -> list[SceneEntry]:
        """All entries within ``radius_m`` of ``center``. No vector search."""
        where: dict[str, Any] = {}
        if class_name is not None:
            where["class_name"] = class_name
        result = self._collection.get(
            where=where or None,
            include=["metadatas", "documents", "embeddings"],
        )
        out = []
        for cid, meta, doc, emb in zip(
            result["ids"],
            result["metadatas"],
            result["documents"],
            _embeddings_or_blanks(result, len(result["ids"])),
        ):
            entry = self._row_to_entry(cid, meta, doc, emb)
            if l2_distance(entry.position, center) <= radius_m:
                out.append(entry)
        out.sort(key=lambda e: l2_distance(e.position, center))
        if n_results is not None:
            out = out[:n_results]
        return out

    def recency_query(
        self,
        since_ts: float,
        *,
        class_name: Optional[str] = None,
        n_results: Optional[int] = None,
    ) -> list[SceneEntry]:
        """Entries with ``last_seen_ts > since_ts``, newest first."""
        where: dict[str, Any] = {"last_seen_ts": {"$gt": float(since_ts)}}
        if class_name is not None:
            where = {"$and": [where, {"class_name": class_name}]}
        result = self._collection.get(
            where=where,
            include=["metadatas", "documents", "embeddings"],
        )
        out = [
            self._row_to_entry(cid, meta, doc, emb)
            for cid, meta, doc, emb in zip(
                result["ids"],
                result["metadatas"],
                result["documents"],
                _embeddings_or_blanks(result, len(result["ids"])),
            )
        ]
        out.sort(key=lambda e: e.last_seen_ts, reverse=True)
        if n_results is not None:
            out = out[:n_results]
        return out

    def diff(
        self,
        since_ts: float,
        *,
        within: Optional[tuple[tuple[float, float, float], float]] = None,
    ) -> SceneDiff:
        """Partition entries into appeared / refreshed / disappeared."""
        result = self._collection.get(
            include=["metadatas", "documents", "embeddings"],
        )
        entries = [
            self._row_to_entry(cid, meta, doc, emb)
            for cid, meta, doc, emb in zip(
                result["ids"],
                result["metadatas"],
                result["documents"],
                _embeddings_or_blanks(result, len(result["ids"])),
            )
        ]
        if within is not None:
            center, radius = within
            entries = [
                e for e in entries if l2_distance(e.position, center) <= radius
            ]

        appeared = [e for e in entries if e.first_seen_ts > since_ts]
        refreshed = [
            e for e in entries
            if e.first_seen_ts <= since_ts and e.last_seen_ts > since_ts
        ]
        disappeared = [e for e in entries if e.last_seen_ts <= since_ts]
        for lst in (appeared, refreshed, disappeared):
            lst.sort(key=lambda e: e.last_seen_ts, reverse=True)
        return SceneDiff(
            appeared=tuple(appeared),
            refreshed=tuple(refreshed),
            disappeared=tuple(disappeared),
        )

    # ----------------------------------------------------------------- prune

    def prune(
        self,
        *,
        ttl_sec: Optional[float] = None,
        max_records: Optional[int] = None,
        now: Optional[float] = None,
    ) -> int:
        """Remove stale or excess records. Returns count pruned."""
        now = now if now is not None else time.time()
        removed: set[str] = set()
        result = self._collection.get(include=["metadatas"])
        rows = list(zip(result["ids"], result["metadatas"]))

        if ttl_sec is not None:
            cutoff = now - ttl_sec
            removed.update(
                cid for cid, meta in rows
                if float(meta.get("last_seen_ts", 0)) < cutoff
            )

        if max_records is not None:
            surviving = [(cid, meta) for cid, meta in rows if cid not in removed]
            if len(surviving) > max_records:
                surviving.sort(
                    key=lambda r: float(r[1].get("last_seen_ts", 0)),
                    reverse=True,
                )
                for cid, _ in surviving[max_records:]:
                    removed.add(cid)

        if removed:
            self._collection.delete(ids=list(removed))
            _log.info("scene.prune removed=%d", len(removed))
        return len(removed)

    def clear(self) -> None:
        """Drop and recreate the collection (test/maintenance helper)."""
        self._client.delete_collection(self.COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def get_by_id(self, entry_id: str) -> Optional[SceneEntry]:
        result = self._collection.get(
            ids=[entry_id],
            include=["metadatas", "documents", "embeddings"],
        )
        if not result["ids"]:
            return None
        embs = _embeddings_or_blanks(result, len(result["ids"]))
        return self._row_to_entry(
            result["ids"][0],
            result["metadatas"][0],
            result["documents"][0],
            embs[0],
        )

    # ----------------------------------------------------------------- internals

    @staticmethod
    def _format_document(class_name: str, caption: str) -> str:
        caption = (caption or "").strip()
        return f"{class_name}. {caption}".rstrip(". ").strip()

    def _make_id(self, detection: Detection) -> str:
        radius = get_dedup_radius_m()
        bx, by, bz = _bucket(detection.position, radius)
        short = uuid.uuid4().hex[:8]
        return f"{detection.class_name}:{bx}:{by}:{bz}:{short}"

    def _metadata_for_insert(
        self, detection: Detection, frame_ref: Optional[str]
    ) -> dict[str, Any]:
        dim = len(detection.embedding)
        model_name = self._embedder.model_name if self._embedder is not None else "external"
        return {
            "class_name": detection.class_name,
            "class_id": -1 if detection.class_id is None else int(detection.class_id),
            "first_seen_ts": detection.ts,
            "last_seen_ts": detection.ts,
            "sightings": 1,
            "x": float(detection.position[0]),
            "y": float(detection.position[1]),
            "z": float(detection.position[2]),
            "position_frame": "map",
            "position_conf": float(detection.confidence),
            "caption": detection.caption,
            "bbox_last": json.dumps(list(detection.bbox_xyxy)),
            "frame_ref": frame_ref or "",
            "embedding_model": model_name,
            "embedding_dim": dim,
        }

    def _maybe_archive(
        self,
        detection: Detection,
        source_frame: Optional[Image.Image],
        new_id: str,
    ) -> Optional[str]:
        if self._frames_dir is None or source_frame is None:
            return detection.frame_ref
        # Filename: {ts}_{class}_{id8}.jpg
        ts_compact = time.strftime(
            "%Y%m%dT%H%M%SZ", time.gmtime(detection.ts)
        )
        short = new_id.split(":")[-1]
        fname = f"{ts_compact}_{detection.class_name}_{short}.jpg"
        path = self._frames_dir / fname
        try:
            source_frame.save(path, format="JPEG", quality=85)
            return str(path)
        except OSError as e:
            _log.warning("frame archive failed: %s", e)
            return detection.frame_ref

    @staticmethod
    def _row_to_entry(
        entry_id: str,
        meta: dict,
        document: str,
        embedding: Optional[list[float]],
        distance: Optional[float] = None,
    ) -> SceneEntry:
        bbox_raw = meta.get("bbox_last") or "[0,0,0,0]"
        try:
            bbox_list = json.loads(bbox_raw)
            bbox_tuple = (
                int(bbox_list[0]),
                int(bbox_list[1]),
                int(bbox_list[2]),
                int(bbox_list[3]),
            )
        except (json.JSONDecodeError, ValueError, IndexError, TypeError):
            bbox_tuple = (0, 0, 0, 0)

        class_id = meta.get("class_id")
        if class_id == -1:
            class_id = None

        frame_ref = meta.get("frame_ref") or None

        emb_tuple = tuple(embedding) if embedding is not None else ()
        return SceneEntry(
            id=entry_id,
            class_name=str(meta.get("class_name", "")),
            class_id=class_id,
            position=(
                float(meta.get("x", 0.0)),
                float(meta.get("y", 0.0)),
                float(meta.get("z", 0.0)),
            ),
            position_frame=str(meta.get("position_frame", "map")),
            position_conf=float(meta.get("position_conf", 0.0)),
            caption=str(meta.get("caption", "")),
            bbox_last_xyxy=bbox_tuple,
            frame_ref=frame_ref,
            first_seen_ts=float(meta.get("first_seen_ts", 0.0)),
            last_seen_ts=float(meta.get("last_seen_ts", 0.0)),
            sightings=int(meta.get("sightings", 1)),
            embedding=emb_tuple,
            embedding_model=str(meta.get("embedding_model", "")),
            embedding_dim=int(meta.get("embedding_dim", len(emb_tuple))),
            distance=distance,
        )

    def _unpack_query(self, result: dict) -> list[SceneEntry]:
        ids = result["ids"][0]
        metas = result["metadatas"][0]
        docs = result["documents"][0]
        embs_top = result.get("embeddings")
        dists_top = result.get("distances")
        if embs_top is None or len(embs_top) == 0:
            embs = [None] * len(ids)
        else:
            embs = list(embs_top[0])
        if dists_top is None or len(dists_top) == 0:
            dists = [None] * len(ids)
        else:
            dists = list(dists_top[0])
        out = []
        for cid, meta, doc, emb, dist in zip(ids, metas, docs, embs, dists):
            out.append(self._row_to_entry(cid, meta, doc, emb, dist))
        return out

    @staticmethod
    def _build_where(
        min_last_seen_ts: Optional[float],
        class_name: Optional[str],
    ) -> Optional[dict[str, Any]]:
        clauses: list[dict[str, Any]] = []
        if min_last_seen_ts is not None:
            clauses.append({"last_seen_ts": {"$gt": float(min_last_seen_ts)}})
        if class_name is not None:
            clauses.append({"class_name": class_name})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
