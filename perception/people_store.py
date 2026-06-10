"""Face-keyed people memory for HRI tasks (RoboCup @Home Receptionist).

Unlike :class:`~perception.store.SceneStore`, which is a **spatial** catalogue
deduplicated by 3D position, this store is keyed by **face identity**. People
move and may switch seats (the rulebook explicitly allows it and penalizes
mis-identification heavily), so position can't identify a guest — their face
embedding can. There is therefore **no spatial dedup and no location prune**
here; a person is one record, recalled by nearest face vector.

One ChromaDB collection (``people``), cosine space. Each record stores the
person's face embedding (the query vector) plus their ``name``, ``drink``, a
free-text ``attributes`` string, and provenance/bookkeeping in metadata.

The server (``/face-recognition/embed``) is stateless — all enrollment and
matching lives here. See ``docs/human_recognition_design.md`` (C2/C3).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import chromadb
from chromadb.config import Settings
from PIL import Image


@dataclass(frozen=True)
class PersonRecord:
    """A single enrolled person — read-only view from the store."""

    id: str
    name: str
    drink: str
    attributes: str
    embedding_model: str
    enrollments: int
    first_seen_ts: float
    last_seen_ts: float
    frame_ref: Optional[str] = None
    """Path to the archived face crop, if persisted (shown by the DB viewer)."""
    embedding: tuple[float, ...] = field(default_factory=tuple)
    distance: Optional[float] = None
    """Cosine distance to the query vector when returned from :meth:`recognize`
    (``0`` = identical, ``2`` = opposite). ``None`` for metadata reads."""

    @property
    def similarity(self) -> Optional[float]:
        """Cosine similarity (``1 - distance``) when this came from a query."""
        return None if self.distance is None else 1.0 - self.distance


def _slug(name: str) -> str:
    """Stable record id from a name: lowercased, non-alnum → single hyphen."""
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "unknown"


def _mean_unit(vectors: Sequence[Sequence[float]]) -> list[float]:
    """L2-normalized mean of (already unit-length) vectors — a stable centroid.

    Averaging multiple enrollment frames is more robust for re-ID than keeping a
    single shot. Returns the renormalized mean; falls back to the first vector
    if the mean is degenerate.
    """
    n = len(vectors)
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    norm = sum(x * x for x in acc) ** 0.5
    if norm <= 1e-12:
        return list(vectors[0])
    return [x / norm for x in acc]


class PeopleStore:
    """Read/write façade over the face-keyed people vector DB."""

    COLLECTION = "people"

    def __init__(
        self,
        *,
        persist_dir: str | Path | None = "chroma_db_people",
        embedding_model: str = "",
        frames_dir: str | Path | None = None,
        crop_margin: float = 0.25,
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
        self._embedding_model = embedding_model
        # When set, enroll archives the guest's face crop here so the DB viewer
        # can show who is remembered. Margin pads the bbox for a bit of context.
        self._frames_dir = Path(frames_dir) if frames_dir else None
        self._crop_margin = crop_margin
        if self._frames_dir:
            self._frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        """Underlying chromadb client (so an in-process viewer can read live)."""
        return self._client

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def enroll(
        self,
        name: str,
        drink: str,
        embedding: Sequence[float],
        *,
        attributes: str = "",
        frame: Optional[Image.Image] = None,
        face_bbox_xyxy: Optional[Sequence[int]] = None,
        ts: Optional[float] = None,
    ) -> PersonRecord:
        """Remember a guest, or refresh an existing one with the same name.

        Re-enrolling a known name updates their drink/attributes and folds the
        new face vector into a running centroid (more robust recognition across
        lighting/pose), bumping the enrollment count. When ``frame`` (and a
        ``face_bbox_xyxy``) are given and a frames dir is configured, the guest's
        face crop is archived for the DB viewer. Returns the stored record.
        """
        if not name or not name.strip():
            raise ValueError("name must be non-empty")
        emb = [float(x) for x in embedding]
        if not emb:
            raise ValueError("embedding must be non-empty")
        now = time.time() if ts is None else float(ts)
        rid = _slug(name)

        existing = self._raw_get(rid)
        if existing is not None:
            prev_emb, meta = existing
            new_emb = _mean_unit([prev_emb, emb]) if prev_emb else emb
            metadata = {
                "name": name.strip(),
                "drink": drink.strip(),
                "attributes": attributes.strip() or str(meta.get("attributes", "")),
                "embedding_model": self._embedding_model or str(meta.get("embedding_model", "")),
                "enrollments": int(meta.get("enrollments", 1)) + 1,
                "first_seen_ts": float(meta.get("first_seen_ts", now)),
                "last_seen_ts": now,
            }
            prev_ref = meta.get("frame_ref")
            if prev_ref:
                metadata["frame_ref"] = str(prev_ref)
        else:
            new_emb = emb
            metadata = {
                "name": name.strip(),
                "drink": drink.strip(),
                "attributes": attributes.strip(),
                "embedding_model": self._embedding_model,
                "enrollments": 1,
                "first_seen_ts": now,
                "last_seen_ts": now,
            }
        # Archive / refresh the face crop for the viewer (latest sighting wins).
        ref = self._archive_face(rid, frame, face_bbox_xyxy)
        if ref is not None:
            metadata["frame_ref"] = ref
        document = f"{metadata['name']} — likes {metadata['drink']}".strip(" —")
        self._collection.upsert(
            ids=[rid], embeddings=[new_emb], metadatas=[metadata], documents=[document]
        )
        return self._to_record(rid, new_emb, metadata)

    def _archive_face(
        self,
        rid: str,
        frame: Optional[Image.Image],
        bbox_xyxy: Optional[Sequence[int]],
    ) -> Optional[str]:
        """Save a padded face crop to the frames dir; return its path (or None)."""
        if self._frames_dir is None or frame is None or not bbox_xyxy:
            return None
        try:
            w, h = frame.size
            x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            mx, my = int(bw * self._crop_margin), int(bh * self._crop_margin)
            crop = frame.crop(
                (max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my))
            )
            path = self._frames_dir / f"{rid}.jpg"
            crop.convert("RGB").save(path, format="JPEG", quality=85)
            return str(path)
        except Exception:  # noqa: BLE001 — a thumbnail must never break enrollment
            return None

    def clear(self) -> None:
        """Forget everyone (e.g. between Receptionist runs)."""
        self._client.delete_collection(self.COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recognize(
        self, embedding: Sequence[float], *, max_distance: float = 0.4
    ) -> Optional[PersonRecord]:
        """Return the closest enrolled person, or ``None`` if none is close.

        ``max_distance`` is a cosine distance (``1 - similarity``); a match must
        be at least that close. Tune via ``FACE_MATCH_THRESHOLD``.
        """
        emb = [float(x) for x in embedding]
        if self.count() == 0 or not emb:
            return None
        res = self._collection.query(
            query_embeddings=[emb],
            n_results=1,
            include=["embeddings", "metadatas", "distances"],
        )
        ids = res.get("ids") or [[]]
        if not ids[0]:
            return None
        dist = float(res["distances"][0][0])
        if dist > max_distance:
            return None
        embs = res.get("embeddings")
        stored_emb = embs[0][0] if embs is not None and len(embs) and len(embs[0]) else None
        metas = res.get("metadatas")
        meta = (metas[0][0] if metas is not None and len(metas) and len(metas[0]) else {}) or {}
        return self._to_record(ids[0][0], stored_emb, meta, distance=dist)

    def get(self, name: str) -> Optional[PersonRecord]:
        """Look a person up by name (exact, case-insensitive via the slug)."""
        raw = self._raw_get(_slug(name))
        if raw is None:
            return None
        emb, meta = raw
        return self._to_record(_slug(name), emb, meta)

    def list_people(self) -> list[PersonRecord]:
        """Everyone enrolled, most-recently-seen first."""
        res = self._collection.get(include=["embeddings", "metadatas"])
        ids = res.get("ids") or []
        embs = res.get("embeddings")
        metas = res.get("metadatas")
        records = []
        for i, rid in enumerate(ids):
            emb = embs[i] if embs is not None and i < len(embs) else None
            meta = (metas[i] if metas is not None and i < len(metas) else {}) or {}
            records.append(self._to_record(rid, emb, meta))
        return sorted(records, key=lambda r: r.last_seen_ts, reverse=True)

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _raw_get(self, rid: str):
        """``(embedding, metadata)`` for an id, or ``None`` if absent."""
        res = self._collection.get(ids=[rid], include=["embeddings", "metadatas"])
        ids = res.get("ids") or []
        if not ids:
            return None
        embs = res.get("embeddings")
        metas = res.get("metadatas")
        emb = embs[0] if embs is not None and len(embs) else None
        meta = (metas[0] if metas is not None and len(metas) else {}) or {}
        emb = list(emb) if emb is not None else []
        return emb, meta

    def _to_record(
        self, rid: str, embedding, meta: dict, *, distance: Optional[float] = None
    ) -> PersonRecord:
        emb = tuple(float(x) for x in embedding) if embedding is not None else tuple()
        ref = meta.get("frame_ref")
        return PersonRecord(
            id=rid,
            name=str(meta.get("name", "")),
            drink=str(meta.get("drink", "")),
            attributes=str(meta.get("attributes", "")),
            embedding_model=str(meta.get("embedding_model", "")),
            enrollments=int(meta.get("enrollments", 1)),
            first_seen_ts=float(meta.get("first_seen_ts", 0.0)),
            last_seen_ts=float(meta.get("last_seen_ts", 0.0)),
            frame_ref=str(ref) if ref else None,
            embedding=emb,
            distance=distance,
        )
