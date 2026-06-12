"""Face-keyed people memory for HRI tasks (RoboCup @Home Receptionist).

Unlike the spatial object stores (deduplicated by 3D position), this store is
keyed by **face identity**. People move and may switch seats (the rulebook
explicitly allows it and penalizes mis-identification heavily), so position
can't identify a guest — their face embedding can. There is therefore **no
spatial dedup and no location prune** here; a person is one record, recalled
by nearest face vector.

Two ChromaDB collections, one id space: ``people`` holds the face embedding
(the primary query vector) plus the person's ``name``, ``drink``, free-text
``attributes`` and provenance in metadata; ``people_appearance`` holds an
optional **appearance** (attire/body) embedding for the same id, so a guest
can still be re-identified when their face is not visible (turned away, far,
occluded). Faces fold into a running centroid across enrollments; appearance
is latest-wins, because clothing is session-specific. :meth:`recognize_fused`
combines the two modalities with adaptive weighting by face-detection
confidence.

The two-modality fusion design is by **Chalk (EIC team)** — adopted from the
``eic-human`` subproject (``eic_human/core.py::_fuse_score`` and
``pipeline/store.py``); this file is a port of his ``PeopleStore`` from the
``feat/appearance-reid-chalk`` branch, re-homed onto the shared
:mod:`perception.vector_db` ChromaDB layer and extended with caller-chosen
record ids (``person_id``) so a guest whose name was never understood can
still be enrolled under a stable key ("guest-1").

The server routes (``/face-recognition/embed``, ``/appearance/embed``) are
stateless — all enrollment and matching lives here.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from .vector_db import drop_collection, get_collection, get_rows, make_client, query_rows


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
    (``0`` = identical, ``2`` = opposite). For :meth:`recognize_fused` it is
    ``1 - fused score``, so :attr:`similarity` is the fused score either way.
    ``None`` for metadata reads."""
    matched_by: Optional[str] = None
    """Which modality produced a :meth:`recognize_fused` match —
    ``"face+appearance"``, ``"face"`` or ``"appearance"``. ``None`` otherwise."""

    @property
    def similarity(self) -> Optional[float]:
        """Cosine similarity (``1 - distance``) when this came from a query."""
        return None if self.distance is None else 1.0 - self.distance


def _slug(name: str) -> str:
    """Stable record id from a name: lowercased, non-alnum → single hyphen."""
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "unknown"


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (norm-guarded; inputs are ~unit length)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-8)


# Adaptive face↔appearance fusion weights — design by Chalk (EIC team), from
# eic_human/core.py. A confident face detection is trusted mostly on the face;
# a marginal one leans harder on attire; no face at all → appearance only.
# The appearance weight is always the face weight's complement.
FUSION_DEFAULTS = {
    "face_conf_high": 0.8,    # det_score above this → "high confidence" face
    "face_conf_med": 0.5,     # det_score above this → "medium confidence" face
    "face_weight_high": 0.75,
    "face_weight_med": 0.55,
}


def _mean_unit(vectors: Sequence[Sequence[float]]) -> list[float]:
    """L2-normalized mean of (already unit-length) vectors — a stable centroid.

    Averaging multiple enrollment frames is more robust for re-ID than keeping a
    single shot. Returns the renormalized mean; falls back to the first vector
    if the mean is degenerate.
    """
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
    APP_COLLECTION = "people_appearance"

    def __init__(
        self,
        *,
        persist_dir: str | Path | None = "chroma_db_people",
        embedding_model: str = "",
        frames_dir: str | Path | None = None,
        crop_margin: float = 0.25,
    ) -> None:
        self._client = make_client(persist_dir)
        self._collection = get_collection(
            self._client, self.COLLECTION, unique_if_ephemeral=True
        )
        self._app_collection = get_collection(
            self._client, self.APP_COLLECTION, unique_if_ephemeral=True
        )
        self._embedding_model = embedding_model
        # When set, enroll archives the guest's face crop here so the DB viewer
        # can show who is remembered. Margin pads the bbox for a bit of context.
        self._frames_dir = Path(frames_dir) if frames_dir else None
        self._crop_margin = crop_margin
        if self._frames_dir:
            self._frames_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, *, embedding_model: str = "") -> "PeopleStore":
        """Build from the PEOPLE_* environment (config.toml ``[people]`` table)."""
        return cls(
            persist_dir=os.getenv("PEOPLE_CHROMA_DIR", "chroma_db_people"),
            embedding_model=embedding_model,
            frames_dir=os.getenv("PEOPLE_FRAMES_DIR", "people_frames") or None,
        )

    @staticmethod
    def face_match_max_distance() -> float:
        """Configured cosine-distance gate for :meth:`recognize`."""
        return float(os.getenv("FACE_MATCH_THRESHOLD", "0.4"))

    @staticmethod
    def fused_min_score() -> float:
        """Configured minimum fused similarity for :meth:`recognize_fused`."""
        return float(os.getenv("APPEARANCE_MATCH_THRESHOLD", "0.5"))

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
        person_id: Optional[str] = None,
        attributes: str = "",
        app_embedding: Optional[Sequence[float]] = None,
        frame: Optional[Image.Image] = None,
        face_bbox_xyxy: Optional[Sequence[int]] = None,
        ts: Optional[float] = None,
    ) -> PersonRecord:
        """Remember a guest, or refresh an existing one with the same id.

        The record id is ``person_id`` (slugged) when given, else the slug of
        ``name`` — pass a stable ``person_id`` ("guest-1") when the name may be
        unknown or mis-heard, so recognition keys never depend on STT.
        Re-enrolling a known id updates their name/drink/attributes and folds
        the new face vector into a running centroid (more robust recognition
        across lighting/pose), bumping the enrollment count. ``app_embedding``
        (the OSNet attire/body vector) is stored latest-wins — clothing changes
        between sessions, so averaging it would blur identities. When ``frame``
        (and a ``face_bbox_xyxy``) are given and a frames dir is configured, the
        guest's face crop is archived for the DB viewer. Returns the stored
        record.
        """
        if person_id is None and not (name and name.strip()):
            raise ValueError("name must be non-empty when no person_id is given")
        emb = [float(x) for x in embedding]
        if not emb:
            raise ValueError("embedding must be non-empty")
        now = time.time() if ts is None else float(ts)
        rid = _slug(person_id) if person_id else _slug(name)

        existing = self._raw_get(rid)
        if existing is not None:
            prev_emb, meta = existing
            new_emb = _mean_unit([prev_emb, emb]) if prev_emb else emb
            metadata = {
                "name": name.strip() or str(meta.get("name", "")),
                "drink": drink.strip() or str(meta.get("drink", "")),
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
            ids=[rid], embeddings=[new_emb], metadatas=[metadata], documents=[document or rid]
        )
        if app_embedding is not None:
            app = [float(x) for x in app_embedding]
            if app:
                self._app_collection.upsert(
                    ids=[rid], embeddings=[app], documents=[document or rid]
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
        """Forget everyone (e.g. between Receptionist runs) — both collections."""
        for attr in ("_collection", "_app_collection"):
            col = getattr(self, attr)
            name = col.name  # actual name (may carry an ephemeral suffix)
            drop_collection(self._client, name)
            setattr(self, attr, get_collection(self._client, name))

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
        rows = query_rows(res)
        if not rows:
            return None
        rid, stored_emb, meta, dist = rows[0]
        if dist is None or dist > max_distance:
            return None
        return self._to_record(rid, stored_emb, meta, distance=dist)

    def recognize_fused(
        self,
        face_embedding: Optional[Sequence[float]] = None,
        app_embedding: Optional[Sequence[float]] = None,
        *,
        face_confidence: float = 0.0,
        min_score: float = 0.5,
        fusion: Optional[dict] = None,
    ) -> Optional[PersonRecord]:
        """Two-modality recognition: face + appearance, adaptively fused.

        Scores every enrolled person on both modalities and combines them with
        confidence-adaptive weights (design by Chalk, EIC team): a confident
        face detection (``face_confidence`` above ``face_conf_high``) trusts
        the face at ``face_weight_high``; a marginal one (above
        ``face_conf_med``) at ``face_weight_med``; with no usable face the
        match is appearance-only. A modality missing on either side (no query
        vector, or the person was enrolled without one) falls back to the
        available modality alone rather than scoring it as zero.

        The store is tiny (a handful of Receptionist guests), so this scans
        all records exactly instead of merging two approximate HNSW queries.

        Args:
            face_embedding: Query face vector, or ``None`` if no face visible.
            app_embedding: Query appearance (attire) vector, or ``None``.
            face_confidence: Face detection score in ``[0, 1]``.
            min_score: Minimum fused similarity to count as a match
                (Chalk's default 0.5; tune via ``APPEARANCE_MATCH_THRESHOLD``).
            fusion: Optional overrides for :data:`FUSION_DEFAULTS` keys.

        Returns:
            The best-matching person with :attr:`PersonRecord.matched_by` set,
            or ``None`` when nobody clears ``min_score``.
        """
        face = [float(x) for x in face_embedding] if face_embedding else None
        app = [float(x) for x in app_embedding] if app_embedding else None
        if (face is None and app is None) or self.count() == 0:
            return None
        cfg = {**FUSION_DEFAULTS, **(fusion or {})}

        rows = get_rows(self._collection.get(include=["embeddings", "metadatas"]))
        app_by_id = {
            rid: emb
            for rid, emb, _meta in get_rows(self._app_collection.get(include=["embeddings"]))
            if emb is not None
        }

        best: Optional[tuple[float, str, list | None, dict, str]] = None
        for rid, stored_face, meta in rows:
            # A zero-norm stored face means "enrolled without a face" (an
            # attire-only sighting) — treat it as absent, not as similarity 0,
            # so the appearance modality carries the match alone.
            face_sim = (
                _cosine_sim(face, stored_face)
                if face is not None and stored_face and any(stored_face)
                else None
            )
            stored_app = app_by_id.get(rid)
            app_sim = (
                _cosine_sim(app, stored_app)
                if app is not None and stored_app is not None
                else None
            )

            if face_sim is not None and app_sim is not None:
                if face_confidence > cfg["face_conf_high"]:
                    w = cfg["face_weight_high"]
                elif face_confidence > cfg["face_conf_med"]:
                    w = cfg["face_weight_med"]
                else:
                    w = 0.0
                score = w * face_sim + (1.0 - w) * app_sim
                matched_by = "face+appearance" if w > 0.0 else "appearance"
            elif face_sim is not None:
                score = face_sim
                matched_by = "face"
            elif app_sim is not None:
                score = app_sim
                matched_by = "appearance"
            else:
                continue

            if score >= min_score and (best is None or score > best[0]):
                best = (score, rid, stored_face, meta, matched_by)

        if best is None:
            return None
        score, rid, stored_face, meta, matched_by = best
        return self._to_record(
            rid, stored_face, meta, distance=1.0 - score, matched_by=matched_by
        )

    def get(self, name_or_id: str) -> Optional[PersonRecord]:
        """Look a person up by name or record id (case-insensitive via the slug)."""
        rid = _slug(name_or_id)
        raw = self._raw_get(rid)
        if raw is None:
            return None
        emb, meta = raw
        return self._to_record(rid, emb, meta)

    def list_people(self) -> list[PersonRecord]:
        """Everyone enrolled, most-recently-seen first."""
        rows = get_rows(self._collection.get(include=["embeddings", "metadatas"]))
        records = [self._to_record(rid, emb, meta) for rid, emb, meta in rows]
        return sorted(records, key=lambda r: r.last_seen_ts, reverse=True)

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _raw_get(self, rid: str):
        """``(embedding, metadata)`` for an id, or ``None`` if absent."""
        rows = get_rows(self._collection.get(ids=[rid], include=["embeddings", "metadatas"]))
        if not rows:
            return None
        _rid, emb, meta = rows[0]
        return (emb if emb is not None else []), meta

    def _to_record(
        self,
        rid: str,
        embedding,
        meta: dict,
        *,
        distance: Optional[float] = None,
        matched_by: Optional[str] = None,
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
            matched_by=matched_by,
        )
