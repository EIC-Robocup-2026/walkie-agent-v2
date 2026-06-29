"""Shared ChromaDB plumbing for every vector store in the repo.

Extracted from ``services/walkie_graphs/memory.py`` so the scene graph, the
people memory (:mod:`perception.people_store`), and future long-term memory
all build their clients/collections the same way instead of hand-rolling
``chromadb`` calls.

IMPORTANT — single-process only: ChromaDB's ``PersistentClient`` is not safe
for concurrent multi-process access. Opening a directory that another process
(e.g. the running robot) is writing corrupts the HNSW index. Tools that
inspect a live store must work on a snapshot copy (see ``tools/chroma_viewer``).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings

_SETTINGS = Settings(anonymized_telemetry=False, allow_reset=True)


def make_client(persist_dir: str | Path | None):
    """A ChromaDB client: persistent at *persist_dir*, in-memory when None.

    The in-memory variant (tests, throwaway runs) shares ONE database
    process-wide — give each store distinct collection names, or pass
    ``unique_if_ephemeral=True`` to :func:`get_collection`.
    """
    if persist_dir is None:
        return chromadb.EphemeralClient(settings=_SETTINGS)
    path = Path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path.resolve()), settings=_SETTINGS)


def is_ephemeral(client) -> bool:
    """True when *client* is an in-memory (non-persistent) client."""
    try:
        return not client.get_settings().is_persistent
    except Exception:
        return False


def get_collection(
    client,
    name: str,
    *,
    space: str = "cosine",
    unique_if_ephemeral: bool = False,
):
    """Get-or-create a collection with the given distance space.

    ``unique_if_ephemeral`` appends a random suffix on in-memory clients so
    parallel store instances (unit tests) don't share state through the
    process-wide ephemeral database.
    """
    if unique_if_ephemeral and is_ephemeral(client):
        name = f"{name}_{uuid.uuid4().hex[:8]}"
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": space})


def drop_collection(client, name: str) -> None:
    """Delete a collection, ignoring a missing one."""
    try:
        client.delete_collection(name)
    except Exception:
        pass


def get_rows(res: dict) -> list[tuple[str, list | None, dict]]:
    """Unwrap a ``collection.get(...)`` response into (id, embedding, metadata) rows.

    Centralizes the ``or []`` / index-bound guards every store repeats — any of
    embeddings/metadatas may be None or shorter than ids depending on
    ``include``.
    """
    ids = res.get("ids") or []
    embs = res.get("embeddings")
    metas = res.get("metadatas")
    rows = []
    for i, rid in enumerate(ids):
        emb = embs[i] if embs is not None and i < len(embs) else None
        meta = (metas[i] if metas is not None and i < len(metas) else {}) or {}
        rows.append((rid, list(emb) if emb is not None else None, meta))
    return rows


def query_rows(res: dict) -> list[tuple[str, list | None, dict, float | None]]:
    """Unwrap a single-query ``collection.query(...)`` response into
    (id, embedding, metadata, distance) rows (first query only)."""
    ids = (res.get("ids") or [[]])[0]
    embs = res.get("embeddings")
    metas = res.get("metadatas")
    dists = res.get("distances")
    rows = []
    for i, rid in enumerate(ids):
        emb = embs[0][i] if embs is not None and len(embs) and i < len(embs[0]) else None
        meta = (metas[0][i] if metas is not None and len(metas) and i < len(metas[0]) else {}) or {}
        dist = float(dists[0][i]) if dists is not None and len(dists) and i < len(dists[0]) else None
        rows.append((rid, list(emb) if emb is not None else None, meta, dist))
    return rows
