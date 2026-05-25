"""Diagnose desync / corruption in Walkie's ChromaDB vector stores (read-only).

Concurrent multi-process access to a ChromaDB persist dir — typically the robot
(``main.py``'s ScenePerceptionService) writing while ``tools/chroma_viewer``
reads the *same* directory — corrupts the HNSW index so it no longer agrees with
the metadata. The two symptoms are:

  * ``InternalError: Error finding id`` when a record's vector is fetched, and
  * vector queries returning ids / rankings that don't match the stored records.

This tool reports how bad the damage is, without changing anything:

  * per-collection record counts,
  * **dangling vectors** — ids whose embedding can't be read (the "Error finding
    id" set), found by bisection,
  * **caption desync** — for the scene store, ids in ``scene_captions`` with no
    matching ``scene_entries`` row (orphans) and ``scene_entries`` rows missing
    from the caption index (which `text_query` therefore can never return).

    uv run python -m tools.db_doctor --scene     # CLIP scene memory (default)
    uv run python -m tools.db_doctor --object    # legacy object DB
    uv run python -m tools.db_doctor --all
    uv run python -m tools.db_doctor --dirs path/to/chroma_db_scene

It opens a **snapshot copy** of each directory (never the live files), so it is
safe to run even if you're unsure the robot is stopped — and the snapshot can't
be skewed by an in-flight write the way the live store can. Paths default to the
same config the robot uses (SCENE_CHROMA_DIR / CHROMA_DIR).
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from perception.store import SceneStore
from walkie_config import load_config

_TMPDIRS: list[str] = []


@atexit.register
def _cleanup() -> None:
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


def _open_snapshot(directory: str) -> Optional[chromadb.api.ClientAPI]:
    """Copy ``directory`` to a temp dir and open a client on the copy.

    Returns None if the directory is absent. Reading a snapshot guarantees we
    never write to (or read a half-written) live store.
    """
    src = Path(directory).resolve()
    if not src.exists():
        print(f"  (directory absent: {src})")
        return None
    tmp = tempfile.mkdtemp(prefix="db_doctor_")
    _TMPDIRS.append(tmp)
    dst = Path(tmp) / src.name
    shutil.copytree(src, dst)
    return chromadb.PersistentClient(
        path=str(dst), settings=Settings(anonymized_telemetry=False)
    )


def _all_ids(coll) -> list[str]:
    return list(coll.get(include=[])["ids"])


def _find_dangling(coll, ids: list[str]) -> list[str]:
    """Ids whose embedding can't be read (raises on get). Bisection: O(k·log n)."""
    if not ids:
        return []
    try:
        coll.get(ids=ids, include=["embeddings"])
        return []
    except Exception:  # noqa: BLE001 — narrow down which id(s) are bad
        if len(ids) == 1:
            return list(ids)
        mid = len(ids) // 2
        return _find_dangling(coll, ids[:mid]) + _find_dangling(coll, ids[mid:])


def _count_null_embeddings(coll, ids: list[str]) -> int:
    """Among readable ids, how many have a missing/empty embedding."""
    if not ids:
        return 0
    res = coll.get(ids=ids, include=["embeddings"])
    embs = res.get("embeddings")
    try:
        if embs is None or len(embs) == 0:
            return len(ids)
    except TypeError:
        return len(ids)
    null = 0
    for e in embs:
        try:
            if e is None or len(e) == 0:
                null += 1
        except TypeError:
            null += 1
    return null


def _diagnose_collection(coll, name: str) -> dict:
    print(f"  · {name}: count={coll.count()}")
    ids = _all_ids(coll)
    dangling = _find_dangling(coll, ids)
    readable = [i for i in ids if i not in set(dangling)]
    null_emb = _count_null_embeddings(coll, readable)
    if dangling:
        print(f"      DANGLING vectors (unreadable embedding): {len(dangling)}")
        for d in dangling[:10]:
            print(f"        - {d}")
        if len(dangling) > 10:
            print(f"        … and {len(dangling) - 10} more")
    if null_emb:
        print(f"      records with NULL embedding: {null_emb}")
    if not dangling and not null_emb:
        print("      embeddings: all readable ✓")
    return {"ids": set(ids), "dangling": set(dangling), "null_emb": null_emb}


def _diagnose_dir(directory: str) -> dict:
    print(f"\n[{directory}]")
    client = _open_snapshot(directory)
    if client is None:
        return {"present": False}
    colls = {c.name: client.get_collection(c.name) for c in client.list_collections()}
    if not colls:
        print("  (no collections)")
        return {"present": True, "empty": True}

    stats = {name: _diagnose_collection(coll, name) for name, coll in colls.items()}

    # Scene-store cross-check: scene_entries ⟷ scene_captions share one id space.
    ent, cap = SceneStore.COLLECTION, SceneStore.CAPTION_COLLECTION
    desync = {}
    if ent in stats and cap in stats:
        entry_ids, cap_ids = stats[ent]["ids"], stats[cap]["ids"]
        orphans = cap_ids - entry_ids      # caption → no record (text_query skips)
        missing = entry_ids - cap_ids      # record → no caption (text_query can't find)
        desync = {"orphans": orphans, "missing": missing}
        print(f"  · caption index vs entries:")
        print(f"      caption orphans (caption with no entry): {len(orphans)}")
        print(f"      entries missing from caption index:      {len(missing)}")
        if missing:
            print("        → these records are INVISIBLE to text_query "
                  "(the 'query doesn't match DB' symptom)")
    return {"present": True, "stats": stats, "desync": desync}


def _recommend(results: dict[str, dict]) -> None:
    print("\n" + "=" * 60)
    any_dangling = any(
        any(s["dangling"] for s in r.get("stats", {}).values())
        for r in results.values() if r.get("present")
    )
    any_missing = any(
        bool(r.get("desync", {}).get("missing")) for r in results.values() if r.get("present")
    )
    any_orphan = any(
        bool(r.get("desync", {}).get("orphans")) for r in results.values() if r.get("present")
    )
    print("Recommendation:")
    if any_dangling:
        print("  ✗ Dangling vectors found — the HNSW index is corrupt and cannot be")
        print("    patched in place. Rebuild from archived frames (re-embed) or wipe:")
        print("      uv run python -m tools.reset_db --scene   # then re-explore")
        print("    (run with the robot AND viewer stopped).")
    elif any_missing:
        print("  ! No corruption, but records are missing from the caption index, so")
        print("    text_query / find_object can't see them. Backfill it in place:")
        print("      SCENE_REINDEX_CAPTIONS=1 uv run python main.py   # one run")
        print("    (this calls store.reindex_captions() at startup — safe, idempotent).")
    elif any_orphan:
        print("  ! Only caption orphans (stale captions pointing at deleted records).")
        print("    Harmless to queries (joined-out), cleaned up by the next prune.")
    else:
        print("  ✓ No desync or dangling vectors detected. The stores look healthy.")
        print("    If queries still look wrong, the corruption may be in HNSW ordering")
        print("    only — re-run after stopping the robot, or rebuild to be sure.")
    print("=" * 60)


def main() -> None:
    load_dotenv()
    load_config()
    ap = argparse.ArgumentParser(
        description="Diagnose ChromaDB desync/corruption (read-only snapshot)."
    )
    ap.add_argument("--scene", action="store_true", help="CLIP scene memory (SCENE_CHROMA_DIR)")
    ap.add_argument("--object", action="store_true", help="legacy object DB (CHROMA_DIR)")
    ap.add_argument("--all", action="store_true", help="both stores")
    ap.add_argument("--dirs", default="", help="comma-separated dirs to check (overrides the above)")
    args = ap.parse_args()

    if args.dirs:
        dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    else:
        dirs = []
        if args.object or args.all:
            dirs.append(os.getenv("CHROMA_DIR", "chroma_db"))
        if args.scene or args.all or not (args.object or args.all):
            dirs.append(os.getenv("SCENE_CHROMA_DIR", "chroma_db_scene"))

    print(f"[db-doctor] checking (read-only snapshot): {dirs}")
    results = {d: _diagnose_dir(d) for d in dirs}
    _recommend(results)


if __name__ == "__main__":
    main()
