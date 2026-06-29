# `perception/` — long-term memory stores

Despite the name, this package is **not** the perception *pipeline*. The pipeline
that captures frames, detects, lifts to 3D, associates, captions, and embeds lives in
[`services/walkie_graphs/`](../services/walkie_graphs/); its 3D scene-graph store is
the numpy `SceneStore` (`services/walkie_graphs/scene.py`), which uses **no** ChromaDB.

This package holds the **face-recognition persistence layer**:

| File | What it is |
|------|------------|
| `vector_db.py` | Shared ChromaDB plumbing (`make_client`, `get_collection`, row get/query helpers). |
| `people_store.py` | `PeopleStore` — face-keyed people memory with two-modality (face + attire) re-ID fusion. Used by the HRI task (`tasks/HRI/`), **not** wired into `main.py`'s production loop. |

Kept under the historical `perception/` name to avoid churning imports across
`tasks/HRI` and the tests. Think of it as "the people-memory store."

> Single-process invariant: ChromaDB's `PersistentClient` is not safe for
> concurrent multi-process access — don't open a `PeopleStore` directory from two
> processes at once.
