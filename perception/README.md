# `perception/` — long-term memory stores

Despite the name, this package is **not** the perception *pipeline*. The pipeline
that captures frames, detects, lifts to 3D, fuses, captions, and embeds lives in
[`services/walkie_graphs/`](../services/walkie_graphs/). The 3D scene-graph store
itself is `GraphMemory` in `services/walkie_graphs/memory.py`.

This package holds the **persistence layer** shared across the app:

| File | What it is |
|------|------------|
| `vector_db.py` | Shared ChromaDB plumbing (`make_client`, `get_collection`, row get/query helpers). Imported by both `walkie_graphs/memory.py` and `people_store.py`. |
| `people_store.py` | `PeopleStore` — face-keyed people memory with two-modality (face + attire) re-ID fusion. Used by the HRI task (`tasks/HRI/`), **not** wired into `main.py`'s production loop. |

Kept under the historical `perception/` name to avoid churning imports across
`walkie_graphs`, `tasks/HRI`, and the tests. Think of it as "the stores layer."

> Single-process invariant: ChromaDB's `PersistentClient` is not safe for
> concurrent multi-process access. The `walkie_graphs` loop writes its store
> continuously while running — don't open the same directory from a second process.
