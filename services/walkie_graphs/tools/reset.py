"""Wipe the walkie_graphs store for a clean slate.

Removes the lean scene store (`graph_scene/` — nodes.json + embeddings.npy + edges.json
+ map.npz) and the snapshot ring buffer (`graph_buffer/`). Paths come from the
WALKIE_GRAPHS_* config. Run with the robot stopped so the capture/build threads aren't
writing concurrently.

    uv run python -m services.walkie_graphs.tools.reset       # asks for confirmation
    uv run python -m services.walkie_graphs.tools.reset -y     # no confirmation
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from walkie_config import load_config


def main() -> None:
    load_dotenv()
    load_config()

    ap = argparse.ArgumentParser(description="Wipe the walkie_graphs store.")
    ap.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    dirs = [
        Path(os.getenv("WALKIE_GRAPHS_STORE_DIR", "graph_scene")),    # nodes/embeddings/edges/map
        Path(os.getenv("WALKIE_GRAPHS_BUFFER_DIR", "graph_buffer")),  # snapshot ring buffer
    ]
    files: list[Path] = []

    targets = [d for d in dirs if d.exists()] + [f for f in files if f.exists()]
    if not targets:
        print("Nothing to remove — the walkie_graphs store is already empty.")
        return

    print("Will remove:")
    for t in targets:
        print(f"  - {t}")
    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
    for f in files:
        if f.exists():
            f.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
