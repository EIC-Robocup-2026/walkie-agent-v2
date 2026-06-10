"""Wipe the walkie_graphs store for a clean slate.

Removes the Chroma node DB, the per-object point-cloud sidecars, the thumbnails,
and the edge file (paths come from the WALKIE_GRAPHS_* config). Run with the robot
and any viewer stopped — ChromaDB's persistent client is single-process.

    uv run python -m walkie_graphs.tools.reset       # asks for confirmation
    uv run python -m walkie_graphs.tools.reset -y     # no confirmation
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
        Path(os.getenv("WALKIE_GRAPHS_CHROMA_DIR", "chroma_db_graph")),
        Path(os.getenv("WALKIE_GRAPHS_PCDS_DIR", "graph_pcds")),
        Path(os.getenv("WALKIE_GRAPHS_THUMBS_DIR", "graph_thumbs")),
    ]
    edges = Path(os.getenv("WALKIE_GRAPHS_EDGES_PATH", "graph_edges.json"))

    targets = [d for d in dirs if d.exists()] + ([edges] if edges.exists() else [])
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
    if edges.exists():
        edges.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
