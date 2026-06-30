"""Offline scene build from a saved snapshot buffer — the v2 dev-velocity harness.

Record one real robot run (the capture thread fills ``graph_buffer/``), then replay it
here as many times as you like to tune association / poses / TSDF **deterministically,
without the robot**. This is also the Stage-2 go/no-go measurement: compare
``--pose-mode baseline`` vs ``--pose-mode auto`` (and ``--tsdf``) on the *same* buffer.

    uv run python -m services.realtime_explore.tools.replay graph_buffer
    uv run python -m services.realtime_explore.tools.replay graph_buffer --pose-mode auto --tsdf
    uv run python -m services.realtime_explore.tools.replay graph_buffer --store graph_scene  # also persist

Builds object observations over the window and prints them; with ``--store`` it merges
into a real SceneStore (no embed server → keyword-only) and prints to_text_description.
"""

from __future__ import annotations

import argparse
import time

from dotenv import load_dotenv

from walkie_config import load_config
from walkie_world import WalkieWorld
from ..buffer import SnapshotBuffer
from ..builder import build_scene


def main(argv=None) -> int:
    load_dotenv()
    load_config()
    ap = argparse.ArgumentParser(description="Replay a snapshot buffer into an offline scene build.")
    ap.add_argument("buffer_dir", nargs="?", default="graph_buffer", help="snapshot buffer dir")
    ap.add_argument("--pose-mode", choices=["baseline", "auto"], default="baseline")
    ap.add_argument("--tsdf", action="store_true", help="also fuse the volumetric map")
    ap.add_argument("--window", type=int, default=0, help="newest N snapshots (0 = all)")
    ap.add_argument("--store", default=None, help="also merge+install into this SceneStore dir")
    args = ap.parse_args(argv)

    buf = SnapshotBuffer(args.buffer_dir)
    n = len(buf)
    if n == 0:
        print(f"No snapshots in {args.buffer_dir!r}.")
        return 1
    print(f"Loaded {n} snapshots from {args.buffer_dir!r}.")
    snaps = buf.load_window(None if args.window <= 0 else args.window)

    t0 = time.monotonic()
    res = build_scene(
        snaps, pose_mode=args.pose_mode, do_tsdf=args.tsdf, log=print,
    )
    dt = time.monotonic() - t0
    print(f"\nBuild ({args.pose_mode}, tsdf={args.tsdf}) over {res.n_snapshots} snapshots "
          f"/ {res.n_detections} lifts → {len(res.observations)} objects in {dt:.1f}s")
    by_class: dict[str, int] = {}
    for o in res.observations:
        by_class[o.class_name] = by_class.get(o.class_name, 0) + 1
    for cls, k in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {k}")
    for o in sorted(res.observations, key=lambda o: -o.n_obs)[:30]:
        x, y, z = o.centroid
        cap = f' "{o.captions[0]}"' if o.captions else ""
        print(f"  - {o.class_name}{cap} @ ({x:.2f}, {y:.2f}, {z:.2f}) seen {o.n_obs}x")
    if res.structural_cloud is not None:
        print(f"  structural cloud: {len(res.structural_cloud)} points")

    if args.store:
        # No embed server offline → keyword search only; objects-only world (no people).
        world = WalkieWorld(scene_dir=args.store, embed_text=None, enable_people=False)
        world.observe_objects(res.observations)
        print(f"\nMerged into {args.store!r} ({world.count()} total nodes):")
        print(world.to_text_description())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
