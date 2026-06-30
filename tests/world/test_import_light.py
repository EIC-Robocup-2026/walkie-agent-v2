"""walkie_world must stay import-light: no chromadb/Open3D/torch/SDK on bare import,
and chromadb only once a people method is actually used.

Run this test in a FRESH interpreter (subprocess) so other tests that already
imported chromadb don't pollute sys.modules.
"""

from __future__ import annotations

import subprocess
import sys

_PROG = r"""
import sys
import walkie_world
heavy = [m for m in ("chromadb", "open3d", "torch", "walkie_sdk") if m in sys.modules]
assert not heavy, f"bare import walkie_world pulled: {heavy}"

import os, tempfile
d = tempfile.mkdtemp()
w = walkie_world.WalkieWorld(scene_dir=os.path.join(d, "scene"), enable_people=True)
assert "chromadb" not in sys.modules, "chromadb loaded before any people method"

# First people access loads chromadb (lazily).
_ = w.people_count()
assert "chromadb" in sys.modules, "chromadb not loaded after a people method"
print("OK")
"""


def test_import_light_in_subprocess():
    out = subprocess.run(
        [sys.executable, "-c", _PROG], capture_output=True, text=True
    )
    assert out.returncode == 0, f"stdout={out.stdout!r}\nstderr={out.stderr[-2000:]!r}"
    assert out.stdout.strip().endswith("OK")
