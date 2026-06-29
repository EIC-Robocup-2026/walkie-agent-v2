"""Compatibility shim — moved to :mod:`walkie_world.people.vector_db`.

Thin module alias kept during the walkie_world migration; removed in the final
phase. See :mod:`walkie_world.people.vector_db` for the ChromaDB plumbing.
"""

import sys

import walkie_world.people.vector_db as _impl

sys.modules[__name__] = _impl
