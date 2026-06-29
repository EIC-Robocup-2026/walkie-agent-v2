"""Compatibility shim — moved to :mod:`walkie_world.scene.relations`.

Thin module alias kept during the walkie_world migration; removed in the final
phase. See :mod:`walkie_world.scene.relations` for ``derive_relations``.
"""

import sys

import walkie_world.scene.relations as _impl

sys.modules[__name__] = _impl
