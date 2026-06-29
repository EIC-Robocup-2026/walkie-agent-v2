"""Compatibility shim — moved to :mod:`walkie_world.scene.store`.

Thin module alias kept during the walkie_world migration so existing imports
(``from services.walkie_graphs.scene import SceneStore, ObjectNode, ...``) keep
working. Removed in the final migration phase. Aliasing the module object (rather
than re-exporting names) keeps any module state and monkeypatching shared.
"""

import sys

import walkie_world.scene.store as _impl

sys.modules[__name__] = _impl
