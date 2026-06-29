"""Compatibility shim — moved to :mod:`walkie_world.map.vocab`.

Thin module alias kept during the walkie_world migration so existing imports
(``from tasks.GPSR.world import WorldModel, load_world, _fuzzy_match, ...``) keep
working. Removed in the final phase.
"""

import sys

import walkie_world.map.vocab as _impl

sys.modules[__name__] = _impl
