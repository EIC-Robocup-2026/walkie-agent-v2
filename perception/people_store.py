"""Compatibility shim — moved to :mod:`walkie_world.people.store`.

Thin module alias kept during the walkie_world migration so existing imports
(``from perception.people_store import PeopleStore, _cosine_sim, _mean_unit, ...``)
keep working. Removed in the final phase.
"""

import sys

import walkie_world.people.store as _impl

sys.modules[__name__] = _impl
