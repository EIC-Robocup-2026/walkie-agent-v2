"""Compatibility shim — moved to :mod:`walkie_world.map.locations`.

Thin module alias kept during the walkie_world migration so existing imports
(``from tasks.skills.locations import resolve_pose, get_location_book, ...``) and
test monkeypatches (``tasks.skills.locations.get_location_book``) keep working —
aliasing the module object shares its ``_CACHE`` state and patch points with the
real module. Removed in the final phase.
"""

import sys

import walkie_world.map.locations as _impl

sys.modules[__name__] = _impl
