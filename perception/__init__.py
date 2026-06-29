"""Perception-side memory stores.

The people memory moved to :mod:`walkie_world.people`; this package re-exports it
for back-compat (``from perception import PeopleStore``). New code should import
from :mod:`walkie_world` instead.
"""

from walkie_world.people import FUSION_DEFAULTS, PeopleStore, PersonRecord

__all__ = ["FUSION_DEFAULTS", "PeopleStore", "PersonRecord"]
