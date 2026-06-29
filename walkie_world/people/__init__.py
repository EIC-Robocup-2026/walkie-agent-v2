"""Face + appearance person memory (ChromaDB-backed).

Importing this sub-package loads ``chromadb`` (via :mod:`walkie_world.people.store`),
so the :class:`~walkie_world.world.WalkieWorld` facade imports it lazily — only when
a people method is first used — to keep ``import walkie_world`` light.
"""

from walkie_world.people.store import FUSION_DEFAULTS, PeopleStore, PersonRecord

__all__ = ["FUSION_DEFAULTS", "PeopleStore", "PersonRecord"]
