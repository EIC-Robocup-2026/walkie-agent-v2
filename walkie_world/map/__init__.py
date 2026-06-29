"""Static arena map: rooms / locations / doors (waypoints + shapes), object
vocabulary grounding, and pure-numpy polygon helpers. No robot, LLM, or network.
"""

from walkie_world.map.locations import (
    Door,
    Location,
    LocationBook,
    MapObject,
    Pose,
    Room,
    build_doors,
    build_rooms_locations,
    get_location_book,
    load_location_book,
    resolve_pose,
)
from walkie_world.map.polygon import bbox_to_polygon, point_in_polygon, polygon_centroid
from walkie_world.map.vocab import WorldModel, load_world

__all__ = [
    "Door",
    "Location",
    "LocationBook",
    "MapObject",
    "Pose",
    "Room",
    "WorldModel",
    "bbox_to_polygon",
    "build_doors",
    "build_rooms_locations",
    "get_location_book",
    "load_location_book",
    "load_world",
    "point_in_polygon",
    "polygon_centroid",
    "resolve_pose",
]
