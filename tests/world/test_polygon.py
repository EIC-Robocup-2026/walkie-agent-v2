"""Pure point-in-polygon + room_at geometry for the walkie_world map layer."""

from __future__ import annotations

import textwrap

import pytest

from walkie_world.map.locations import load_location_book
from walkie_world.map.polygon import bbox_to_polygon, point_in_polygon, polygon_centroid

_SQUARE = [[0, 0], [4, 0], [4, 4], [0, 4]]
# An L-shaped (concave) polygon.
_L = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]


def test_point_in_convex_polygon():
    assert point_in_polygon(2, 2, _SQUARE)
    assert not point_in_polygon(5, 2, _SQUARE)
    assert not point_in_polygon(-1, 2, _SQUARE)


def test_point_in_concave_polygon():
    assert point_in_polygon(1, 1, _L)       # in the filled corner
    assert point_in_polygon(1, 3, _L)       # in the tall arm
    assert not point_in_polygon(3, 3, _L)   # the notch is outside


def test_degenerate_polygon_is_outside():
    assert not point_in_polygon(0, 0, [])
    assert not point_in_polygon(0, 0, [[0, 0], [1, 1]])  # < 3 vertices


def test_polygon_centroid_and_bbox():
    assert polygon_centroid(_SQUARE) == (2.0, 2.0)
    assert polygon_centroid([]) is None
    poly = bbox_to_polygon(1, 2, 3, 5)
    assert poly == [[1.0, 2.0], [3.0, 2.0], [3.0, 5.0], [1.0, 5.0]]
    assert point_in_polygon(2, 3, poly)


def _book(tmp_path, toml: str):
    p = tmp_path / "world.toml"
    p.write_text(textwrap.dedent(toml))
    return load_location_book(p)


def test_room_at_point_in_polygon(tmp_path):
    book = _book(
        tmp_path,
        """
        [rooms]
        kitchen = { pose = [0,0,0], polygon = [[0,0],[4,0],[4,4],[0,4]] }
        living_room = { pose = [0,0,0], polygon = [[4,0],[8,0],[8,4],[4,4]] }
        """,
    )
    assert book.room_at(2, 2) == "kitchen"
    assert book.room_at(6, 2) == "living_room"
    assert book.room_at(20, 20) is None


def test_room_at_backcompat_no_polygon(tmp_path):
    # Rooms without a surveyed polygon are simply skipped (never crash).
    book = _book(tmp_path, "[rooms]\nkitchen = { pose = [1,2,0] }\n")
    assert book.rooms["kitchen"].polygon == ()
    assert book.room_at(1, 2) is None
