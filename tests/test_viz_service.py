"""Unit tests for the shared visualization service (services.viz).

These stay rerun-free: they exercise the singleton, the no-op stub, backend
resolution (incl. the legacy WALKIE_GRAPHS_VIZ fallback), the NoOpViz/RerunSession
signature parity that lets call sites skip null checks, and the axes() triad math.
The real Rerun draw calls are covered by the manual_tests / on-robot path.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import services.viz as viz_pkg
from services.viz import NoOpViz, RerunSession, VizSession, get_viz, reset_viz

PRIMITIVES = ("axes", "points", "box", "arrow", "lines", "clear", "robot", "camera")


@pytest.fixture(autouse=True)
def _clean_singleton(monkeypatch):
    """Each test gets a fresh singleton and no inherited WALKIE_VIZ* env."""
    for k in ("WALKIE_VIZ", "WALKIE_GRAPHS_VIZ"):
        monkeypatch.delenv(k, raising=False)
    reset_viz()
    yield
    reset_viz()


def test_disabled_by_default_returns_noop():
    s = get_viz()
    assert isinstance(s, NoOpViz)


def test_noop_methods_are_safe_and_return_none():
    s = get_viz()
    assert s.axes("grasp/ee", (1, 0, 0.5), rotation=None, length=0.2, labels=True) is None
    assert s.points("x", [[0, 0, 0]], colors=[(1, 2, 3)]) is None
    assert s.box("b", (0, 0, 0), (1, 1, 1)) is None
    assert s.arrow("a", (0, 0, 0), (1, 0, 0)) is None
    assert s.lines("l", [[[0, 0, 0], [1, 1, 1]]]) is None
    assert s.clear("grasp") is None
    assert s.robot({"x": 1, "y": 2, "heading": 0.0}) is None
    assert s.camera(None) is None


def test_noop_path_never_imports_rerun():
    get_viz()
    assert "rerun" not in sys.modules


def test_get_viz_is_a_singleton():
    assert get_viz() is get_viz()
    reset_viz()
    # A fresh build after reset is a distinct instance.
    assert get_viz() is not None


def test_backend_resolution_and_legacy_fallback(monkeypatch):
    assert viz_pkg._resolve_backend() == "none"
    monkeypatch.setenv("WALKIE_GRAPHS_VIZ", "rerun")  # legacy name still honored
    assert viz_pkg._resolve_backend() == "rerun"
    monkeypatch.setenv("WALKIE_VIZ", "none")  # new name takes precedence
    assert viz_pkg._resolve_backend() == "none"


def test_signature_parity_noop_matches_session():
    for name in PRIMITIVES:
        assert callable(getattr(RerunSession, name)), f"RerunSession missing {name}"
        assert callable(getattr(NoOpViz, name)), f"NoOpViz missing {name}"
    assert isinstance(NoOpViz(), VizSession)


def test_axes_triad_vectors_are_scaled_rotation_columns():
    # axes() draws three arrows = the rotation's columns scaled by `length`.
    R = np.eye(3)
    assert (R * 0.1).T.tolist() == [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
    # A non-identity rotation: column i (axis i) becomes arrow i.
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # +90deg about z
    arrows = (Rz * 0.2).T.tolist()
    assert arrows[0] == pytest.approx([0, 0.2, 0])   # X axis -> +Y
    assert arrows[1] == pytest.approx([-0.2, 0, 0])  # Y axis -> -X
    assert arrows[2] == pytest.approx([0, 0, 0.2])   # Z axis unchanged
