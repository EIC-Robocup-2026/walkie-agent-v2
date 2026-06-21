"""Shared 3D visualization service (process-wide, Rerun-backed, optional).

One viz session per process. :func:`get_viz` returns a singleton — the scene-graph
perception loop and any competition task both draw into the *same* Rerun recording
(Rerun's recording + sink are process-global, so there can only be one), separated
by entity-path namespace. When viz is disabled it returns a :class:`NoOpViz` so
callers never null-check.

Enable with ``WALKIE_VIZ=rerun`` (legacy ``WALKIE_GRAPHS_VIZ`` is honored as a
fallback). Disabled (``none``) by default — walkie_graphs and tasks run fine without
it. See :mod:`services.viz.session` for the drawing primitives and the env knobs.

    from services.viz import get_viz
    viz = get_viz()                       # RerunSession or NoOpViz
    viz.axes("grasp/ee", xyz, rotation=R) # XYZ triad at a pose
"""

from __future__ import annotations

import threading

from .session import NoOpViz, RerunSession, VizSession, _env

__all__ = ["get_viz", "reset_viz", "VizSession", "NoOpViz", "RerunSession"]

_LOCK = threading.Lock()
_INSTANCE: VizSession | None = None


def _resolve_backend() -> str:
    return _env("WALKIE_VIZ", "WALKIE_GRAPHS_VIZ", "none").strip().lower()


def _build() -> VizSession:
    backend = _resolve_backend()
    if backend in ("", "none"):
        return NoOpViz()
    if backend == "rerun":
        try:
            return RerunSession()
        except Exception as e:  # noqa: BLE001 — viz is best-effort, never crash a caller
            print(f"[viz] backend 'rerun' unavailable: {e}; visualization disabled")
            return NoOpViz()
    print(f"[viz] unknown backend {backend!r}; visualization disabled")
    return NoOpViz()


def get_viz() -> VizSession:
    """Return the process-wide viz session, building it once (thread-safe).

    Returns a :class:`NoOpViz` when viz is disabled or the backend fails to build,
    so the result is always safe to call.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _LOCK:
        if _INSTANCE is None:  # double-checked: graphs daemon thread vs task main thread
            _INSTANCE = _build()
    return _INSTANCE


def reset_viz() -> None:
    """Drop the singleton so the next :func:`get_viz` rebuilds (test-only).

    Does not tear down Rerun (it has no teardown); tests should monkeypatch the
    backend / env rather than expecting a fresh ``rr.init``.
    """
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None
