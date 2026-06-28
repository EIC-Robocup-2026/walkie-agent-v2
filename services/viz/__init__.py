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


def _build_rerun() -> VizSession:
    """Build a RerunSession, but never let it HANG the caller.

    ``RerunSession.__init__`` stands up rerun's gRPC sink + web viewer in native
    code. In a busy multi-threaded robot process that native setup can *deadlock*
    rather than raise — observed on-robot during a task: ``serve_grpc`` binds its
    port but ``serve_web_viewer`` never returns and the calling thread parks in a
    futex, freezing the whole task before its first step (it builds on the MAIN
    thread via ``TaskContext.__post_init__``). A plain ``try/except`` can't catch a
    hang. Since viz is best-effort, build it in a daemon thread and fall back to
    :class:`NoOpViz` if it doesn't come up within ``WALKIE_VIZ_INIT_TIMEOUT_S``
    (default 10s; 0 = wait forever, the old behavior) — so a viz deadlock degrades
    to no-viz instead of hanging the run. When the build is fast (the normal case)
    ``join`` returns the instant it finishes, adding no startup latency.
    """
    timeout = float(_env("WALKIE_VIZ_INIT_TIMEOUT_S", None, "10"))
    box: dict[str, object] = {}

    def _work() -> None:
        try:
            box["viz"] = RerunSession()
        except Exception as e:  # noqa: BLE001 — viz is best-effort, never crash a caller
            box["err"] = e

    t = threading.Thread(target=_work, name="viz-init", daemon=True)
    t.start()
    t.join(timeout if timeout > 0 else None)
    if t.is_alive():
        print(f"[viz] rerun init did not finish in {timeout:.0f}s (likely a serve "
              "deadlock); visualization disabled for this run. Set WALKIE_VIZ=none "
              "to skip it, or raise WALKIE_VIZ_INIT_TIMEOUT_S to wait longer.")
        return NoOpViz()
    if "err" in box:
        print(f"[viz] backend 'rerun' unavailable: {box['err']}; visualization disabled")
        return NoOpViz()
    return box["viz"]  # type: ignore[return-value]


def _build() -> VizSession:
    backend = _resolve_backend()
    if backend in ("", "none"):
        return NoOpViz()
    if backend == "rerun":
        return _build_rerun()
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
