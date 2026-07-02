"""Live top-down 2D matplotlib view of the hybrid lidar+CV person follow.

Shared by the runtime loop (``_follow_person_lidar`` when
``HRI_FOLLOW_LIDAR_VIZ=1`` — e.g. ``HRI_SLICE=follow_host``) and the offline
tuning tool (``manual_tests/test_lidar_follow_viz.py``) so both render
identically, in the map frame:

  * grey raw scan points (every valid beam),
  * hollow-orange candidate cluster centroids,
  * the RED member points of the cluster currently SELECTED to follow — the
    person's lidar cloud, the thing this whole request is about,
  * the green alpha-beta track dot + its 1-second velocity lead and the
    staleness-grown association gate circle,
  * the purple CV fix (identity source), and the blue robot triangle.

matplotlib is imported lazily and every public method is best-effort: a viz
glitch prints once and is swallowed, so it can never disturb the follow loop.
The constructor raises ``ImportError`` only when matplotlib is missing (or a
window can't be created headless) — callers catch that to degrade to no viz.
"""

from __future__ import annotations

import math

from typing import Iterable, Sequence


def _xs_ys(pts: Iterable[Sequence[float]]) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for x, y in pts:
        xs.append(x)
        ys.append(y)
    return xs, ys


class LidarFollowViz:
    """A single reusable matplotlib figure; :meth:`update` redraws each tick."""

    def __init__(self, *, lim: float = 6.0, gate: float = 0.6) -> None:
        import matplotlib.pyplot as plt  # lazy: raises ImportError if unavailable

        self._plt = plt
        self.lim = lim
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_aspect("equal")
        ax.set_xlabel("map x (m)")
        ax.set_ylabel("map y (m)")
        ax.grid(True, linestyle=":", alpha=0.5)
        (self._scan,) = ax.plot([], [], ".", color="0.6", markersize=2, label="scan")
        (self._cand,) = ax.plot([], [], "o", color="tab:orange", markersize=7,
                                fillstyle="none", label="candidates")
        (self._sel,) = ax.plot([], [], ".", color="tab:red", markersize=6,
                               label="selected cloud")
        (self._track,) = ax.plot([], [], "o", color="tab:green", markersize=10, label="track")
        (self._fix,) = ax.plot([], [], "*", color="tab:purple", markersize=14, label="CV fix")
        (self._robot,) = ax.plot([], [], "b^", markersize=12, label="robot")
        (self._vel,) = ax.plot([], [], "-", color="tab:green", lw=2)  # 1 s velocity lead
        self._gate = plt.Circle((0, 0), gate, fill=False, color="tab:green",
                                linestyle="--", alpha=0.7)
        ax.add_patch(self._gate)
        self._gate.set_visible(False)
        ax.legend(loc="upper right")
        self.fig, self.ax = fig, ax
        plt.ion()
        plt.show()

    def alive(self) -> bool:
        """False once the window is closed (so the loop can drop the viz)."""
        try:
            return bool(self._plt.fignum_exists(self.fig.number))
        except Exception:  # noqa: BLE001
            return False

    def update(
        self,
        *,
        robot_xy: Sequence[float],
        scan_xy: Iterable[Sequence[float]] = (),
        cand_xy: Iterable[Sequence[float]] = (),
        selected_xy: Iterable[Sequence[float]] = (),
        track_xy: Sequence[float] | None = None,
        track_vel: Sequence[float] = (0.0, 0.0),
        gate: float | None = None,
        fix_xy: Sequence[float] | None = None,
        title: str | None = None,
    ) -> None:
        """Redraw one frame. Never raises — a viz glitch must not stop the follow."""
        try:
            self._scan.set_data(*_xs_ys(scan_xy))
            self._cand.set_data(*_xs_ys(cand_xy))
            self._sel.set_data(*_xs_ys(selected_xy))
            self._robot.set_data([robot_xy[0]], [robot_xy[1]])
            if track_xy is not None:
                tx, ty = track_xy
                self._track.set_data([tx], [ty])
                self._vel.set_data([tx, tx + track_vel[0]], [ty, ty + track_vel[1]])
                self._gate.center = (tx, ty)
                if gate is not None:
                    self._gate.set_radius(gate)
                self._gate.set_visible(True)
            else:
                self._track.set_data([], [])
                self._vel.set_data([], [])
                self._gate.set_visible(False)
            if fix_xy is not None:
                self._fix.set_data([fix_xy[0]], [fix_xy[1]])
            self.ax.set_xlim(robot_xy[0] - self.lim, robot_xy[0] + self.lim)
            self.ax.set_ylim(robot_xy[1] - self.lim, robot_xy[1] + self.lim)
            if title is not None:
                self.ax.set_title(title)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()  # pump GUI events without an extra sleep
        except Exception as exc:  # noqa: BLE001
            print(f"[skills] lidar follow viz update failed ({exc})")

    def save(self, path: str) -> None:
        """Best-effort snapshot of the current frame to *path*."""
        try:
            self.fig.savefig(path, dpi=100, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            print(f"[skills] lidar follow viz save failed ({exc})")

    def close(self) -> None:
        try:
            self._plt.close(self.fig)
        except Exception:  # noqa: BLE001
            pass


def follow_title(state: str, n_candidates: int, track) -> str:
    """One-line status matching the manual test's title."""
    if track is not None:
        speed = math.hypot(track.vx, track.vy)
        return f"{state} — {n_candidates} candidates — track v={speed:.2f} m/s"
    return f"{state} — {n_candidates} candidates — no track"
