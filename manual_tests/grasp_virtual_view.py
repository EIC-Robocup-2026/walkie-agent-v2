"""A/B harness for the virtual-viewpoint grasp transform (experiment).

Walkie's head camera is fixed at a ~35deg downward tilt. The hypothesis: GraspNet
generates better grasps when the graspable surface faces the (virtual) camera
square-on. This harness tests it WITHOUT moving the robot between trials — it captures
ONE real 35deg object cloud, then replays GraspNet over it under a fixed comparison
matrix so the rotation is the only variable:

    | arm            | cloud transform | infer kwargs                          |
    | baseline       | none            | approach_preference="none"            |
    | re-rank side   | none            | approach_preference="side", up=up_opt |
    | re-rank top    | none            | approach_preference="top",  up=up_opt |
    | virtual side   | side            | approach_preference="none"            |
    | virtual top    | top             | approach_preference="none"            |
    | virtual+rerank | side            | approach_preference="side", up=up_virt|

Why a captured fixture (not a synthetic box): the whole premise is the *partial,
self-occluded* structure of a real oblique capture. A full synthetic box has no
occlusion and would test nothing — so replay REFUSES to run without a real fixture.

Reading the results
-------------------
* GraspNet's ``score`` ranks grasps WITHIN one arm only — it is NOT comparable across
  arms, because the transformed inputs are out-of-distribution and the learned score
  drifts arbitrarily. Use it to read within-arm ranking, never to pick a winner.
* ``antipodal_score`` (force-closure quality) IS a rigid-invariant geometric property,
  so it is the fair cross-arm signal here — alongside the Rerun visual (does the grasp
  sit on the object, is the approach reachable). Real on-robot grasp success is the
  only gold metric for the final go/no-go.
* If quality swings a lot across arms -> the bottleneck is GraspNet's VIEW PRIOR and
  the transform is worth pursuing. If every arm is mediocre -> the bottleneck is depth
  COVERAGE at the grazing angle, which a rotation cannot fix (use multi-view fusion +
  the existing re-rank instead).

Two modes (WALKIE_GRASP_VV_MODE, default "replay"):

  capture  — needs the robot (ZED + transforms) and a running walkie-ai-server. Grabs
             one snapshot, detects+lifts the nearest match, cleans the cloud, and saves
             {cloud, cam_R, cam_t} to the fixture .npz.
                 WALKIE_GRASP_VV_MODE=capture uv run python -m manual_tests.grasp_virtual_view

  replay   — needs only walkie-ai-server. Loads the fixture and runs the matrix above.
             Enable Rerun with WALKIE_VIZ=rerun to see clouds + grasp axes in the map
             frame.
                 WALKIE_GRASP_VV_MODE=replay uv run python -m manual_tests.grasp_virtual_view

Knobs: WALKIE_GRASP_VV_FIXTURE (npz path), WALKIE_GRASP_VV_PROMPTS (comma list),
WALKIE_GRASP_VV_MAX_GRASPS, WALKIE_GRASP_VV_CAPTURE_TRIES, WALKIE_GRASP_VV_REPEATS
(replay the matrix N times and aggregate antipodal mean/max/std per arm — set >1 to tell
a real lift apart from GraspNet's run-to-run noise).

The production counterpart is the WALKIE_GRASP_VIRTUAL_VIEW config knob (default "none"),
which applies the SAME transform inside tasks/skills/grasp.py::get_object_grasp_pos on the
real robot. This harness is how you pick its value before flipping it on.
"""

import os

import numpy as np
from dotenv import load_dotenv

from client import WalkieAIClient
from services.viz import get_viz
from tasks.skills.grasp import (
    _apply_virtual_view,
    _clean_object_cloud,
    _grasp_to_optical,
    _optical_ref,
    _virtual_view_rotation,
    locate_object,
)
from walkie_config import load_config

ZENOH_PORT = 7447
ROBOT_IP = "127.0.0.1"

DEFAULT_FIXTURE = "graph_debug/grasp_vv_fixture.npz"
DEFAULT_PROMPTS = ["red can", "can", "bottle", "cup", "mug", "box", "object"]

# (name, cloud-transform mode, approach_preference). `up` is supplied automatically:
# the real optical up for non-virtual arms, the rotated virtual up for virtual arms.
MATRIX = [
    ("baseline",       "none", "none"),
    ("re-rank side",   "none", "side"),
    ("re-rank top",    "none", "top"),
    ("virtual side",   "side", "none"),
    ("virtual top",    "top",  "none"),
    ("virtual+rerank", "side", "side"),
]

# Distinct colors per arm for the Rerun overlay (grasp axes share the cloud).
_ARM_COLORS = {
    "baseline": (200, 200, 200),
    "re-rank side": (60, 160, 255),
    "re-rank top": (255, 160, 60),
    "virtual side": (60, 220, 120),
    "virtual top": (220, 80, 200),
    "virtual+rerank": (240, 220, 60),
}


class _Shim:
    """Minimal ctx for locate_object — it only reads ``walkieAI`` when given a snap."""

    def __init__(self, walkieAI):
        self.walkieAI = walkieAI


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def run_capture(walkieAI: WalkieAIClient, fixture: str, prompts: list[str]) -> None:
    from walkie_sdk import WalkieRobot

    from interfaces.devices.camera import CameraSnapshot
    from interfaces.walkie_interface import WalkieInterface

    robot = WalkieRobot(ip=ROBOT_IP, camera_protocol="zenoh", camera_port=ZENOH_PORT)
    walkie = WalkieInterface(robot)
    shim = _Shim(walkieAI)
    tries = int(os.getenv("WALKIE_GRASP_VV_CAPTURE_TRIES", "30"))
    print(f"Capture mode: looking for {prompts} (up to {tries} snapshots)")

    for i in range(tries):
        snap = CameraSnapshot.capture(walkie, log=print)
        if snap is None or not snap.has_geometry:
            print(f"[{i}] no snapshot geometry — is the ZED running?")
            continue
        loc = locate_object(shim, prompts, snap=snap)
        if loc is None:
            print(f"[{i}] nothing matched/lifted; retrying")
            continue

        # Clean ONCE here (background-bleed removal etc.) so the saved fixture is exactly
        # what production would hand GraspNet — and so replay's only variable is the
        # rotation, not the cleanup.
        cloud = _clean_object_cloud(loc.cloud_optical, ref_optical=_optical_ref(loc))
        if cloud.shape[0] < 200:
            print(f"[{i}] cleaned cloud too small ({cloud.shape[0]} pts); retrying")
            continue

        os.makedirs(os.path.dirname(fixture) or ".", exist_ok=True)
        np.savez(
            fixture,
            cloud=cloud.astype(np.float32),
            cam_R=np.asarray(snap.cam.R, dtype=np.float64),
            cam_t=np.asarray(snap.cam.t, dtype=np.float64),
            xyz_map=np.asarray(loc.xyz_map, dtype=np.float64),
            prompts=np.asarray(prompts, dtype=object),
        )
        print(
            f"\nSaved fixture -> {fixture}\n"
            f"  {cloud.shape[0]} cleaned optical points, range~{loc.range_m:.2f}m, "
            f"object at map {tuple(round(v, 2) for v in loc.xyz_map)}\n"
            f"Now run replay:  WALKIE_GRASP_VV_MODE=replay "
            f"uv run python -m manual_tests.grasp_virtual_view"
        )
        return

    print(f"\nNo graspable object found in {tries} snapshots — nothing saved.")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def _rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic rotation magnitude of a 3x3 rotation, in degrees (0 = identity)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _infer_arm(walkieAI, cloud_opt, up_opt, transform, preference, max_grasps):
    """Run one matrix arm: transform the cloud, infer, map grasps back to the true
    optical frame.

    Returns ``(grasps, R_rel, centroid, cloud_v)``: grasps best-first in the TRUE optical
    frame, the rotation applied to the cloud (identity for non-virtual arms), the pivot
    centroid, and the transformed cloud actually sent to GraspNet (so the caller can show
    exactly what the network saw).
    """
    R_rel, c = _virtual_view_rotation(cloud_opt, up_opt, transform)
    cloud_v = _apply_virtual_view(cloud_opt, R_rel, c)
    up_v = R_rel @ up_opt  # world-up in the (virtual) frame the cloud is now in

    # Always hand the frame-correct up: required for the re-rank, and (with preference
    # "none") it still lets the client orient each grasp's wrist X-up consistently
    # (cosmetic; never changes score/approach/position).
    kwargs = {"antipodal": True, "max_grasps": max_grasps, "up": up_v}
    if preference != "none":
        kwargs["approach_preference"] = preference

    grasps_v = walkieAI.grasp.infer(cloud_v, **kwargs)
    # Map each grasp from the virtual frame back to the TRUE optical frame so every arm
    # is comparable and downstream map-frame mapping is identical.
    grasps = [_grasp_to_optical(g, R_rel, c) for g in grasps_v]
    return grasps, R_rel, c, cloud_v


def run_replay(walkieAI: WalkieAIClient, fixture: str) -> None:
    if not os.path.exists(fixture):
        print(
            f"No fixture at {fixture}. Capture a real cloud first:\n"
            f"  WALKIE_GRASP_VV_MODE=capture uv run python -m manual_tests.grasp_virtual_view\n"
            "(A synthetic box has no self-occlusion and would not test the hypothesis.)"
        )
        return

    data = np.load(fixture, allow_pickle=True)
    cloud = np.asarray(data["cloud"], dtype=np.float64)
    cam_R = np.asarray(data["cam_R"], dtype=np.float64)
    cam_t = np.asarray(data["cam_t"], dtype=np.float64)
    up_opt = cam_R.T @ np.array([0.0, 0.0, 1.0])  # world-up in the optical frame
    max_grasps = int(os.getenv("WALKIE_GRASP_VV_MAX_GRASPS", "5"))
    print(f"Loaded fixture {fixture}: {cloud.shape[0]} optical points")
    print(f"up_opt (gravity = -up) in optical frame: {np.round(up_opt, 3)}\n")

    viz = get_viz()
    cloud_map = cloud @ cam_R.T + cam_t
    viz.clear("grasp_vv", recursive=True)
    viz.points("grasp_vv/cloud", cloud_map.astype(np.float32),
               colors=[(120, 120, 120)], radii=[0.004])

    repeats = max(1, int(os.getenv("WALKIE_GRASP_VV_REPEATS", "1")))
    if repeats > 1:
        print(f"Repeating the matrix {repeats}x to gauge GraspNet's run-to-run noise.\n")

    # Accumulate antipodal + approach-z across EVERY repeat so a single lucky/unlucky run
    # can't decide the verdict. `last` keeps the final run's grasps for the JSON dump/viz.
    agg = {name: {"transform": t, "preference": p, "rot": 0.0, "anti": [], "z": []}
           for name, t, p in MATRIX}
    last: dict = {}

    for r in range(repeats):
        verbose = r == 0            # detailed per-grasp table only on the first run
        final = r == repeats - 1    # draw the Rerun overlay only on the last run
        if verbose:
            print(f"{'arm':<16}{'#':>3} {'rot°':>5}  {'score':>7} {'antipodal':>9}  "
                  f"{'grasp xyz (map)':>24}  {'approach (map)':>22}  width")
            print("-" * 112)
        drawn_views: dict = {}  # transform -> gallery offset (draw each distinct view once)
        for name, transform, preference in MATRIX:
            try:
                grasps, R_rel, c, cloud_v = _infer_arm(
                    walkieAI, cloud, up_opt, transform, preference, max_grasps
                )
            except Exception as exc:  # noqa: BLE001 — one bad arm shouldn't kill the run
                if verbose:
                    print(f"{name:<16}  ERROR: {exc}")
                continue
            rot_deg = _rotation_angle_deg(R_rel)
            agg[name]["rot"] = round(rot_deg, 2)
            for gg in grasps:
                if gg.antipodal_score is not None:
                    agg[name]["anti"].append(float(gg.antipodal_score))
                agg[name]["z"].append(float((cam_R @ gg.rotation[:, 2])[2]))

            if final:
                # Stash this run's grasps (map frame) for the JSON, and draw the overlay:
                # the rotated-cloud gallery beside the original + each arm's top grasp axes.
                last[name] = [{
                    "score": float(gg.score),
                    "antipodal": None if gg.antipodal_score is None else float(gg.antipodal_score),
                    "width_m": float(gg.width),
                    "grasp_xyz_map": [round(float(v), 4) for v in (cam_R @ np.asarray(gg.translation) + cam_t)],
                    "approach_map": [round(float(v), 4) for v in (cam_R @ gg.rotation[:, 2])],
                } for gg in grasps]
                if transform != "none" and transform not in drawn_views:
                    offset = np.array([0.0, 0.6 * (len(drawn_views) + 1), 0.0])
                    drawn_views[transform] = offset
                    cloud_v_map = cloud_v @ cam_R.T + cam_t + offset
                    viz.points(f"grasp_vv/rotated_view/{transform}", cloud_v_map.astype(np.float32),
                               colors=[_ARM_COLORS.get(name, (255, 255, 255))], radii=[0.004])
                if grasps:
                    g0 = grasps[0]
                    pm0 = cam_R @ np.asarray(g0.translation, dtype=float) + cam_t
                    viz.axes(f"grasp_vv/{name}/ee", pm0.tolist(),
                             rotation=(cam_R @ g0.rotation), length=0.08, labels=False)
                    viz.points(f"grasp_vv/{name}/pt", [pm0.tolist()], radii=[0.012],
                               colors=[_ARM_COLORS.get(name, (255, 255, 255))], labels=[name])

            if verbose:
                if not grasps:
                    print(f"{name:<16}{0:>3} {rot_deg:>5.1f}  (no grasps)")
                    continue
                g = grasps[0]
                p_map = cam_R @ np.asarray(g.translation, dtype=float) + cam_t
                approach_map = cam_R @ g.rotation[:, 2]
                anti = "n/a" if g.antipodal_score is None else f"{g.antipodal_score:7.3f}"
                print(
                    f"{name:<16}{len(grasps):>3} {rot_deg:>5.1f}  {g.score:7.3f} {anti:>9}  "
                    f"({p_map[0]:+.2f},{p_map[1]:+.2f},{p_map[2]:+.2f})  "
                    f"({approach_map[0]:+.2f},{approach_map[1]:+.2f},{approach_map[2]:+.2f})  "
                    f"{g.width * 100:4.1f}cm"
                )

    # Aggregate across all repeats. antipodal (rigid-invariant) is the fair cross-arm metric;
    # %down = fraction of approaches pointing down (top-down grasps) vs up/horizontal.
    print("-" * 112)
    print(f"AGGREGATE over {repeats} run(s) (antipodal is the cross-arm-fair metric):")
    print(f"{'arm':<16}{'rot°':>6}{'n':>5}{'anti_mean':>11}{'anti_max':>10}{'anti_std':>10}{'%down':>8}")
    summary: dict = {}
    for name, t, p in MATRIX:
        a = np.asarray(agg[name]["anti"], dtype=float)
        z = np.asarray(agg[name]["z"], dtype=float)
        n = int(a.size)
        mean = float(a.mean()) if n else float("nan")
        mx = float(a.max()) if n else float("nan")
        std = float(a.std()) if n else float("nan")
        pct_down = float((z < 0).mean() * 100.0) if z.size else float("nan")
        print(f"{name:<16}{agg[name]['rot']:>6.1f}{n:>5}{mean:>11.3f}{mx:>10.3f}{std:>10.3f}{pct_down:>7.0f}%")
        summary[name] = {
            "transform": t, "preference": p, "rotation_deg": agg[name]["rot"], "n_grasps": n,
            "antipodal_mean": round(mean, 4) if n else None,
            "antipodal_max": round(mx, 4) if n else None,
            "antipodal_std": round(std, 4) if n else None,
            "frac_down": round(pct_down / 100.0, 3) if z.size else None,
            "final_run_grasps": last.get(name, []),
        }

    import json

    results_path = os.getenv("WALKIE_GRASP_VV_RESULTS", f"{os.path.splitext(fixture)[0]}.results.json")
    try:
        with open(results_path, "w") as fh:
            json.dump({"fixture": fixture, "n_points": int(cloud.shape[0]), "repeats": repeats,
                       "up_opt": [round(float(v), 4) for v in up_opt], "arms": summary}, fh, indent=2)
        print(f"\nResults saved -> {results_path}")
    except Exception as exc:  # noqa: BLE001 — dump is best-effort
        print(f"\n(could not save results: {exc})")

    print(
        "\nRead `anti_*` (rigid-invariant), NOT `score` (OOD across arms). A transform wins\n"
        "only if it beats BOTH baseline AND the re-rank arms on anti_mean with low std.\n"
        "%down tells whether an arm's approaches are top-down (high) or side/upward (low).\n"
        "Still a geometric proxy — confirm the winner with real on-robot grasps, across\n"
        "2-3 objects (side suits tall/round, top suits flat/wide-topped). To raise repeats:\n"
        "WALKIE_GRASP_VV_REPEATS=5 WALKIE_GRASP_VV_MODE=replay uv run python -m manual_tests.grasp_virtual_view"
    )
    if viz.__class__.__name__ != "NoOpViz":
        print("\nRerun overlay (final run) under entity path 'grasp_vv/*'.")


def main() -> None:
    load_dotenv()
    load_config()
    mode = os.getenv("WALKIE_GRASP_VV_MODE", "replay").strip().lower()
    fixture = os.getenv("WALKIE_GRASP_VV_FIXTURE", DEFAULT_FIXTURE)
    prompts_env = os.getenv("WALKIE_GRASP_VV_PROMPTS", "").strip()
    prompts = [p.strip() for p in prompts_env.split(",") if p.strip()] or DEFAULT_PROMPTS

    walkieAI = WalkieAIClient(base_url=os.getenv("WALKIE_AI_BASE_URL", "http://localhost:5000"))
    print(f"Mode: {mode}  fixture: {fixture}")
    if mode == "capture":
        run_capture(walkieAI, fixture, prompts)
    else:
        run_replay(walkieAI, fixture)


if __name__ == "__main__":
    main()
