"""Best-of-N grasp selection — snap the camera a few times, keep the best grasp.

GraspNet only ever sees a camera-**optical** cloud (X-right, Y-down, Z-forward),
which is what keeps it in-distribution. This skill maps the winning grasp back to
the **map frame** using the snapshot's frozen capture-time pose, so callers get a
map-frame grasp point (and a backed-off pre-grasp point) ready to hand to the arm.

    cand = grasp_object(ctx, ["red can"], attempts=5, approach_preference="side")
    if cand:
        ctx.goto_pregrasp(cand.pregrasp_xyz)   # caller's arm/nav logic
        ...
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, replace

import numpy as np
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from interfaces.devices.camera import camera_pose
from interfaces.perception.dbscan import dbscan_labels, statistical_outlier_removal
from interfaces.perception.geometry import voxel_downsample
from tasks.base import TaskContext
from tasks.skills.navigation import creep_base_relative, strafe_servo, tilt_head

Vec3 = tuple[float, float, float]

# Head tilt limits (walkie_sdk.modules.head): 0 = level, +down. head.tilt RAISES
# outside this band, so we clamp locally before commanding.
_HEAD_TILT_MIN = -math.pi / 4  # -45deg, look up
_HEAD_TILT_MAX = math.pi / 3  # +60deg, look down


# --- config helpers (mirror tasks.skills.place) -----------------------------
def _f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _b(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# --- descriptor-prompt detection + CLIP rerank ------------------------------
# YOLOE is open-vocab but PROMPT-DRIVEN: it can't box a brand name ("coke") even
# though the scene-graph loop already prompts it with that exact word. But under a
# generic *visual* descriptor ("can", "red can") it returns SEVERAL boxes (the coke
# plus neighbours). So we detect with descriptors, then CLIP-rerank the boxes against
# the SPECIFIC target name to pick the right one. The correctness trap: detect with the
# GENERIC descriptor, but embed_text the SPECIFIC target — embedding "can" scores every
# can equally and defeats disambiguation.
#
# The descriptors themselves are now **LLM-generated** (`_llm_descriptors`): for an
# arbitrary GPSR target the LLM emits the same kind of generic visual phrases, few-shot
# -anchored on the curated map below. That map is therefore no longer the only source of
# descriptors — it is (a) the few-shot examples that steer the LLM and (b) the fallback
# used when the LLM is disabled (WALKIE_GRASP_LLM_DESCRIPTORS=0) or the call fails. Keys
# are lowercased item names matching the known set (services/walkie_graphs/config.toml
# WALKIE_EXPLORE_INTERESTED_CLASSES); the wording is a starting point, tuned on-robot
# against YOLOE's vocab, and doubles as the demonstration the LLM imitates.
_GRASP_DESCRIPTORS: dict[str, list[str]] = {
    "cola": ["can", "red can", "soda can"],
    "coke": ["can", "red can", "soda can"],
    "red bull": ["can", "blue can", "slim can", "energy drink can"],
    "ice tea": ["bottle", "carton", "drink bottle"],
    "orange juice": ["carton", "bottle", "juice carton"],
    "milk": ["carton", "bottle", "milk carton"],
    "bottle": ["bottle", "water bottle"],
    "water bottle": ["bottle", "water bottle"],
    "pringles": ["can", "tube", "cylindrical can", "chips can"],
    "chips": ["bag", "snack bag", "chip bag"],
    "cookies": ["box", "package", "snack box"],
    "cornflakes": ["box", "cereal box"],
    "instant noodles": ["cup", "package", "noodle cup"],
    "tomato soup": ["can", "soup can"],
    "mixed nuts": ["can", "jar", "package"],
    "gum": ["box", "small box", "package"],
    "hand cream": ["tube", "bottle"],
    "soap": ["bottle", "box", "bar"],
    "toothpaste": ["box", "tube"],
}

# Success-only cache for CLIP text embeddings, keyed by the formatted query. NOT
# functools.lru_cache: that would pin a None failure forever and never retry the server.
_TEXT_EMB_CACHE: dict[str, list[float]] = {}

# Success-only cache for LLM-generated descriptors, keyed by lowercased target. Same
# rationale as _TEXT_EMB_CACHE: a pick runs many locates (multi-tilt × attempts), so the
# LLM is asked once per unique target and reused; a failed call is NOT cached, so a flaky
# endpoint is retried on the next locate instead of being pinned to the static fallback.
_DESCRIPTOR_CACHE: dict[str, list[str]] = {}

# Hard cap on descriptors per target: instructions ask for 2-4, but a misbehaving model
# could dump 20 — and every extra prompt costs YOLOE compute on EVERY locate, then dilutes
# the CLIP rerank. Trim defensively regardless of what the model returns.
_MAX_DESCRIPTORS = 5


class _DescriptorList(BaseModel):
    """Structured-output schema for `_llm_descriptors` (one field: the phrase list)."""

    descriptors: list[str]


def _call_with_timeout(fn, timeout_s: float):
    """Run ``fn()`` on a daemon thread, returning its result or raising on timeout.

    Scoped guard for the one blocking LLM round-trip this module makes: ``ctx.model`` (the
    OpenRouter/local ``ChatOpenAI``) is built WITHOUT an HTTP timeout, so a stalled endpoint
    would otherwise hang the grasp hot path indefinitely. The worker is a daemon, so an
    abandoned (timed-out) call can never block process exit. Re-raises the worker's own
    exception so the caller's degrade-to-static path fires identically for hang or error.
    """
    box: dict = {}

    def _run():
        try:
            box["v"] = fn()
        except Exception as exc:  # noqa: BLE001 — relayed to the caller verbatim
            box["e"] = exc

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError(f"LLM call exceeded {timeout_s:.0f}s")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _llm_descriptors(ctx, target: str) -> list[str]:
    """Generic visual descriptors for *target*, generated by the LLM, cached per target.

    YOLOE can't box a brand name, so we feed it generic visual phrases ("can", "red can").
    The static :data:`_GRASP_DESCRIPTORS` only covers a handful of competition items; for an
    arbitrary GPSR target we ask ``ctx.model`` (via :meth:`TaskContext.extract`, which keeps
    a JSON-mode fallback for tool-call-less local backends) for the same kind of phrases,
    few-shot-anchored on that map. Returns ``[]`` — the signal for the caller to fall back to
    the static map — when disabled, given an empty target, or on ANY failure (timeout,
    network, no ``extract``/``model`` on the ctx, empty result).

    Every outcome (success, empty, AND failure) is cached per target — unlike
    :data:`_TEXT_EMB_CACHE` (success-only), because a pick runs many locates and an embed
    miss fails *fast* whereas an LLM miss can fail *slow* (an 8 s timeout to a dead host).
    Caching the miss costs nothing — the fallback IS the pre-LLM behaviour — and turns a
    per-pick stall (timeout × locates) into one timeout per target per session.
    """
    target = (target or "").strip().lower()
    if not target or not _b("WALKIE_GRASP_LLM_DESCRIPTORS", "1"):
        return []
    cached = _DESCRIPTOR_CACHE.get(target)
    if cached is not None:
        return cached
    examples = "\n".join(
        f"- {name} -> {', '.join(descs)}" for name, descs in _GRASP_DESCRIPTORS.items()
    )
    instructions = (
        "You name an object the way an open-vocabulary object detector (YOLOE) can find it. "
        "Given a specific item — often a brand name the detector cannot recognise — output "
        "1-4 SHORT, GENERIC visual descriptors in lowercase: the object's container/shape "
        "type (can, bottle, box, carton, bag, tube, jar, cup, ...) optionally qualified by a "
        "distinctive colour or size word. NEVER include the brand name itself, and never "
        "invent a shape you are unsure of. Order them from most generic to most specific.\n\n"
        f"Examples:\n{examples}"
    )
    timeout_s = _f("WALKIE_GRASP_LLM_DESCRIPTORS_TIMEOUT_S", "8")
    try:
        result = _call_with_timeout(
            lambda: ctx.extract(_DescriptorList, instructions, target), timeout_s
        )
    except Exception as exc:  # noqa: BLE001 — timeout/network/bad-ctx -> static fallback
        print(f"[grasp] LLM descriptor gen failed for {target!r} ({exc}); using static map")
        _DESCRIPTOR_CACHE[target] = []  # cache the miss: don't re-pay the timeout per locate
        return []
    raw = list(result.descriptors) if result else []
    descs: list[str] = []
    for d in raw:
        d = (d or "").strip().lower()
        if d and d != target and d not in descs:  # skip the brand name + dupes
            descs.append(d)
    descs = descs[:_MAX_DESCRIPTORS]
    _DESCRIPTOR_CACHE[target] = descs  # cache success AND empty alike
    print(f"[grasp] llm descriptors outputs: {descs}")
    return descs


def _detection_prompts(ctx, prompts: list[str]) -> list[str]:
    """Expand the caller's prompts with generic visual descriptors for CLIP rerank.

    The first prompt is the human target (Restaurant passes ``[item]``); keep it first,
    append descriptors (LLM-generated and few-shot-anchored on :data:`_GRASP_DESCRIPTORS`,
    falling back to a static lookup in that map when the LLM is off/fails), dedup preserving
    order. Unknown items with no LLM/static descriptors -> ``[target]`` (harmless under
    rerank). Returns *prompts* unchanged when rerank is off.
    """
    if not prompts or not _b("WALKIE_GRASP_CLIP_RERANK", "1"):
        return list(prompts)
    target = prompts[0].strip().lower()
    descriptors = _llm_descriptors(ctx, target) or _GRASP_DESCRIPTORS.get(target, [])
    out = list(prompts)
    for d in descriptors:
        if d not in out:
            out.append(d)
    return out


def _cosine(a, b) -> float:
    """Cosine similarity of two vectors; 0.0 on degenerate (zero-norm/empty) input."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b) / (na * nb)


def _rank_by_clip(detections, text_emb, *, sim_floor: float):
    """Rank detections by ``cosine(det.embedding, text_emb)`` descending.

    Returns ``[(det, sim)]`` for detections whose cosine is ``>= sim_floor``. Returns
    ``[]`` (the fall-back-to-nearest signal) when *text_emb* is falsy or no detection
    carries an embedding. Pure: no network, no snapshot.
    """
    if not text_emb:
        return []
    scored = []
    for d in detections:
        emb = getattr(d, "embedding", None)
        if not emb:
            continue
        sim = _cosine(emb, text_emb)
        if sim >= sim_floor:
            scored.append((d, sim))
    scored.sort(key=lambda ds: ds[1], reverse=True)
    return scored


def _target_text_embedding(ctx, target: str) -> list[float] | None:
    """CLIP text embedding for the SPECIFIC target, cached across one pick's many locates.

    Builds the query via ``WALKIE_GRASP_CLIP_QUERY_TMPL`` (default ``"a photo of {t}"`` —
    a bare brand string embeds poorly zero-shot). Returns ``None`` (never raises) on any
    failure, so the next call retries (the cache stores successes only).
    """
    target = (target or "").strip()
    if not target:
        return None
    tmpl = os.getenv("WALKIE_GRASP_CLIP_QUERY_TMPL", "a photo of {t}")
    try:
        key = tmpl.format(t=target)
    except (KeyError, IndexError, ValueError):
        key = target
    cached = _TEXT_EMB_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        emb = ctx.walkieAI.image.embed_text(key)
    except Exception as exc:  # noqa: BLE001 — network/empty-text; degrade to nearest
        print(f"[grasp] embed_text failed for {key!r} ({exc}); CLIP rerank off this call")
        return None
    if emb:
        _TEXT_EMB_CACHE[key] = list(emb)
        return _TEXT_EMB_CACHE[key]
    return None


def _clean_object_cloud(pts: np.ndarray, *, ref_optical: np.ndarray | None = None) -> np.ndarray:
    """Strip residue noise from an object cloud before handing it to GraspNet.

    A masked detection's lifted cloud carries two kinds of junk that ruin a grasp:
    sparse flying-pixel scatter, and a *locally-dense* blob where the mask bled onto
    the supporting table/wall at the object's silhouette. The remote server already
    runs statistical-outlier removal, so the scatter is mostly handled there — the blob
    is not: it is dense enough to survive SOR and pulls GraspNet's grasp point/width off
    the real object. This pass removes both, in two rigid-invariant steps (so they are
    valid in the camera-optical frame the cloud arrives in):

    1. :func:`statistical_outlier_removal` — cheap belt-and-suspenders for scatter.
    2. **Nearest-cluster keep** — DBSCAN the cloud and keep the single cluster whose
       centroid is closest to *ref_optical* (the object's expected centre). Picking the
       *nearest* cluster rather than the *largest* is the guard: a badly-bled mask can
       make the table the most populous cluster, and a largest-cluster rule would then
       drop the real object. When *ref_optical* is ``None`` the cloud's median is used
       (robust while the object is the majority of a good mask).

    Never trims below ``WALKIE_GRASP_CLEAN_MIN_KEEP`` points: on any degenerate result it
    falls back to the fuller (pre-cluster, post-SOR) cloud, so cleanup can only help. That
    floor is kept above the server's own ``min_points`` (200) because the server *re*-voxels
    (0.005 m) + runs SOR after receiving the cloud and rejects what's left under 200 — so a
    too-small clean cluster must defer to the noisier-but-fuller cloud rather than hand the
    server something it will throw away. Disabled wholesale by ``WALKIE_GRASP_CLEAN_ENABLE=0``.
    """
    pts = np.asarray(pts)
    pts = pts[np.isfinite(pts).all(axis=1)]  # defensive: drop NaN/inf rows
    if not _b("WALKIE_GRASP_CLEAN_ENABLE", "1"):
        return pts

    min_keep = _i("WALKIE_GRASP_CLEAN_MIN_KEEP", "250")
    if pts.shape[0] < min_keep:
        return pts  # too sparse to risk trimming — a real detection is never thrown away

    # 1. Statistical-outlier removal (no-op if k <= 0). Already falls back to the input
    #    on a degenerate spread, so it can only shed genuine low-density scatter.
    sor = statistical_outlier_removal(
        pts,
        k=_i("WALKIE_GRASP_CLEAN_SOR_K", "16"),
        std_ratio=_f("WALKIE_GRASP_CLEAN_SOR_STD", "2.0"),
    )
    base = sor if sor.shape[0] >= min_keep else pts

    # 2. Nearest-cluster keep — drop a dense background-bleed blob.
    eps = _f("WALKIE_GRASP_CLEAN_DBSCAN_EPS_M", "0.02")
    min_pts = _i("WALKIE_GRASP_CLEAN_DBSCAN_MIN_PTS", "10")
    cluster_min = _i("WALKIE_GRASP_CLEAN_CLUSTER_MIN", "20")
    labels = dbscan_labels(base, eps, min_pts)
    uniq = [lbl for lbl in set(labels.tolist()) if lbl >= 0]
    if not uniq:
        return base  # everything is "noise" at this eps — keep the post-SOR cloud as-is

    ref = (
        np.asarray(ref_optical, dtype=float)
        if ref_optical is not None
        else np.median(base, axis=0)
    )
    # Among clusters big enough to be the object, keep the one centred nearest ref.
    candidates = [(lbl, base[labels == lbl]) for lbl in uniq]
    eligible = [(lbl, c) for lbl, c in candidates if c.shape[0] >= cluster_min]
    pool = eligible if eligible else candidates  # fall back to all clusters if all small
    best = min(pool, key=lambda lc: float(np.linalg.norm(np.median(lc[1], axis=0) - ref)))[1]
    if best.shape[0] < min_keep:
        return base  # nearest cluster too small to trust — keep the post-SOR cloud
    if best.shape[0] < base.shape[0]:
        print(f"[grasp] cleanup: {pts.shape[0]} -> {best.shape[0]} pts "
              f"(dropped {pts.shape[0] - best.shape[0]})")
    return best


# --- virtual-viewpoint transform (experiment) -------------------------------
# Rigidly rotate the lifted object cloud into a virtual "side" or "top" viewpoint
# before handing it to GraspNet, then rotate the returned grasps back into the true
# optical frame so the rest of the pipeline (``_to_map_frame``) is unchanged. The
# hypothesis: GraspNet generates better candidates when the graspable surface faces
# the (virtual) camera square-on, rather than the real ~35deg-down oblique view. A
# rigid rotation adds NO occluded data, so this is a generation-bias experiment, not a
# coverage fix — see manual_tests/grasp_virtual_view.py for the A/B harness. Kept as
# pure, robot-free helpers so they are unit-testable on a single saved cloud.
def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal (geodesic) rotation matrix taking unit vector *a* onto unit vector *b*."""
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(a @ b)
    if s < 1e-9:  # parallel (c>0 -> identity) or anti-parallel (c<0 -> 180deg)
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(math.pi * axis).as_matrix()
    return Rotation.from_rotvec(math.atan2(s, c) * (v / s)).as_matrix()


def _virtual_view_rotation(
    cloud_opt: np.ndarray, up_opt: np.ndarray, mode: str, *, center_xy: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid remap reorienting (and optionally recentring) an optical cloud for GraspNet.

    The lifted cloud is in the camera-optical frame (cam at origin, +Z forward, +Y
    down). Walkie's head looks down at a fixed tilt, so the object is seen obliquely.
    This returns a rotation ``R_rel`` (in the optical frame) that, applied **about
    the cloud centroid**, makes a *virtual* camera look:

      - ``"side"``: horizontally at the object (undo the downward tilt) — virtual forward
        is the horizontal projection of the current forward axis (same azimuth).
      - ``"top"``: straight down at the object — virtual forward is gravity (down).
      - ``"none"``: identity.

    Rotating about the centroid preserves the object's range to the (virtual) camera, so
    the cloud stays at the ~0.4-0.7 m standoff GraspNet expects. ``up_opt`` is world-up
    expressed in the optical frame (``snap.cam.R.T @ [0,0,1]``); gravity = ``-up_opt``.

    When *center_xy* is set, the centroid is additionally repositioned so its **XY** sits
    on the optical axis (lateral offset zeroed) while its **depth (Z) is kept** — the
    object stays at the standoff GraspNet trained on; zeroing Z would put it at the camera,
    out of distribution. This is orthogonal to the rotation: it applies in every mode
    (including ``"none"``), as a pure translation. GraspNet-1Billion's PointNet++ backbone
    is *approximately* translation-invariant, so in theory this is a near no-op — it exists
    to A/B whether this particular server has any lateral-position dependence (see
    manual_tests/grasp_virtual_view.py).

    Returns ``(R_rel, c_in, c_out)``: ``R_rel`` a ``(3, 3)`` proper rotation (identity for
    ``"none"`` or a degenerate gravity reference); ``c_in`` the ``(3,)`` median point the
    rotation pivots about; ``c_out`` where that pivot lands afterwards — equal to ``c_in``
    unless ``center_xy`` zeroes its XY.
    """
    pts = np.asarray(cloud_opt, dtype=float)
    c = np.median(pts, axis=0) if pts.shape[0] else np.zeros(3)
    # Optional pure-translation recentring: drop the centroid's XY onto the optical axis,
    # keep its depth. Orthogonal to the rotation, so it is computed once for every mode.
    c_out = np.array([0.0, 0.0, c[2]]) if center_xy else c.copy()
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return np.eye(3), c, c_out
    if mode not in ("side", "top"):
        raise ValueError(f"mode must be 'none', 'side', or 'top'; got {mode!r}")

    u = np.asarray(up_opt, dtype=float).reshape(3)
    nu = float(np.linalg.norm(u))
    if nu < 1e-9:  # no gravity reference — leave the cloud's orientation untouched
        return np.eye(3), c, c_out
    u = u / nu

    fwd = np.array([0.0, 0.0, 1.0])  # current optical viewing axis (camera at origin)
    if mode == "side":
        target = fwd - (fwd @ u) * u  # drop the vertical component -> horizontal look
        nt = float(np.linalg.norm(target))
        if nt < 1e-9:  # forward is (anti)parallel to up — nothing sensible to do
            return np.eye(3), c, c_out
        target = target / nt
    else:  # "top"
        target = -u  # look straight down (gravity)

    # `target` is the world-direction we want the *virtual camera* to look along (in
    # optical coords). Rotating the cloud by R_rel is equivalent to moving the camera by
    # R_rel.T, so the virtual viewing axis is ``R_rel.T @ fwd``. To make that equal
    # `target` we need ``R_rel @ target = fwd`` — i.e. the rotation taking target -> fwd,
    # NOT fwd -> target. (Equivalently: this maps the object's target-facing surface
    # normal onto virtual -Z so it faces the virtual camera.)
    return _rotation_between(target, fwd), c, c_out


def _apply_virtual_view(
    cloud_opt: np.ndarray, R_rel: np.ndarray, c_in: np.ndarray, c_out: np.ndarray | None = None
) -> np.ndarray:
    """Rigidly remap an optical cloud: ``p -> R_rel @ (p - c_in) + c_out``.

    Rotates about ``c_in`` then places that pivot at ``c_out`` (defaults to ``c_in`` — a
    pure rotation about the centroid). A ``c_out`` differing only in XY recentres the cloud
    laterally without touching its depth or orientation.
    """
    c_in = np.asarray(c_in, dtype=float)
    c_out = c_in if c_out is None else np.asarray(c_out, dtype=float)
    pts = np.asarray(cloud_opt, dtype=float)
    return (pts - c_in) @ np.asarray(R_rel, dtype=float).T + c_out


def _invert_grasp_virtual(
    rotation: np.ndarray, translation, R_rel: np.ndarray,
    c_in: np.ndarray, c_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a grasp returned in the virtual frame back to the true optical frame.

    Inverse of :func:`_apply_virtual_view`: ``rot_opt = R_rel.T @ rot_v`` and
    ``t_opt = R_rel.T @ (t_v - c_out) + c_in``. With ``c_out is None`` this collapses to the
    pure-rotation case (``c_out = c_in``). Returns ``(rot_opt (3,3), t_opt (3,))``.
    """
    R_rel = np.asarray(R_rel, dtype=float)
    c_in = np.asarray(c_in, dtype=float).reshape(3)
    c_out = c_in if c_out is None else np.asarray(c_out, dtype=float).reshape(3)
    t_v = np.asarray(translation, dtype=float).reshape(3)
    rot_opt = R_rel.T @ np.asarray(rotation, dtype=float)
    t_opt = R_rel.T @ (t_v - c_out) + c_in
    return rot_opt, t_opt


def _grasp_to_optical(g, R_rel: np.ndarray, c_in: np.ndarray, c_out: np.ndarray | None = None):
    """Return a copy of grasp *g* (virtual frame) re-expressed in the true optical frame.

    Duck-typed on the client ``GraspPose`` dataclass (``rotation``/``translation``); the
    grasp's frame-relative geometry (approach, width, score) is preserved — only the
    frame it is expressed in changes. Ready to hand to :func:`_to_map_frame` unchanged.
    """
    rot_opt, t_opt = _invert_grasp_virtual(g.rotation, g.translation, R_rel, c_in, c_out)
    return replace(
        g, rotation=rot_opt,
        translation=(float(t_opt[0]), float(t_opt[1]), float(t_opt[2])),
    )


def _resolve_virtual_view(setting: str, approach_preference: str) -> str:
    """Resolve the ``WALKIE_GRASP_VIRTUAL_VIEW`` setting, expanding ``"auto"``.

    ``"auto"`` couples the transform to the caller's *approach_preference* — so passing
    ``approach_preference="side"`` both re-ranks AND rotates the cloud to a side view
    (``"top"`` likewise; any other preference -> ``"none"``). Explicit ``"none"`` /
    ``"side"`` / ``"top"`` pass through unchanged and are validated downstream by
    :func:`_virtual_view_rotation`. Note the coupled "side" combo was the weakest arm in
    the replay experiment (manual_tests/grasp_virtual_view.py), so ``"auto"`` is opt-in;
    the default stays ``"none"``.
    """
    setting = (setting or "none").strip().lower()
    if setting != "auto":
        return setting
    pref = (approach_preference or "none").strip().lower()
    return pref if pref in ("side", "top") else "none"


@dataclass
class GraspCandidate:
    """The best grasp found, expressed in the **map frame**.

    ``grasp_xyz`` is the gripper-closing centre; ``pregrasp_xyz`` is that point
    backed off ``standoff_m`` along the *negative* approach direction — where the
    gripper should arrive before driving straight in. ``rotation`` is the 3x3 grasp
    frame in the map frame (column 0 = approach/travel, column 1 = closing/spread);
    ``approach`` is its unit approach axis. ``score`` is GraspNet's quality and
    ``width`` the gripper opening in metres.
    """

    grasp_xyz: tuple[float, float, float]
    pregrasp_xyz: tuple[float, float, float]
    rotation: np.ndarray  # (3, 3) in the map frame
    approach: np.ndarray  # (3,) unit approach direction in the map frame
    width: float
    score: float
    # Filled in after best-of-N selection (see get_object_grasp_pos): the surface the
    # grasped object was sitting on, and the grasp height above it — remembered so the
    # object can be placed back at the same relative height on another surface. ``None``
    # when no support surface was found (or detection was skipped).
    support_surface_z: float | None = None
    grasp_to_surface_offset: float | None = None
    object_footprint_m: float | None = None  # map-frame XY span of the grasped cloud


@dataclass
class ObjectLocation:
    """A cheap detect+lift result — *position only*, no GraspNet (see locate_object).

    ``xyz_map`` is the map-frame centroid of the nearest matching detection;
    ``cloud_optical`` is that detection's lifted cloud in the camera-optical frame
    (camera at the origin, +Z forward — the frame GraspNet wants). ``snap`` is the
    snapshot it came from (carries ``cam.R``/``cam.t`` for re-framing). ``range_m``
    is the camera-frame median range (nearest object wins).
    """

    xyz_map: tuple[float, float, float]
    cloud_optical: np.ndarray
    snap: object
    range_m: float
    confidence: float | None = None  # detector confidence (drives multi-tilt tie-break)
    clip_sim: float | None = None  # CLIP cosine to the target (None on the nearest-wins path)


def _to_map_frame(snap, g, standoff_m: float) -> GraspCandidate:
    """Lift one GraspNet pose (optical frame) into the snapshot's map frame."""
    R_cam, t_cam = snap.cam.R, snap.cam.t
    grasp = R_cam @ np.asarray(g.translation, dtype=float) + t_cam
    R_map = R_cam @ g.rotation
    approach = R_map[:, 2]  # unit travel direction toward the object
    pregrasp = grasp - approach * standoff_m
    return GraspCandidate(
        grasp_xyz=(float(grasp[0]), float(grasp[1]), float(grasp[2])),
        pregrasp_xyz=(float(pregrasp[0]), float(pregrasp[1]), float(pregrasp[2])),
        rotation=R_map,
        approach=approach,
        width=float(g.width),
        score=float(g.score),
    )


def _optical_ref(loc: "ObjectLocation") -> np.ndarray:
    """The located object's map-frame centroid, expressed in its snapshot's optical frame.

    The cleanup pass (:func:`_clean_object_cloud`) keeps the cluster nearest this point,
    so it must be in the same frame as the cloud GraspNet receives (optical). ``xyz_map``
    is a robust detection median, so it survives the very bleed the cleanup removes.
    """
    return (np.asarray(loc.xyz_map, dtype=float) - loc.snap.cam.t) @ loc.snap.cam.R


def _detect_and_lift(
    ctx: TaskContext,
    snap,
    prompts: list[str],
    *,
    voxel: float,
    erode_px: int,
    min_points: int,
    min_confidence: float,
) -> tuple[np.ndarray, float, float | None, float | None] | None:
    """One detect (descriptor prompts + per-detection embed) -> the chosen object's cloud.

    Shared by :func:`locate_object` and the best-of-N loop in
    :func:`get_object_grasp_pos`. Expands *prompts* with visual descriptors, runs masked
    detection (requesting per-crop CLIP embeddings in the SAME round trip), drops
    detections below *min_confidence*, and selects:

      * **CLIP-similarity-first** — when rerank is on and there is more than one box to
        disambiguate, embed the SPECIFIC target text once and keep the highest-cosine box
        that lifts to ``>= min_points``;
      * **nearest-wins fallback** — when rerank is off, no target/embeddings are available,
        or every CLIP pick lifts too sparse: the smallest-median-range box (today's rule).

    Returns ``(cloud_optical, range_m, det_confidence, clip_sim)`` or ``None`` (never
    raises). ``clip_sim`` is ``None`` on the fallback path.
    """
    det_prompts = _detection_prompts(ctx, prompts)
    rerank_on = _b("WALKIE_GRASP_CLIP_RERANK", "1")

    detections = None
    if rerank_on:
        try:
            # detect() can't pass per_detection, so call process() directly to get masks
            # AND per-crop CLIP embeddings in one round trip (client/image.py).
            res = ctx.walkieAI.image.process(
                snap.img,
                detection={"prompts": det_prompts, "return_mask": True},
                per_detection={"embed": True},
            )
            detections = res.detection or []
        except Exception as exc:  # noqa: BLE001 — degrade to a plain masked detect
            print(f"[grasp] detect+embed failed ({exc}); falling back to plain detect")
            detections = None
    if detections is None:
        detections = ctx.walkieAI.image.detect(snap.img, prompts=det_prompts, return_mask=True)

    detections = [
        d for d in detections
        if d.mask is not None
        and (d.confidence is None or d.confidence >= min_confidence)
    ]
    if not detections:
        return None

    def _lift(det):
        pts = snap.mask_to_points(det.mask, voxel=voxel, frame="optical", erode_px=erode_px)
        if pts.shape[0] < min_points:
            return None
        return pts, float(np.median(np.linalg.norm(pts, axis=1)))

    # CLIP rerank — only worth a text-embed call when there's more than one box to tell apart.
    if rerank_on and len(detections) > 1:
        text_emb = _target_text_embedding(ctx, prompts[0] if prompts else "")
        scored = _rank_by_clip(
            detections, text_emb,
            sim_floor=_f("WALKIE_GRASP_CLIP_SIM_THRESHOLD", "0.0"),
        )
        for det, sim in scored:
            lifted = _lift(det)
            if lifted is not None:
                pts, rng = lifted
                return pts, rng, det.confidence, sim

    # Fallback: nearest-wins among the (expanded) detections — today's selector.
    best = None
    for det in detections:
        lifted = _lift(det)
        if lifted is None:
            continue
        pts, rng = lifted
        if best is None or rng < best[1]:
            best = (pts, rng, det.confidence, None)
    return best


def locate_object(
    ctx: TaskContext,
    prompts: list[str],
    *,
    voxel: float = 0.005,
    erode_px: int = 5,
    min_points: int = 50,
    min_confidence: float = 0.3,
    snap=None,
) -> ObjectLocation | None:
    """Cheap detect+lift of the nearest object matching *prompts* — NO GraspNet.

    The fast "where is it" primitive ``pick_object`` uses to position the base/head
    before committing to the (expensive) grasp plan. Takes a snapshot (or reuses
    *snap*), runs masked open-vocab detection (descriptor prompts + CLIP rerank — see
    :func:`_detect_and_lift`), drops detections below *min_confidence*, and lifts the
    chosen detection's mask to a camera-optical cloud: the **CLIP-best** box matching
    the target when rerank disambiguates several, else the **nearest** (smallest median
    range). Returns the lifted cloud plus the map-frame centroid, or ``None`` (never
    raises) when nothing graspable is in view.
    """
    snap = snap if snap is not None else ctx.snapshot()
    if snap is None or not snap.has_geometry:
        print("[grasp] locate: no snapshot geometry (is the ZED running?)")
        return None

    found = _detect_and_lift(
        ctx, snap, prompts, voxel=voxel, erode_px=erode_px,
        min_points=min_points, min_confidence=min_confidence,
    )
    if found is None:
        print(f"[grasp] locate: no graspable detection for {prompts} "
              f"(confidence >= {min_confidence}, lifted >= {min_points} pts)")
        return None
    cloud, nearest_range, nearest_conf, clip_sim = found

    cm = cloud @ snap.cam.R.T + snap.cam.t
    xyz = np.median(cm, axis=0)
    xyz_map = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    sim_str = f" clip={clip_sim:.3f}" if clip_sim is not None else ""
    print(f"[grasp] locate: {prompts} at ({xyz_map[0]:+.2f},{xyz_map[1]:+.2f},"
          f"{xyz_map[2]:+.2f}) range={nearest_range:.2f}m conf={nearest_conf}{sim_str}")
    return ObjectLocation(
        xyz_map=xyz_map, cloud_optical=cloud, snap=snap,
        range_m=nearest_range, confidence=nearest_conf, clip_sim=clip_sim,
    )


def _grasp_cloud_multi_tilt(
    ctx: TaskContext,
    prompts: list[str],
    *,
    tilts: tuple[float, ...] | None = None,
    settle_sec: float | None = None,
    dedup_radius_m: float | None = None,
    voxel: float = 0.002,
    erode_px: int = 5,
    min_points: int = 50,
    min_confidence: float = 0.3,
    merge_voxel: float | None = None,
) -> tuple[np.ndarray, object, np.ndarray] | None:
    """Two snapshots at different head tilts → one dense, deduped optical cloud.

    Tilts the head to each of *tilts* (radians, +down — bypasses ``look_at_object``'s
    0.436-rad clamp so we can use the shallow operator-tuned angles), locates the
    nearest object at each, deduplicates them by 3D proximity (same physical object
    seen twice), and fuses their clouds into ONE denser cloud expressed in the first
    view's optical frame — ready for a single GraspNet inference. A denser cloud (two
    viewpoints fill each other's self-occlusion) yields markedly better grasps than a
    single shot.

    Dedup is the cheap spatial half of the ``services.realtime_explore`` fusion: two
    views are the same object iff their map-frame centroids are within
    *dedup_radius_m*. On a mismatch (the second view's nearest was a different object)
    we keep the higher-confidence view rather than fuse mismatched clouds.

    Returns ``(cloud_optical, chosen_snap)`` or ``None`` (never raises). Degrades to a
    single view when only one tilt yields geometry.
    """
    tilts = tilts if tilts is not None else (
        _f("WALKIE_GRASP_TILT_A", "0.2"), _f("WALKIE_GRASP_TILT_B", "0.35"),
    )
    settle_sec = _f("WALKIE_GRASP_TILT_SETTLE_SEC", "0.4") if settle_sec is None else settle_sec
    dedup_radius_m = (
        _f("WALKIE_GRASP_DEDUP_RADIUS_M", "0.10") if dedup_radius_m is None else dedup_radius_m
    )
    merge_voxel = _f("WALKIE_GRASP_MERGE_VOXEL_M", "0.003") if merge_voxel is None else merge_voxel

    locs: list[ObjectLocation] = []
    for t in tilts:
        tilt_head(ctx, t, settle=settle_sec)
        loc = locate_object(
            ctx, prompts, voxel=voxel, erode_px=erode_px,
            min_points=min_points, min_confidence=min_confidence,
        )
        if loc is not None:
            locs.append(loc)

    if not locs:
        print(f"[grasp] multi-tilt: no object located across tilts {tilts}")
        return None
    if len(locs) == 1:
        print("[grasp] multi-tilt: only one view yielded geometry; using it alone")
        return locs[0].cloud_optical, locs[0].snap, _optical_ref(locs[0])

    # Dedup the two views: same object iff their map-frame centroids are close.
    a, b = locs[0], locs[1]
    sep = float(np.linalg.norm(np.asarray(a.xyz_map) - np.asarray(b.xyz_map)))
    if sep > dedup_radius_m:
        # Different objects in the two views — don't fuse mismatched clouds. Keep
        # the higher-confidence one (None treated lowest), tie-break on nearer range.
        def _key(loc: ObjectLocation) -> tuple[float, float]:
            return (loc.confidence if loc.confidence is not None else -1.0, -loc.range_m)

        keep = max(locs, key=_key)
        print(f"[grasp] multi-tilt: views {sep:.2f}m apart (> {dedup_radius_m:.2f}m); "
              f"not fusing, keeping conf={keep.confidence} range={keep.range_m:.2f}m")
        return keep.cloud_optical, keep.snap, _optical_ref(keep)

    # Fuse: lift each view's optical cloud to MAP using its OWN pose (common frame),
    # merge + voxel-downsample, then re-frame into the chosen (first) view's optical
    # frame so GraspNet sees one consistent cloud.
    chosen = locs[0]
    try:
        clouds_map = []
        for loc in (a, b):
            clouds_map.append(loc.cloud_optical @ loc.snap.cam.R.T + loc.snap.cam.t)
        merged_map = voxel_downsample(np.vstack(clouds_map), merge_voxel)
        merged_optical = (merged_map - chosen.snap.cam.t) @ chosen.snap.cam.R
        print(f"[grasp] multi-tilt: fused {a.cloud_optical.shape[0]}+{b.cloud_optical.shape[0]} "
              f"-> {merged_optical.shape[0]} pts (sep {sep:.2f}m)")
        return merged_optical, chosen.snap, _optical_ref(chosen)
    except Exception as exc:  # noqa: BLE001 — fall back to the single chosen view
        print(f"[grasp] multi-tilt: fuse failed ({exc}); using first view alone")
        return chosen.cloud_optical, chosen.snap, _optical_ref(chosen)


def _grasp_cloud_multi_snap(
    ctx: TaskContext,
    prompts: list[str],
    *,
    snaps: int | None = None,
    settle_sec: float | None = None,
    dedup_radius_m: float | None = None,
    voxel: float = 0.002,
    erode_px: int = 5,
    min_points: int = 50,
    min_confidence: float = 0.3,
    merge_voxel: float | None = None,
) -> tuple[np.ndarray, object, np.ndarray] | None:
    """*snaps* snapshots at the CURRENT head angle → one cleaner, deduped optical cloud.

    The head holds still — unlike :func:`_grasp_cloud_multi_tilt`, which moves between
    shots (and which ``pick_object`` no longer uses, because fusing across two head
    tilts depends on each tilt's pose being exact and misaligns when it isn't). Here we
    just capture the SAME view *snaps* times and fuse, so there is no cross-pose error
    to introduce. At a fixed camera pose the per-view clouds overlap voxel-for-voxel, so
    fusing does NOT add self-occlusion-filling density; instead ``voxel_downsample``
    returns the *mean* point per cell, so more samples per voxel means **less depth
    jitter**, and a voxel that dropped out (NaN) in one frame is recovered from another.
    The result is a lower-noise, fewer-holes cloud at roughly single-shot density —
    which keeps it in-distribution for GraspNet while giving it a steadier surface.

    Locates the nearest object per frame, drops any frame whose nearest landed beyond
    *dedup_radius_m* of the consensus (a flickering false positive elsewhere in view,
    not the target), and fuses the survivors into the consensus view's optical frame.
    Each survivor's cloud is lifted to MAP via its OWN capture pose before merging, so
    the fuse stays correct even if the base/head drifted a hair between shots.

    Returns ``(cloud_optical, chosen_snap)`` or ``None`` (never raises). Degrades to a
    single view when only one snapshot yields geometry.
    """
    snaps = _i("WALKIE_GRASP_FUSE_SNAPS", "3") if snaps is None else snaps
    settle_sec = _f("WALKIE_GRASP_FUSE_SETTLE_SEC", "0.15") if settle_sec is None else settle_sec
    dedup_radius_m = (
        _f("WALKIE_GRASP_DEDUP_RADIUS_M", "0.10") if dedup_radius_m is None else dedup_radius_m
    )
    merge_voxel = _f("WALKIE_GRASP_MERGE_VOXEL_M", "0.003") if merge_voxel is None else merge_voxel

    locs: list[ObjectLocation] = []
    for i in range(max(1, snaps)):
        if i > 0 and settle_sec > 0:
            time.sleep(settle_sec)  # let the camera serve a fresh, independent frame
        loc = locate_object(
            ctx, prompts, voxel=voxel, erode_px=erode_px,
            min_points=min_points, min_confidence=min_confidence,
        )
        if loc is not None:
            locs.append(loc)

    if not locs:
        print(f"[grasp] multi-snap: no object located across {snaps} snapshot(s)")
        return None
    if len(locs) == 1:
        print("[grasp] multi-snap: only one snapshot yielded geometry; using it alone")
        return locs[0].cloud_optical, locs[0].snap, _optical_ref(locs[0])

    # Consensus anchor = the nearest-to-camera view (least likely a far false positive,
    # mirroring locate_object's nearest-wins rule). Keep every frame whose object
    # centroid is within dedup_radius_m of it; drop frames that locked onto something
    # else so we never average two different objects' clouds together.
    chosen = min(locs, key=lambda loc: loc.range_m)
    kept = [
        loc for loc in locs
        if float(np.linalg.norm(np.asarray(loc.xyz_map) - np.asarray(chosen.xyz_map)))
        <= dedup_radius_m
    ]
    if len(kept) == 1:
        print(f"[grasp] multi-snap: {len(locs)} views disagree on position "
              f"(> {dedup_radius_m:.2f}m apart); using the nearest alone")
        return chosen.cloud_optical, chosen.snap, _optical_ref(chosen)

    # Fuse: lift each kept view's optical cloud to MAP via its OWN pose (common frame),
    # concatenate + voxel-downsample (the mean-per-cell averages out per-voxel depth
    # noise and fills single-frame dropouts), then re-frame back into the consensus
    # view's optical frame so GraspNet sees one consistent cloud.
    try:
        clouds_map = [loc.cloud_optical @ loc.snap.cam.R.T + loc.snap.cam.t for loc in kept]
        merged_map = voxel_downsample(np.vstack(clouds_map), merge_voxel)
        merged_optical = (merged_map - chosen.snap.cam.t) @ chosen.snap.cam.R
        print(f"[grasp] multi-snap: fused {len(kept)}/{len(locs)} views "
              f"-> {merged_optical.shape[0]} pts")
        return merged_optical, chosen.snap, _optical_ref(chosen)
    except Exception as exc:  # noqa: BLE001 — fall back to the nearest view alone
        print(f"[grasp] multi-snap: fuse failed ({exc}); using nearest view alone")
        return chosen.cloud_optical, chosen.snap, _optical_ref(chosen)


def get_object_grasp_pos(
    ctx: TaskContext,
    prompts: list[str],
    *,
    attempts: int = 5,
    standoff_m: float = 0.10,
    voxel: float = 0.001,
    erode_px: int = 2,
    min_points: int = 50,
    min_confidence: float = 0.2,
    antipodal: bool = True,
    approach_preference: str = "none",
    approach_weight: float | None = None,
    compute_support: bool = True,
    prebuilt: tuple[np.ndarray, object] | None = None,
    multi_tilt: bool = False,
    tilts: tuple[float, ...] | None = None,
    fuse_snaps: int | None = None,
) -> GraspCandidate | None:
    """Best-of-N grasp for the nearest object matching *prompts*, in the map frame.

    Captures up to *attempts* snapshots; on each it runs masked open-vocab
    detection for *prompts*, drops detections below *min_confidence*, lifts the
    **nearest** surviving detection's mask to a camera-optical cloud, and asks
    GraspNet for the single best grasp. The highest-scoring grasp
    across all attempts wins, mapped to the map frame against the geometry of the
    very snapshot it came from (accurate even after detection/GraspNet latency).

    Args:
        ctx: Task context (camera, AI client).
        prompts: Open-vocab detector prompts for the target (e.g. ``["red can"]``).
        attempts: How many snapshots to take and score (best-of-N).
        standoff_m: Pre-grasp back-off distance along the approach axis (metres).
        voxel: Voxel-downsample size for the lifted object cloud (metres).
        erode_px: Mask erosion before lifting, to shed rim/background pixels.
        min_points: Skip a detection whose lifted cloud is smaller than this.
        min_confidence: Drop detections whose detector confidence is below this
            (detections with no confidence reported are kept). Among the survivors,
            the one closest to the camera is grasped.
        antipodal: Run GraspNet's antipodal surface-normal validation.
        approach_preference: Bias grasp selection by approach direction relative to
            gravity: ``"side"`` favours horizontal approaches (e.g. grabbing a can
            around its side under a high fixed camera), ``"top"`` favours approaches
            pointing straight down (e.g. a spoon lying flat), ``"none"`` leaves
            GraspNet's ranking untouched. The "up" reference is derived
            automatically from each snapshot's pose (the map frame's +Z gravity axis,
            expressed in the cloud's optical frame), so the caller need not supply it.
        approach_weight: How strongly the preference outranks GraspNet's own score
            (server default ~1.0; higher favours the preferred approach harder). Only
            used when ``approach_preference`` is set; ``None`` keeps the server default.
        prebuilt: A pre-built ``(optical_cloud, snap)`` to grasp directly — skips the
            best-of-N snapshot loop and runs GraspNet ONCE on the given cloud. Used by
            the multi-tilt fast path (see ``_grasp_cloud_multi_tilt``).
        multi_tilt: When True (and *prebuilt* is None), build the cloud with
            :func:`_grasp_cloud_multi_tilt` (2 snapshots at *tilts*, deduped + fused)
            and run GraspNet once. The cheap, accurate path ``pick_object`` uses after
            it has already approached the object.
        tilts: Head-tilt angles (rad, +down) for the multi-tilt build; ``None`` uses
            the ``WALKIE_GRASP_TILT_A``/``_B`` config defaults.
        fuse_snaps: When > 1 (and neither *prebuilt* nor *multi_tilt* is set), take this
            many snapshots at the **current head angle**, fuse their object clouds into
            one lower-noise / fewer-dropout cloud (:func:`_grasp_cloud_multi_snap`), and
            run GraspNet **once** on it — instead of the best-of-N loop (which scores
            *attempts* snapshots separately and keeps the highest). ``None`` reads
            ``WALKIE_GRASP_FUSE_SNAPS`` (code default ``1`` = best-of-N; the repo's
            ``config.toml`` sets ``3`` to fuse). Unlike *multi_tilt*, the head never
            moves, so there is no cross-pose misalignment to spoil the fuse.

    Returns:
        The winning :class:`GraspCandidate` (with ``grasp_xyz`` and
        ``pregrasp_xyz`` in the map frame), or ``None`` if no attempt produced a
        graspable detection.
    """
    best: GraspCandidate | None = None
    best_snap = None  # the snapshot the winning grasp came from (for surface lookup)
    best_cloud: np.ndarray | None = None  # the winning object cloud (optical frame)

    def _infer_best(cloud: np.ndarray, snap, ref_optical: np.ndarray | None = None) -> bool:
        """Run GraspNet once on *cloud* (optical frame) and keep the best result.

        Cleans the cloud first (:func:`_clean_object_cloud`) so GraspNet never sees the
        background-bleed blob, and remembers the *cleaned* cloud so the object footprint
        is measured on real object points only.
        """
        nonlocal best, best_snap, best_cloud
        raw_cloud = cloud
        cloud = _clean_object_cloud(cloud, ref_optical=ref_optical)
        _draw_cloud_viz(ctx, snap, raw_cloud, cloud)

        # Optional virtual-viewpoint transform (WALKIE_GRASP_VIRTUAL_VIEW, default "none").
        # Rigidly rotate the cleaned cloud into a "side"/"top" virtual view so GraspNet
        # *generates* grasps for that approach (which re-ranking alone cannot — it only
        # reorders what GraspNet already proposes), then map the winning grasp back to the
        # true optical frame so everything downstream is unchanged. Evidence in
        # manual_tests/grasp_virtual_view.py: on a downward-tilted camera "top" can unlock
        # steeper top-down grasps the oblique view never proposes, while "side" tends to
        # produce unreachable grasps (the object's sides aren't captured at a down tilt).
        # The footprint below is still measured on the UN-rotated `cloud`.
        #
        # Optional XY-recentring (WALKIE_GRASP_CENTER_XY, default off): drop the cloud's
        # lateral offset onto the optical axis before inference (depth kept), mapping the
        # winning grasp back afterwards. Orthogonal to the rotation above — it composes with
        # any vview, including "none". Theory says GraspNet (PointNet++) is ~translation-
        # invariant so this is near no-op; it's a knob to A/B that on the real server.
        up_opt = snap.cam.R.T @ np.array([0.0, 0.0, 1.0])
        vview = _resolve_virtual_view(
            os.getenv("WALKIE_GRASP_VIRTUAL_VIEW", "none"), approach_preference
        )
        center_xy = _b("WALKIE_GRASP_CENTER_XY", "0")
        v_rot, v_center, v_center_out = _virtual_view_rotation(
            cloud, up_opt, vview, center_xy=center_xy
        )
        cloud_infer = _apply_virtual_view(cloud, v_rot, v_center, v_center_out)

        infer_kwargs: dict = {"antipodal": antipodal, "max_grasps": 10}
        # Optional: feed GraspNet a finer/denser cloud (off by default). The cleaned object
        # cloud carries detail the server re-collapses at its 0.005 m voxel; a smaller voxel
        # + larger sample preserves it for better grasp-point localization on small objects.
        server_voxel = os.getenv("WALKIE_GRASP_SERVER_VOXEL_M", "").strip()
        if server_voxel:
            infer_kwargs["voxel_size"] = float(server_voxel)
        server_npts = os.getenv("WALKIE_GRASP_SERVER_NUM_POINT", "").strip()
        if server_npts:
            infer_kwargs["num_point"] = int(server_npts)
        if approach_preference != "none":
            infer_kwargs["approach_preference"] = approach_preference
            if approach_weight is not None:
                infer_kwargs["approach_weight"] = approach_weight
            # Hard-drop "bottom-up" grasps: the max allowed approach·up. 0.0 keeps only
            # at/below-horizontal approaches (the true bottom-hemisphere cut); the server
            # default (~0.2) tolerates ~11.5deg of upward tilt. Sent per-request so it
            # holds even against a remote/shared server whose own default differs. Empty
            # = leave the server default. Only the server's side/top filter reads it.
            max_up = os.getenv("WALKIE_GRASP_MAX_APPROACH_UP", "0.0").strip()
            if max_up:
                infer_kwargs["max_approach_up"] = float(max_up)
        # Gravity expressed in the (possibly rotated) frame of the cloud sent — once a
        # transform is active this is NOT global up: it works out to -Y for "side" and -Z
        # for "top" (v_rot @ up_opt). The re-rank needs it, and it also lets the client
        # orient each grasp's wrist X-up against the right "down" in the rotated frame.
        # Sent whenever the transform OR the re-rank is active, so the default
        # (no-transform, no-preference) path is byte-for-byte unchanged.
        if approach_preference != "none" or vview != "none":
            infer_kwargs["up"] = v_rot @ up_opt
        grasps = ctx.walkieAI.grasp.infer(cloud_infer, **infer_kwargs)
        if not grasps:
            print("[grasp] GraspNet returned nothing")
            return False
        # Back to the true optical frame before lifting to map (no-op when vview="none"
        # and center_xy off).
        g = _grasp_to_optical(grasps[0], v_rot, v_center, v_center_out)
        if best is not None and g.score <= best.score:
            return False
        best = _to_map_frame(snap, g, standoff_m)
        best_snap, best_cloud = snap, cloud
        gx, gy, gz = best.grasp_xyz
        print(f"[grasp] grasp score {best.score:.3f} "
              f"grasp=({gx:+.3f},{gy:+.3f},{gz:+.3f}) width={best.width * 100:.1f}cm")
        return True

    fuse_snaps = _i("WALKIE_GRASP_FUSE_SNAPS", "1") if fuse_snaps is None else fuse_snaps

    # Fast path: a single GraspNet inference on a cloud built up front, skipping the
    # best-of-N snapshot loop entirely. The cloud comes from one of:
    #   * prebuilt        — the caller already lifted it,
    #   * multi_tilt      — fused across two head tilts (_grasp_cloud_multi_tilt),
    #   * fuse_snaps > 1  — fused across N snapshots at the CURRENT angle, for
    #                       noise/dropout reduction (_grasp_cloud_multi_snap).
    if prebuilt is not None or multi_tilt or fuse_snaps > 1:
        if prebuilt is not None:
            # Caller-supplied (cloud, snap) — no located centroid, so cleanup falls
            # back to the cloud's own median as the nearest-cluster reference.
            built = (prebuilt[0], prebuilt[1], None)
        elif multi_tilt:
            built = _grasp_cloud_multi_tilt(
                ctx, prompts, tilts=tilts, voxel=voxel, erode_px=erode_px,
                min_points=min_points, min_confidence=min_confidence,
            )
        else:
            built = _grasp_cloud_multi_snap(
                ctx, prompts, snaps=fuse_snaps, voxel=voxel, erode_px=erode_px,
                min_points=min_points, min_confidence=min_confidence,
            )
        if built is None:
            print(f"[grasp] no graspable detection for {prompts} (fused build)")
            return None
        cloud, snap, ref_optical = built
        _infer_best(cloud, snap, ref_optical=ref_optical)
        if best is None:
            return None
    else:
        for i in range(attempts):
            tag = f"attempt {i + 1}/{attempts}"
            snap = ctx.snapshot()
            if snap is None or not snap.has_geometry:
                print(f"[grasp] {tag}: no snapshot geometry (is the ZED running?)")
                continue

            # Detect + CLIP-rerank + lift the chosen object via the shared path (so the
            # best-of-N loop disambiguates brand items the same way locate_object does).
            # Passing snap reuses this attempt's frame; ref_optical anchors the cleanup
            # on the detection centroid (the old inline loop passed none).
            loc = locate_object(
                ctx, prompts, voxel=voxel, erode_px=erode_px,
                min_points=min_points, min_confidence=min_confidence, snap=snap,
            )
            if loc is None:
                print(f"[grasp] {tag}: no graspable detection for {prompts}")
                continue

            _infer_best(loc.cloud_optical, loc.snap, ref_optical=_optical_ref(loc))

        if best is None:
            print(f"[grasp] no graspable detection for {prompts} in {attempts} attempt(s)")
            return None

    # Remember the support surface + object footprint from the winning snapshot, so the
    # object can be placed back later (tasks.skills.place). Computed against the RAW
    # grasp z, BEFORE the side-grasp nudge below, so the stored offset is the true
    # grasp-to-surface height (not inflated by the nudge).
    if compute_support and best_snap is not None:
        try:
            from interfaces.perception.surfaces import (
                detect_horizontal_surfaces,
                support_surface_for,
            )
            from tasks.skills.place import _full_scene_cloud, _surface_kwargs

            scene = _full_scene_cloud(best_snap)
            surfaces = detect_horizontal_surfaces(scene, **_surface_kwargs())
            gx, gy, gz = best.grasp_xyz
            sup = support_surface_for(surfaces, gx, gy, gz)
            if sup is not None:
                best.support_surface_z = sup.z
                best.grasp_to_surface_offset = gz - sup.z
                print(f"[grasp] support surface z={sup.z:.2f}m "
                      f"(grasp {gz:.2f}m, offset {best.grasp_to_surface_offset:+.2f}m)")
            else:
                print("[grasp] no support surface found under the grasp")
        except Exception as exc:  # noqa: BLE001 — surface lookup is best-effort
            print(f"[grasp] support-surface lookup failed ({exc})")

    if best_cloud is not None and best_snap is not None and best_snap.cam is not None:
        try:
            cm = best_cloud @ best_snap.cam.R.T + best_snap.cam.t
            span = cm[:, :2].max(axis=0) - cm[:, :2].min(axis=0)
            best.object_footprint_m = float(max(span))
        except Exception as exc:  # noqa: BLE001
            print(f"[grasp] object footprint estimate failed ({exc})")

    # offset height a little for side grasps
    best.grasp_xyz = (best.grasp_xyz[0], best.grasp_xyz[1], best.grasp_xyz[2] + 0.03)
    return best


# ---------------------------------------------------------------------------
# Grasp execution: aim, approach, de-deadzone, and command the arm.
#
# The planner above only finds a map-frame grasp pose. The helpers below drive
# the robot to actually take it, handling Walkie's real constraints: the arms
# can't reach across the body centreline (a lateral dead-zone), the lift gives
# extra reach when commanded via the "*_arm_lift" groups, and objects too far /
# too low / off-camera need the base + head to reposition first.
# ---------------------------------------------------------------------------
def _arm_sides(arm: str) -> tuple[str, str, str]:
    """(motion_group, home_group, gripper_group) for an arm side.

    ``go_to_pose`` uses the lift group ("left_arm_lift") so MoveIt can solve the
    lift joint for extra reach; ``go_to_home`` uses the plain arm group, where the
    SRDF named poses (e.g. "hands_up") live. Unknown side -> warn + default left.
    """
    side = (arm or "left").strip().lower()
    if side not in ("left", "right"):
        print(f"[grasp] unknown arm {arm!r}; defaulting to left")
        side = "left"
    return f"{side}_arm_lift", f"{side}_arm", f"{side}_gripper"


def _world_to_base(ctx: TaskContext, xyz_map: Vec3) -> Vec3:
    """Map-frame point -> base_footprint (forward, left, z). Mirrors
    tasks.manipulation.world_to_base; inlined to keep grasp.py dependency-light."""
    ox, oy, oz = xyz_map
    p = ctx.current_pose()
    dx, dy = ox - p["x"], oy - p["y"]
    c, s = math.cos(p["heading"]), math.sin(p["heading"])
    return c * dx + s * dy, -s * dx + c * dy, oz


def _xy_dist(ctx: TaskContext, xyz_map: Vec3) -> float:
    """Planar distance from the robot base to a map-frame point (metres)."""
    p = ctx.current_pose()
    return math.hypot(xyz_map[0] - p["x"], xyz_map[1] - p["y"])


def _look_down_tilt(cam_t, xyz_map: Vec3) -> float:
    """Head tilt (rad, +down) that points the camera at *xyz_map*, clamped.

    ``cam_t`` is the camera optical-centre in the map frame. The pitch below the
    horizon is atan2(height_drop, horizontal_distance); positive when the object
    sits below the camera (the usual table-top case).
    """
    dx, dy = xyz_map[0] - cam_t[0], xyz_map[1] - cam_t[1]
    horiz = math.hypot(dx, dy)
    tilt = math.atan2(cam_t[2] - xyz_map[2], horiz)
    return max(_HEAD_TILT_MIN, min(_HEAD_TILT_MAX, tilt))


def _draw_cloud_viz(ctx: TaskContext, snap, raw_optical: np.ndarray, kept_optical: np.ndarray) -> None:
    """Best-effort: show the exact cloud handed to GraspNet and what the cleanup removed.

    Draws the kept/cleaned cloud (green) and the removed points (red) in the **map
    frame**, so the user can confirm at a glance that (a) the cloud sits on the object
    and is upright — **not vertically inverted** — and (b) the cleanup is stripping the
    table/wall bleed rather than the object. The **whole frame** is also drawn as gray
    dots behind them (``WALKIE_GRASP_SCENE_VIZ``, default on) so it is obvious where the
    green object sits relative to the table/surroundings. When ``WALKIE_GRASP_DEBUG_DUMP``
    names a directory, both optical clouds are also written there (``grasp_cloud_raw.npy``
    / ``grasp_cloud_kept.npy``) for offline inspection. Never load-bearing.
    """
    dump_dir = os.getenv("WALKIE_GRASP_DEBUG_DUMP", "").strip()
    if getattr(ctx, "viz", None) is None and not dump_dir:
        return
    try:
        raw = np.ascontiguousarray(np.asarray(raw_optical, dtype=np.float32))
        kept = np.ascontiguousarray(np.asarray(kept_optical, dtype=np.float32))
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            np.save(os.path.join(dump_dir, "grasp_cloud_raw.npy"), raw)
            np.save(os.path.join(dump_dir, "grasp_cloud_kept.npy"), kept)
        if getattr(ctx, "viz", None) is None:
            return
        cam = snap.cam
        kept_map = kept @ cam.R.T + cam.t
        # kept rows are a byte-identical subset of raw (both filters return raw subsets),
        # so an exact-bytes set membership recovers exactly the removed points.
        kept_keys = {row.tobytes() for row in kept}
        removed = raw[np.array([row.tobytes() not in kept_keys for row in raw], dtype=bool)] \
            if raw.shape[0] else raw
        ctx.viz.clear("grasp/cloud", recursive=True)
        # Whole-frame scene cloud (gray) as context behind the object cloud, so it's
        # obvious where the green object sits relative to the table/surroundings. The
        # lift is a local depth deprojection (no server call), capped by the place-scene
        # depth/voxel; gated by WALKIE_GRASP_SCENE_VIZ (default on) and best-effort.
        if _b("WALKIE_GRASP_SCENE_VIZ", "1"):
            try:
                from tasks.skills.place import _full_scene_cloud  # lazy: place imports grasp
                scene_map = _full_scene_cloud(snap)
                if scene_map.shape[0]:
                    ctx.viz.points("grasp/cloud/scene", scene_map,
                                   colors=[(150, 150, 150)], radii=[0.0025])
            except Exception as exc:  # noqa: BLE001 — scene context is never load-bearing
                print(f"[grasp] scene cloud viz failed ({exc})")
        ctx.viz.points("grasp/cloud/kept", kept_map, colors=[(40, 220, 40)], radii=[0.004])
        if removed.shape[0]:
            removed_map = removed @ cam.R.T + cam.t
            ctx.viz.points("grasp/cloud/removed", removed_map, colors=[(230, 40, 40)], radii=[0.006])
    except Exception as exc:  # noqa: BLE001 — viz/dump is never load-bearing
        print(f"[grasp] cloud viz failed ({exc})")


def _draw_grasp_viz(ctx: TaskContext, candidate: GraspCandidate) -> None:
    """Best-effort: drop the planned grasp markers into the shared viewer."""
    if getattr(ctx, "viz", None) is None:
        return
    try:
        ctx.viz.clear("grasp", recursive=True)
        ctx.viz.axes("grasp/ee", candidate.grasp_xyz, rotation=candidate.rotation,
                     length=0.10, labels=True)
        ctx.viz.points("grasp/pregrasp", [list(candidate.pregrasp_xyz)], radii=[0.02],
                       colors=[(255, 180, 0)], labels=["pregrasp"])
        approach = (np.asarray(candidate.grasp_xyz) - np.asarray(candidate.pregrasp_xyz)).tolist()
        ctx.viz.arrow("grasp/approach", candidate.pregrasp_xyz, approach, color=(255, 180, 0))
    except Exception as exc:  # noqa: BLE001 — viz is never load-bearing
        print(f"[grasp] viz failed ({exc})")


def look_at_object(ctx: TaskContext, xyz_map: Vec3) -> bool:
    """Tilt the head so the camera points at the map-frame object. Best-effort.

    Reads the live camera pose (cheap TF lookup, no RGB-D grab), computes the
    clamped look-down tilt, and commands the head servo. Returns False (never
    raises) when the camera pose is unavailable or the servo command fails.
    """
    cam = camera_pose(ctx.walkie)
    if cam is None:
        print("[grasp] look_at_object: no camera pose")
        return False
    tilt = _look_down_tilt(cam.t, xyz_map)
    try:
        # Limit tilt because graspnet is bad
        tilt = max(0.436332, tilt)
        ctx.walkie.robot.head.tilt(tilt)
        time.sleep(1)
    except Exception as exc:  # noqa: BLE001 — off-robot stub may lack robot.head
        print(f"[grasp] look_at_object: head tilt failed ({exc})")
        return False
    return True


def face_object(ctx: TaskContext, xyz_map: Vec3) -> bool:
    """Rotate the base so the robot heading points straight at the object.

    Centres the object in the camera's horizontal FOV so detection/GraspNet see
    it square-on — better masks, better grasps. One-shot rotate-in-place toward
    the map-frame point; best-effort, returns ``rotate_to``'s result.
    """
    p = ctx.current_pose()
    desired = math.atan2(xyz_map[1] - p["y"], xyz_map[0] - p["x"])
    return ctx.rotate_to(desired)


def _approach_once(
    ctx: TaskContext,
    xyz_map: Vec3,
    *,
    standoff_m: float,
    track: bool,
    tick_sec: float,
    timeout_sec: float,
) -> str:
    """One drive attempt toward the object. Returns ``"MOVED"`` or ``"FAILED"``.

    Factored out of :func:`approach_object` so the caller can retry on
    ``"FAILED"`` — Nav2 aborts here are frequently transient (planner failed to
    find a path that cycle, costmap not yet settled), so a plain re-issue usually
    succeeds.
    """
    ox, oy = float(xyz_map[0]), float(xyz_map[1])
    if not track:
        try:
            res = ctx.walkie.nav.go_to(
                x=ox, y=oy, blocking=True, standoff=standoff_m, align_method="nearest_edge",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[grasp] approach_object: nav raised ({exc})")
            return "FAILED"
        look_at_object(ctx, xyz_map)
        print(f"[grasp] approach_object: nav -> {res}")
        return "MOVED" # Tests
        # return "MOVED" if res in ("SUCCEEDED", "CLOSE_ENOUGH") else "FAILED"

    try:
        ctx.walkie.nav.go_to(
            x=ox, y=oy, blocking=False, standoff=standoff_m, align_method="nearest_edge",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] approach_object: nav raised ({exc})")
        return "FAILED"
    deadline = time.monotonic() + timeout_sec
    while ctx.walkie.nav.is_navigating and time.monotonic() < deadline:
        look_at_object(ctx, xyz_map)
        time.sleep(tick_sec)
    if ctx.walkie.nav.is_navigating:  # timed out while still driving
        print(f"[grasp] approach_object: timed out after {timeout_sec:.0f}s; cancelling")
        ctx.walkie.nav.cancel()
        return "FAILED"
    status = ctx.walkie.nav.status
    look_at_object(ctx, xyz_map)
    print(f"[grasp] approach_object: nav -> {status}")
    return "MOVED" if status in ("SUCCEEDED", "CLOSE_ENOUGH") else "FAILED"


def approach_object(
    ctx: TaskContext,
    xyz_map: Vec3,
    *,
    standoff_m: float = 0.60,
    trigger_m: float = 0.70,
    track: bool = True,
    tick_sec: float = 0.2,
    timeout_sec: float = 30.0,
    retries: int = 2,
    retry_settle_sec: float = 0.5,
    success_tolerance_m: float = 1.0,
    min_progress_m: float = 0.10,
) -> str:
    """Drive to a standoff facing the object, tilting the head to keep it in view.

    Uses ``nav.go_to`` with the heading omitted (NavigateToObject) and
    ``align_method="face_target"`` so the base ends up facing the object at
    *standoff_m* metres — short of a table edge. With *track* the drive is issued
    non-blocking and the head is re-aimed every *tick_sec* as the robot closes in
    (the object is static, so tracking just re-tilts as the distance shrinks).

    **A FAILED nav is not always a real failure.** When the requested standoff is
    physically unreachable (the classic case: an object in the *middle* of a table,
    so the standoff point sits inside the table footprint / Nav2 inflation layer),
    Nav2 drives the base to the closest reachable spot — the table edge — and then
    reports the goal aborted. But that closest spot is usually close enough to
    grasp from: the downstream arm-align servo only moves the base laterally, and
    the grasp is rejected only past the arm reach (``max_reach_xy_m`` ≈ 0.70 m).
    *success_tolerance_m* defaults a little beyond that reach (an on-robot tuning
    candidate now that no forward creep closes the gap), so a FAILED drive that
    left the base within it is accepted as ``"MOVED"``.

    Otherwise a FAILED drive is retried up to *retries* times — the lingering goal
    is cancelled, the base settles for *retry_settle_sec*, and the drive is
    re-issued (transient planner/costmap aborts usually clear on a re-issue). If a
    re-issue closes less than *min_progress_m* of distance, the base is already at
    the closest position Nav2 will give it, so retrying further is futile and the
    call stops early. The trigger distance is re-checked before each attempt, so a
    partial drive that closed the gap short-circuits to ``"CLOSE"``.

    Returns ``"CLOSE"`` (already within *trigger_m*, no move — head still aimed),
    ``"MOVED"`` (drove and the nav goal succeeded, or failed but the base is within
    *success_tolerance_m* — close enough to grasp from), or ``"FAILED"`` (every
    attempt aborted with the base still too far to reach the object).
    """
    ctx.walkie.robot.head.set_auto_tilt(False)
    attempts = max(1, retries + 1)
    prev_dist: float | None = None
    for attempt in range(attempts):
        if _xy_dist(ctx, xyz_map) <= trigger_m:
            look_at_object(ctx, xyz_map)
            return "CLOSE"

        status = _approach_once(
            ctx, xyz_map, standoff_m=standoff_m, track=track,
            tick_sec=tick_sec, timeout_sec=timeout_sec,
        )
        if status != "FAILED":
            return status

        # Nav aborted. Did the base still get close enough to grasp from? The
        # standoff goal is often unreachable (object mid-table) yet the base parked
        # at the closest reachable spot — accept that as success.
        dist = _xy_dist(ctx, xyz_map)
        if dist <= success_tolerance_m:
            print(f"[grasp] approach_object: nav FAILED but base within "
                  f"{dist:.2f}m <= {success_tolerance_m:.2f}m; accepting closest reachable")
            look_at_object(ctx, xyz_map)
            return "MOVED"

        # Still too far. A re-issue only helps if the last drive actually closed
        # distance; if it stalled, the base is at the closest Nav2 will give it and
        # retrying is pointless.
        if prev_dist is not None and (prev_dist - dist) < min_progress_m:
            print(f"[grasp] approach_object: nav FAILED, no further progress "
                  f"({dist:.2f}m, still > {success_tolerance_m:.2f}m); aborting")
            return "FAILED"
        prev_dist = dist

        if attempt + 1 < attempts:
            print(f"[grasp] approach_object: attempt {attempt + 1}/{attempts} failed at "
                  f"{dist:.2f}m; retrying")
            try:
                ctx.walkie.nav.cancel()  # clear any half-issued goal before re-aiming
            except Exception as exc:  # noqa: BLE001
                print(f"[grasp] approach_object: cancel before retry raised ({exc})")
            time.sleep(retry_settle_sec)
    ctx.walkie.robot.head.set_auto_tilt(True)
    print(f"[grasp] approach_object: all {attempts} attempt(s) failed "
          f"(base still > {success_tolerance_m:.2f}m from object)")
    return "FAILED"


def in_arm_deadzone(ctx: TaskContext, xyz_map: Vec3, *, half_width_m: float = 0.20) -> bool:
    """Whether the object sits in the central lateral dead-zone the arms can't reach.

    The arms can't rotate toward the robot centreline, so an object whose
    base_footprint lateral offset |y| is within *half_width_m* is unreachable
    even when dead ahead — the base must rotate first (see face_object_with_arm).
    """
    _, left, _ = _world_to_base(ctx, xyz_map)
    return -half_width_m <= left <= half_width_m


def face_object_with_arm(ctx: TaskContext, xyz_map: Vec3, *, arm: str = "left") -> bool:
    """Rotate the base so *arm* faces the object, lifting it out of the dead-zone.

    Looks up the arm's shoulder link (``openarm_{side}_link3``) in the map frame
    and rotates the base until the shoulder->object bearing is the robot heading
    (the arm's forward direction is taken to be the robot's heading). One-shot
    approximation: rotating the base also swings the shoulder on a small arc, so
    the post-turn aim is a few degrees off — but enough to move the object off the
    centreline and in front of that arm. Best-effort; False on no transform/odom.
    """
    side = (arm or "left").strip().lower()
    if side not in ("left", "right"):
        side = "left"
    frame = f"openarm_{side}_link3"
    try:
        tf = ctx.walkie.robot.transform.lookup("map", frame, timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] face_object_with_arm: transform.lookup({frame}) raised ({exc})")
        return False
    if not tf or "position" not in tf:
        print(f"[grasp] face_object_with_arm: no transform for {frame}; cannot face")
        return False
    link = tf["position"]
    desired = math.atan2(xyz_map[1] - link["y"], xyz_map[0] - link["x"])
    print(f"[grasp] face_object_with_arm[{side}]: link=({link['x']:+.2f},{link['y']:+.2f}) "
          f"obj=({xyz_map[0]:+.2f},{xyz_map[1]:+.2f}) -> heading {math.degrees(desired):+.0f}deg")
    return ctx.rotate_to(desired)


def align_arm_to_object(ctx: TaskContext, xyz_map: Vec3, *, arm: str = "left") -> bool:
    """Servo the base sideways until *arm* lines up laterally with the object.

    Walkie's base is omnidirectional, so instead of rotating to face the object
    with the arm (which swings the shoulder on an arc and re-aims the base), we
    *translate* the base sideways until the object's lateral offset in the base
    frame matches the arm's own lateral mounting offset — putting the object
    directly in front of that arm and out of the centreline dead-zone. Heading is
    held throughout (a pure strafe), so the map-frame grasp candidate stays valid
    (no re-plan needed).

    The drive is a closed-loop :func:`strafe_servo`: the object is static in the
    map frame, so the residual (object lateral offset − arm mounting offset) is
    recomputed from live odometry every tick and the strafe self-corrects until
    it is within tolerance. Obstacles are not a stop condition — the nav stack
    scales cmd_vel down near them — so a base that is commanded but not moving
    for a while is simply as far over as the world allows, and the servo stops
    there: good enough to grasp from. Returns True when aligned within tolerance,
    False when it stopped early (blocked/timeout/no odom) — callers treat either
    as good enough and carry on. Never raises. Also used by ``place_object`` to
    line the holding arm up with the placement spot.
    """
    side = (arm or "left").strip().lower()
    if side not in ("left", "right"):
        side = "left"
    frame = f"openarm_{side}_link3"
    arm_left = 0.0  # static mounting offset in base_footprint — one lookup is enough
    try:
        tf = ctx.walkie.robot.transform.lookup("base_footprint", frame, timeout=5.0)
        if tf and "position" in tf:
            arm_left = float(tf["position"]["y"])
    except Exception as exc:  # noqa: BLE001 — fall back to the centreline
        print(f"[grasp] align_arm_to_object: transform.lookup({frame}) raised ({exc}); "
              f"assuming arm on centreline")

    def _err() -> float | None:
        """Metres the base must still move LEFT, from the live pose; None = no fix."""
        try:
            return _world_to_base(ctx, xyz_map)[1] - arm_left
        except Exception:  # noqa: BLE001
            return None

    initial = _err()
    print(f"[grasp] align_arm_to_object[{side}]: arm_left={arm_left:+.2f}m "
          f"initial_err={initial if initial is None else format(initial, '+.2f')}m (servo)")
    return strafe_servo(
        ctx, _err,
        tol_m=_f("WALKIE_GRASP_SERVO_TOL_M", "0.015"),
        timeout_sec=_f("WALKIE_GRASP_SERVO_TIMEOUT_S", "12"),
    )


def aim_forward_candidate(
    ctx: TaskContext, candidate: GraspCandidate, *, standoff_m: float = 0.10,
) -> GraspCandidate:
    """Re-point a grasp's wrist straight along the robot heading (map frame).

    GraspNet's returned wrist orientation is often awkward / IK-unsolvable on
    OpenArm. Instead of "positioning the arm at the object's full grasp pose", this
    points the gripper's approach axis (EE **+z**) along the robot's forward heading
    — taking the arm's forward to be the robot's. ``pick_object`` has already faced
    the object, so "forward" is "at the object". The base-frame wrist orientation is
    read from ``WALKIE_GRASP_POINT_RPY`` (default ``"0,1.5708,0"`` -> EE +z = base
    +x, horizontal forward) and rotated into the map frame by the current heading.

    That fixed RPY only constrains the approach (EE +z); its roll about that axis
    can leave the wrist's X axis (rotation column 0) pointing **down**. We roll the
    wrist 180° about the approach whenever X points down so it always ends up
    pointing up — the parallel-jaw symmetry (fingers swap, approach unchanged),
    mirroring ``client.grasp._orient_x_up`` but with "down" taken as the map
    frame's -Z (true gravity). This holds regardless of the configured RPY.

    The grasp *point* is kept; only the orientation, approach axis, and pre-grasp
    (re-backed-off *standoff_m* along the new -forward axis) change. Returns a new
    :class:`GraspCandidate` so the same pose drives both the arm and the held-object
    record (so the placer sets the object back down the way it was grasped).
    """
    raw = os.getenv("WALKIE_GRASP_POINT_RPY", "0,1.5708,0")
    rpy_base = [float(p.strip()) for p in raw.split(",")]
    theta = ctx.current_pose()["heading"]
    R_base = Rotation.from_euler("xyz", rpy_base).as_matrix()
    R_map = Rotation.from_euler("z", theta).as_matrix() @ R_base
    if R_map[2, 0] < 0.0:  # X axis (col 0) points down in the map frame (-Z = down)
        R_map = R_map @ np.diag([-1.0, -1.0, 1.0])  # 180° about approach (Z, col 2)
        print("[grasp] aim_forward_candidate: rolled wrist 180° so X points up")
    approach = R_map[:, 2]  # gripper points this way (map frame)
    grasp = np.asarray(candidate.grasp_xyz, dtype=float)
    pregrasp = grasp - approach * standoff_m
    print(f"[grasp] aim_forward_candidate: heading={math.degrees(theta):+.0f}deg "
          f"approach=({approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f})")
    return replace(
        candidate,
        rotation=R_map,
        approach=approach,
        pregrasp_xyz=(float(pregrasp[0]), float(pregrasp[1]), float(pregrasp[2])),
    )


def execute_grasp(
    ctx: TaskContext,
    candidate: GraspCandidate,
    *,
    arm: str = "left",
    home_pose: str = "hands_up",
    tuck_on_abort: bool = True,
    viz: bool = True,
) -> bool:
    """Command *arm* to take the map-frame *candidate*: open -> pre-grasp -> grasp -> close.

    Drives the arm to the pre-grasp then grasp pose (both map-frame, via the
    ``*_arm_lift`` MoveIt group so the lift solves for reach), **checking each
    move's result string** and aborting on anything but ``"SUCCEEDED"``. Always
    re-enables gripper collision; tucks the arm home on abort. Returns True only
    when both moves succeeded and the gripper closed. Never raises.

    Executes whatever orientation the *candidate* carries — callers wanting a
    forward-pointing wrist (instead of GraspNet's) re-point it first with
    :func:`aim_forward_candidate`.
    """
    motion_group, home_group, gripper_group = _arm_sides(arm)

    side = motion_group.split("_")[0]
    hand = ctx.walkie.arm.left if side == "left" else ctx.walkie.arm.right

    grasp_effort_threshold = float(os.getenv("GRASP_EFFORT_THRESHOLD", "0.5").strip())

    try:
        roll, pitch, yaw = Rotation.from_matrix(candidate.rotation).as_euler("xyz")
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] execute_grasp[{side}]: bad rotation matrix ({exc})")
        return False
    grasp_xyz, pregrasp_xyz = candidate.grasp_xyz, candidate.pregrasp_xyz
    print(f"[grasp] execute_grasp[{side}]: RPY=({roll:+.2f},{pitch:+.2f},{yaw:+.2f}) "
          f"grasp={grasp_xyz} pregrasp={pregrasp_xyz}")
    if viz:
        _draw_grasp_viz(ctx, candidate)

    succeeded = False
    collision_disabled = False
    try:
        # original_planner = ctx.walkie.robot.arm.get_param(name="planner_id")  # warm up the planner cache
        # ctx.walkie.robot.arm.set_param_result(name="planner_id", value="RRTstar")
        hand.gripper(1.0, blocking=True)  # open, ready to receive

        ctx.walkie.arm.toggle_gripper_collision(gripper_group, False)
        collision_disabled = True

        res = ctx.walkie.arm.go_to_pose(
            *pregrasp_xyz, roll, pitch, yaw,
            group_name=home_group, frame_id="map", blocking=True,
        )
        print(f"[grasp] execute_grasp[{side}]: pregrasp -> {res}")
        if res != "SUCCEEDED":
            print(f"[grasp] execute_grasp[{side}]: pregrasp move failed; aborting")
            return False

        res = ctx.walkie.arm.go_to_pose(
            *grasp_xyz, roll, pitch, yaw,
            group_name=home_group, frame_id="map", blocking=True, cartesian_path=True
        )
        print(f"[grasp] execute_grasp[{side}]: grasp -> {res}")
        if res != "SUCCEEDED":
            print(f"[grasp] execute_grasp[{side}]: grasp move failed; aborting")
            return False

        hand.gripper(0.0, blocking=True)  # close on the object
        ctx.walkie.arm.go_to_home(group_name=motion_group, pose_name=home_pose, blocking=True)
        print(f"[grasp] execute_grasp[{side}]: success")
        _, _, effort = hand.get_gripper_states()
        print(f"[grasp]: current gripper effort {effort}")
        if effort < grasp_effort_threshold:
            # Grasp failed
            return False
        succeeded = True
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] execute_grasp[{side}]: hardware error ({exc})")
        return False
    finally:
        if tuck_on_abort and not succeeded:
            try:
                result = ctx.walkie.arm.go_to_home(group_name=home_group, pose_name=home_pose, blocking=True)
                print(f"[grasp] execute_grasp[{side}]: tuck-on-abort home -> {result}")
            except Exception as exc:  # noqa: BLE001
                print(f"[grasp] execute_grasp[{side}]: tuck-on-abort home failed ({exc})")
        # ctx.walkie.robot.arm.set_param_result(name="planner_id", value=original_planner)
        if collision_disabled:
            try:
                ctx.walkie.arm.toggle_gripper_collision(gripper_group, True)
            except Exception as exc:  # noqa: BLE001
                print(f"[grasp] execute_grasp[{side}]: re-enable gripper collision failed ({exc})")


def execute_grasp_with_retry(
    ctx: TaskContext,
    candidate: GraspCandidate,
    *,
    arm: str,
    max_reach_xy_m: float,
    attempts: int | None = None,
    strafe_m: float | None = None,
    back_m: float | None = None,
    viz: bool = True,
) -> bool:
    """Execute *candidate*, repositioning the base and retrying on arm-move failure.

    ``go_to_pose`` sometimes fails to plan/reach an otherwise-valid grasp (IK corner
    cases near the workspace edge). Because the base move is a pure translation
    (heading held), the map-frame *candidate* stays geometrically valid, and shifting
    the arm's relative geometry can turn a failed IK into a solvable one. So on each
    failure we strafe the base sideways (away from the centreline on the chosen arm's
    side) and step slightly back, then **re-execute the same candidate** — up to
    *attempts* times, as long as the grasp stays within *max_reach_xy_m*.

    Returns True on the first clean grasp, else False after exhausting attempts (or
    once the target falls out of reach). Never raises.
    """
    side = (arm or "left").strip().lower()
    if side not in ("left", "right"):
        side = "left"
    attempts = _i("WALKIE_GRASP_RETRY_ATTEMPTS", "3") if attempts is None else attempts
    strafe_m = _f("WALKIE_GRASP_RETRY_STRAFE_M", "0.08") if strafe_m is None else strafe_m
    back_m = _f("WALKIE_GRASP_RETRY_BACK_M", "0.05") if back_m is None else back_m

    for i in range(max(1, attempts)):
        reach = _xy_dist(ctx, candidate.grasp_xyz)
        if reach > max_reach_xy_m:
            print(f"[grasp] retry {i + 1}/{attempts}: out of reach "
                  f"(xy={reach:.2f}m > {max_reach_xy_m:.2f}m); stopping")
            return False
        print(f"[grasp] grasp attempt {i + 1}/{attempts} (reach {reach:.2f}m)")
        if execute_grasp(ctx, candidate, arm=side, viz=viz):
            return True
        if i + 1 >= attempts:
            break
        # Reposition: strafe away from the centreline on the arm's side (+y = base
        # left, so left arm strafes +y / right arm -y) and step slightly back.
        dy = strafe_m if side == "left" else -strafe_m
        print(f"[grasp] retry: repositioning base (back {back_m:.2f}m, strafe {dy:+.2f}m) "
              f"then re-executing the same candidate")
        # Direct cmd_vel creep, not nav.go_to: these cm-scale repositions next to the
        # table are exactly where Nav2 nudges backwards / refuses to strafe.
        creep_base_relative(ctx, -back_m, dy)
    print(f"[grasp] grasp failed after {attempts} attempt(s)")
    return False


def pick_object(
    ctx: TaskContext,
    prompts: list[str],
    *,
    arm: str = "auto",
    attempts: int = 10,
    pregrasp_standoff_m: float = 0.10,
    approach_preference: str = "none",
    approach_weight: float | None = None,
    optimal_standoff_m: float = 0.55,
    approach_trigger_m: float = 0.60,
    max_reach_xy_m: float = 0.70,
    min_grasp_z_m: float = 0.70,
    deadzone_half_m: float = 0.20,
    default_arm: str = "left",
    point_at_object: bool = True,
    track: bool = True,
    viz: bool = True,
) -> bool:
    """Full pick for the nearest object matching *prompts*: locate -> approach -> de-deadzone -> grasp.

    Positioning runs on a cheap detect+lift (:func:`locate_object`, NO GraspNet); the
    expensive grasp plan runs exactly ONCE, after the robot has approached and is in
    its final grasp pose. This is the main speedup — the old flow planned a full grasp
    far away and then again up close, paying for GraspNet twice and throwing the first
    plan away. Each locate first faces the object (:func:`face_object` +
    :func:`look_at_object`) to centre it in view for a more accurate detection.

    1. Cheap locate; bail if the object is below *min_grasp_z_m* (no remedy). Raise the
       lift to the estimated grasp height for better reach.
    2. If it's farther than *approach_trigger_m* (XY), drive to *optimal_standoff_m*
       facing it (head tracking), then re-locate from the new viewpoint.
    3. Pick the arm (``"auto"`` -> object's side, dead-centre -> *default_arm*).
    4. Servo the base sideways (closed-loop cmd_vel strafe, heading held) until the
       chosen arm is laterally lined up with the object (:func:`align_arm_to_object`);
       ``WALKIE_GRASP_ARM_ALIGN=rotate`` falls back to turning the base instead
       (:func:`face_object_with_arm`), recording the pre-rotate heading.
    5. Run the ONE heavy grasp plan from the final pose: by default
       (``WALKIE_GRASP_FUSE_SNAPS`` > 1) take that many snapshots at the **current** head
       angle, dedup + fuse them into one lower-noise cloud, strip residue/background-bleed
       noise (:func:`_clean_object_cloud`), and run a single GraspNet inference
       (:func:`get_object_grasp_pos`).
    6. With *point_at_object* (default), re-point the gripper straight forward along
       the robot heading (:func:`aim_forward_candidate`) instead of GraspNet's
       wrist orientation (often IK-unsolvable on OpenArm).
    7. Execute the grasp with base-reposition retries (:func:`execute_grasp_with_retry`).
    8. On success, tuck both arms into the travel pose (``WALKIE_CARRY_POSE``, default
       ``"standby"``) so the base can navigate on — the grasping arm is left extended
       over the table otherwise and Nav2 won't plan around it. Stow in place (no base
       retreat).
    9. Restore the pre-align heading (:func:`TaskContext.rotate_to`) so the base
       ends the pick in its original orientation, whether or not the grasp succeeded
       (a no-op for the default strafe path, which never changes heading).

    Returns True only when the grasp executed cleanly. Degrades to False (never
    raises) at any failing step.

    Note: distinct from :func:`tasks.manipulation.pick_object` (which takes a
    pre-detected ``DetectedObject``) — import this one from ``tasks.skills``.
    """
    last_xyz: Vec3 | None = None

    print(f"[grasp] pick_object: prompts={prompts} arm={arm} attempts={attempts} ")

    def _locate() -> ObjectLocation | None:
        # Face + tilt toward the last known position to centre the object in view
        # (a square-on view detects/lifts more reliably), then detect+lift. Cheap —
        # NO GraspNet. The first call has no estimate yet, so it uses the current view.
        nonlocal last_xyz
        if last_xyz is not None:
            face_object(ctx, last_xyz)
            look_at_object(ctx, last_xyz)
            time.sleep(0.5)  # let the base/head settle before the snapshot
        loc = locate_object(ctx, prompts)
        if loc is not None:
            last_xyz = loc.xyz_map
        return loc

    home_res = ctx.walkie.arm.go_to_home(group_name="both_arms_lift", pose_name="standby", blocking=True)
    if home_res != "SUCCEEDED":  # staging move: warn but press on
        print(f"[grasp] stage home -> {home_res} (continuing)")

    # Aim the head DOWN before the first locate: it has no position estimate yet, so it
    # snapshots at whatever angle the head arrived at (face_object/look_at_object only kick
    # in on later calls, once last_xyz is set). A table-top object sits below a level gaze,
    # so without this the first detection often misses it -> "no detection". Knob-gated so
    # the angle is tunable per surface height; "0" disables the forced tilt.
    locate_tilt = _f("WALKIE_GRASP_LOCATE_TILT", "0.4")
    if locate_tilt > 0:
        print(f"[grasp] pick_object: tilting head down {locate_tilt:.2f}rad before first locate")
        tilt_head(ctx, locate_tilt, settle=_f("WALKIE_GRASP_TILT_SETTLE_SEC", "0.4"))

    # 1. Cheap locate just to position the base/head (no GraspNet yet).
    loc = _locate()
    if loc is None:
        print(f"[grasp] pick_object: no detection for {prompts}")
        return False
    if loc.xyz_map[2] < min_grasp_z_m:
        print(f"[grasp] pick_object: object too low (z={loc.xyz_map[2]:.2f}m < "
              f"{min_grasp_z_m:.2f}m); cannot reach")
        return False

    base_lift_diff_m = ctx.walkie.robot.transform.lookup("base_footprint", "lift_link")["position"]["z"] - ctx.walkie.robot.lift.get(norm_pos=False) / 100.0
    optimum_lift_height = loc.xyz_map[2] + 0.18
    print(f"[grasp] pick_object: setting lift to {((optimum_lift_height - base_lift_diff_m) * 100):.2f}m for better reach")
    ctx.walkie.robot.lift.set(pos=(optimum_lift_height - base_lift_diff_m) * 100, norm_pos=False)

    # 2. Approach to the optimal standoff if too far, tracking with the head.
    status = approach_object(
        ctx, loc.xyz_map, standoff_m=optimal_standoff_m,
        trigger_m=approach_trigger_m, track=track,
    )
    if status == "FAILED":
        print("[grasp] pick_object: approach failed; aborting")
        return False

    loc = _locate()  # re-locate from the new viewpoint
    if loc is None:
        print("[grasp] pick_object: lost the object after approaching")
        return False
    if loc.xyz_map[2] < min_grasp_z_m:
        print(f"[grasp] pick_object: object too low after approach "
                f"(z={loc.xyz_map[2]:.2f}m); cannot reach")
        return False

    # 3. Pick the arm by which side the object is on (dead-centre -> default).
    in_zone = in_arm_deadzone(ctx, loc.xyz_map, half_width_m=deadzone_half_m)
    if arm == "auto":
        left = _world_to_base(ctx, loc.xyz_map)[1]
        if in_zone or left == 0:
            chosen = default_arm
        else:
            chosen = "left" if left > 0 else "right"
    else:
        chosen = (arm or default_arm).strip().lower()
        if chosen not in ("left", "right"):
            print(f"[grasp] pick_object: bad arm {arm!r}; using {default_arm}")
            chosen = default_arm
    print(f"[grasp] pick_object: chosen arm = {chosen}")

    # 4. Line the chosen arm up with the object so it's out of the centreline
    #    dead-zone. Two strategies (WALKIE_GRASP_ARM_ALIGN):
    #      "strafe" (default): closed-loop cmd_vel servo that slides the omni base
    #        sideways with the heading HELD (align_arm_to_object), recomputing the
    #        lateral error from live odom every tick. The nav stack scales cmd_vel
    #        near obstacles, so a blocked strafe just stalls out and the pick
    #        continues from there. Fixes the over-rotation seen when the object is
    #        close (the shoulder->object bearing is ill-conditioned, so "rotate"
    #        can over-shoot and face the wall). Gains robot-unverified.
    #      "rotate" (fallback escape hatch — the usable PR #31 behaviour): turn the
    #        base so the arm shoulder aims at the object (face_object_with_arm).
    #    We record the heading and restore it afterwards (no-op for the strafe path,
    #    which never changes it) so the base ends the pick where it started.
    original_heading = ctx.current_pose()["heading"]
    if os.getenv("WALKIE_GRASP_ARM_ALIGN", "strafe").strip().lower() == "rotate":
        face_object_with_arm(ctx, loc.xyz_map, arm=chosen)
    else:
        align_arm_to_object(ctx, loc.xyz_map, arm=chosen)

    try:
        # 5. The ONE heavy grasp plan, now that we're in the final pose: 2 snapshots at
        #    the configured head tilts, deduped + fused into one dense cloud, GraspNet once.
        cand = get_object_grasp_pos(
            ctx, prompts, attempts=attempts, standoff_m=pregrasp_standoff_m,
            approach_preference=approach_preference, approach_weight=approach_weight,
            # antipodal=False
        )
        if cand is None:
            print("[grasp] pick_object: grasp planning found nothing from the final pose")
            return False
        if cand.grasp_xyz[2] < min_grasp_z_m:
            print(f"[grasp] pick_object: planned grasp too low "
                  f"(z={cand.grasp_xyz[2]:.2f}m); cannot reach")
            return False

        # 6. Re-point the wrist straight forward at the object (instead of GraspNet's
        #    often-IK-unsolvable orientation), taking the arm's forward to be the robot's.
        #    The new candidate drives both the arm and the held-object record (so the placer
        #    reuses the real grasp pose). Keeps GraspNet's orientation when disabled.
        if approach_preference == "side" and point_at_object:
            cand = aim_forward_candidate(ctx, cand, standoff_m=pregrasp_standoff_m)

        # 7. Execute, repositioning the base and retrying on arm-move failure.
        ok = execute_grasp_with_retry(
            ctx, cand, arm=chosen, max_reach_xy_m=max_reach_xy_m, viz=viz,
        )
        if ok:
            # Remember what we're holding (per arm) so tasks.skills.place can put it back
            # down at the same height above whatever surface it's placed on.
            from tasks.skills.held import record_held_object

            # Record the base heading the grasp was taken at, alongside the map-frame
            # rotation: the arm is base-mounted and the base re-orients before placing,
            # so place reproduces the wrist orientation *relative to the base* (rotating
            # the stored rotation by the heading delta), not its absolute map orientation.
            # execute_grasp_with_retry only translates the base (heading held), so the
            # heading here still matches the pose cand.rotation was planned at.
            record_held_object(
                ctx,
                label=prompts[0] if prompts else "object",
                arm=chosen,
                grasp_xyz=cand.grasp_xyz,
                rotation=cand.rotation,
                width=cand.width,
                grasp_heading=ctx.current_pose()["heading"],
                footprint_m=cand.object_footprint_m,
                support_surface_z=cand.support_surface_z,
                grasp_to_surface_offset=cand.grasp_to_surface_offset,
            )

            # Tuck both arms into the travel pose so the base can navigate on. execute_grasp
            # leaves the grasping arm extended ("hands_up") to lift the object clear, which
            # Nav2's costmap reads as an obstacle parked on the robot, so it refuses to plan
            # away ("picked it up but won't walk on" — the one thing the PR #31 build lacked).
            # Stow IN PLACE only: no base retreat (the blind 0.5m cmd_vel reverse added in
            # d6ba8c1 "broke" drove the base off / parallel to the wall). WALKIE_CARRY_POSE
            # is the tuck preset (default "standby").
            stow_pose = os.getenv("WALKIE_CARRY_POSE", "standby").strip() or "standby"
            stow_res = ctx.walkie.arm.go_to_home(
                group_name="both_arms_lift", pose_name=stow_pose, blocking=True,
            )
            if stow_res != "SUCCEEDED":
                print(f"[grasp] pick_object: stow ({stow_pose}) -> {stow_res} (continuing)")
        return ok
    finally:
        # 8. Rotate back to the heading we had before aligning (the arm is home/tucked
        #    by now), so the base ends the pick in its original orientation — a no-op
        #    for the default strafe path, which never changes heading.
        print(f"[grasp] pick_object: restoring heading to {math.degrees(original_heading):+.0f}deg")
        ctx.rotate_to(original_heading)
