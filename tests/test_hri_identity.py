"""Unit tests for the HRI multi-frame enrollment + dedup helpers.

No server, no robot: synthetic unit vectors and tiny fake AI clients exercise
the burst-averaging, within-guest outlier rejection, multi-frame enrollment, and
cross-guest near-duplicate audit added for the posed-capture flow.
"""

import math
from types import SimpleNamespace

import pytest
from PIL import Image

from client import FaceEmbedding, PersonPose
from perception import PeopleStore
from perception.people_store import _mean_unit
from tasks.HRI.identity import (
    _avg_unit,
    _gate_candidates,
    _reject_outliers,
    audit_identity_collisions,
    enroll_guest_frames,
    make_follow_selector,
)
from tasks.HRI.skills import cxcywh_to_xyxy


def _u(*v):
    """A small unit vector (for the pure-function tests)."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _vec(*lead):
    """A 512-d unit vector with the given leading components (rest zero).

    512-d keeps the face collection's dimensionality consistent with
    enroll_guest_frames' ``[0.0] * 512`` zero-face fallback.
    """
    v = [0.0] * 512
    for i, x in enumerate(lead):
        v[i] = x
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _img():
    return Image.new("RGB", (640, 480), (10, 20, 30))


def _fe(emb, bbox=(280, 180, 360, 260), det=0.9):
    return FaceEmbedding(bbox_xyxy=bbox, embedding=emb, det_score=det)


def _pp(bbox=(320, 240, 200, 400)):
    return PersonPose(bbox=bbox, confidence=0.9, keypoints=[])


class _Seq:
    """Returns queued items on each call; repeats the last once exhausted."""

    def __init__(self, items):
        self._items, self._i = list(items), 0

    def next(self):
        item = self._items[min(self._i, len(self._items) - 1)]
        self._i += 1
        return item


class _FaceRec:
    def __init__(self, per_call):
        self._seq = _Seq(per_call)

    def embed(self, img):
        return self._seq.next()


class _Pose:
    def __init__(self, per_call):
        self._seq = _Seq(per_call)

    def estimate(self, img):
        return self._seq.next()


class _Appearance:
    def __init__(self, per_call):
        self._seq = _Seq(per_call)

    def embed(self, crop):
        return self._seq.next()


class _ImageFacade:
    """Unified image facade over the old per-task fakes, exposing the
    client.ImageClient method names that tasks.HRI.identity now calls."""

    def __init__(self, face=None, pose=None, appearance=None):
        self._face, self._pose, self._app = face, pose, appearance

    def faces(self, img):
        return self._face.embed(img)

    def estimate_poses(self, img):
        return self._pose.estimate(img)

    def appearance(self, crop):
        return self._app.embed(crop)

    def process(self, image, *, pose=False, face=False, **_kwargs):
        """Mirror ImageClient.process for the combined pose+face follow call:
        run only the requested tasks (so per-tick face/pose call counts stay
        exactly what the serial single-task path used to make) and return an
        ImageResult-shaped object with .pose / .face populated."""
        return SimpleNamespace(
            pose=(self.estimate_poses(image) if pose else None),
            face=(self.faces(image) if face else None),
        )


class _AI:
    def __init__(self, face_recognition=None, pose_estimation=None, appearance=None):
        self.image = _ImageFacade(face_recognition, pose_estimation, appearance)


class _Ctx:
    def __init__(self, people, walkieAI=None):
        self.people = people
        self.walkieAI = walkieAI


# ---------------------------------------------------------------------------
# _avg_unit
# ---------------------------------------------------------------------------


def test_avg_unit_empty_is_none():
    assert _avg_unit([]) is None


def test_avg_unit_single_is_renormalized():
    assert _avg_unit([_u(2.0, 0.0, 0.0)]) == pytest.approx([1.0, 0.0, 0.0])


def test_avg_unit_mean_is_unit_and_symmetric():
    out = _avg_unit([_u(1.0, 0.0, 0.0), _u(0.0, 1.0, 0.0)])
    assert math.isclose(sum(x * x for x in out) ** 0.5, 1.0, abs_tol=1e-9)
    assert out[0] == pytest.approx(out[1])  # symmetric inputs → equal components


# ---------------------------------------------------------------------------
# _reject_outliers
# ---------------------------------------------------------------------------


def test_reject_outliers_drops_the_odd_frame(monkeypatch):
    monkeypatch.setenv("HRI_BURST_OUTLIER_REJECT", "1")
    a1, a2 = _u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01)
    far = _u(0.0, 1.0, 0.0)  # falls below the absolute cosine-to-centroid floor
    kept = _reject_outliers([a1, a2, far])
    assert len(kept) == 2 and far not in kept


def test_reject_outliers_uniform_burst_keeps_all(monkeypatch):
    monkeypatch.setenv("HRI_BURST_OUTLIER_REJECT", "1")
    vs = [_u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01), _u(0.99, 0.03, 0.0)]
    assert _reject_outliers(vs) == vs


def test_reject_outliers_noop_below_three(monkeypatch):
    monkeypatch.setenv("HRI_BURST_OUTLIER_REJECT", "1")
    vs = [_u(1.0, 0.0, 0.0), _u(0.0, 1.0, 0.0)]  # 2 frames: can't tell, keep both
    assert _reject_outliers(vs) == vs


def test_reject_outliers_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("HRI_BURST_OUTLIER_REJECT", "0")
    vs = [_u(1.0, 0.0, 0.0), _u(1.0, 0.01, 0.0), _u(0.0, 1.0, 0.0)]
    assert _reject_outliers(vs) == vs


def test_reject_outliers_never_returns_empty(monkeypatch):
    monkeypatch.setenv("HRI_BURST_OUTLIER_REJECT", "1")
    # Even a pathological spread (near-degenerate centroid) keeps at least one —
    # a modality must never be stripped to nothing.
    vs = [_u(1.0, 0.0, 0.0), _u(-0.5, 0.866, 0.0), _u(-0.5, -0.866, 0.0)]
    assert len(_reject_outliers(vs)) >= 1


def test_reject_outliers_drops_single_outlier_of_four(monkeypatch):
    monkeypatch.setenv("HRI_BURST_OUTLIER_REJECT", "1")
    cluster = [_u(1.0, 0.02, 0.0), _u(1.0, 0.0, 0.01), _u(0.99, 0.03, 0.0)]
    far = _u(0.0, 1.0, 0.0)
    kept = _reject_outliers(cluster + [far])
    assert far not in kept and len(kept) == 3


# ---------------------------------------------------------------------------
# enroll_guest_frames
# ---------------------------------------------------------------------------


def test_enroll_guest_frames_averages_and_enrolls(tmp_path, monkeypatch):
    monkeypatch.setenv("HRI_BURST_OUTLIER_REJECT", "0")  # isolate averaging
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    f1, f2, f3 = _vec(1, 0.02, 0), _vec(1, 0, 0.01), _vec(0.99, 0.03, 0)
    a1, a2, a3 = _vec(0, 1, 0.10), _vec(0, 1, 0.05), _vec(0, 0.98, 0.12)
    ai = _AI(
        face_recognition=_FaceRec([[_fe(f1)], [_fe(f2)], [_fe(f3)]]),
        pose_estimation=_Pose([[_pp()], [_pp()], [_pp()]]),
        appearance=_Appearance([a1, a2, a3]),
    )
    imgs = [_img() for _ in range(3)]
    ok = enroll_guest_frames(_Ctx(store, ai), imgs, imgs, "guest-1", name="Alice", drink="cola")
    assert ok and store.count() == 1
    rec = store.get("guest-1")
    assert rec.name == "Alice" and rec.drink == "cola"
    assert list(rec.embedding) == pytest.approx(_mean_unit([f1, f2, f3]), abs=1e-6)
    # recognizable by a fresh near-shot, via face and via attire
    assert store.recognize(_vec(1, 0.01, 0)) is not None
    assert store.recognize_fused(None, _vec(0, 1, 0.08)) is not None


def test_enroll_guest_frames_attire_only_when_no_face(tmp_path, monkeypatch):
    monkeypatch.setenv("HRI_BURST_OUTLIER_REJECT", "0")
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    ai = _AI(
        face_recognition=_FaceRec([[]]),  # never any face
        pose_estimation=_Pose([[_pp()]]),
        appearance=_Appearance([_vec(0, 1, 0.10)]),
    )
    imgs = [_img()]
    ok = enroll_guest_frames(_Ctx(store, ai), imgs, imgs, "guest-2", name="Bob", drink="tea")
    assert ok and store.count() == 1
    assert store.recognize_fused(None, _vec(0, 1, 0.08)) is not None  # attire matches
    assert store.recognize(_vec(1, 0, 0)) is None  # zero face never face-matches


def test_enroll_guest_frames_attire_falls_back_to_face_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("HRI_BURST_OUTLIER_REJECT", "0")
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    ai = _AI(
        face_recognition=_FaceRec([[_fe(_vec(1, 0, 0))]]),
        pose_estimation=_Pose([[]]),  # no body box → fall back to the face-frame crop
        appearance=_Appearance([_vec(0, 1, 0)]),
    )
    imgs = [_img()]
    ok = enroll_guest_frames(_Ctx(store, ai), imgs, imgs, "guest-1", name="A")
    assert ok and store.count() == 1
    assert store.recognize(_vec(1, 0, 0)) is not None
    assert store.recognize_fused(None, _vec(0, 1, 0)) is not None


def test_enroll_guest_frames_false_when_nothing_usable(tmp_path):
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    ai = _AI(
        face_recognition=_FaceRec([[]]),
        pose_estimation=_Pose([[]]),
        appearance=_Appearance([_vec(0, 1, 0)]),
    )
    imgs = [_img()]
    assert enroll_guest_frames(_Ctx(store, ai), imgs, imgs, "guest-3") is False
    assert store.count() == 0


def test_enroll_guest_frames_no_store_is_false():
    ai = _AI()
    assert enroll_guest_frames(_Ctx(None, ai), [_img()], [_img()], "guest-1") is False


# ---------------------------------------------------------------------------
# audit_identity_collisions + appearance_vectors
# ---------------------------------------------------------------------------


def test_audit_flags_near_duplicate_faces(tmp_path):
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    store.enroll("", "", _vec(1, 0.02, 0), person_id="guest-1", app_embedding=_vec(1, 0, 0))
    store.enroll("", "", _vec(1, 0, 0.01), person_id="guest-2", app_embedding=_vec(0, 1, 0))
    cols = audit_identity_collisions(_Ctx(store))
    face = [c for c in cols if c[2] == "face"]
    assert len(face) == 1 and face[0][3] >= 0.75
    assert all(c[2] != "appearance" for c in cols)  # distinct outfits → no attire flag


def test_audit_no_collision_for_distinct_people(tmp_path):
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    store.enroll("", "", _vec(1, 0, 0), person_id="guest-1", app_embedding=_vec(1, 0, 0))
    store.enroll("", "", _vec(0, 1, 0), person_id="guest-2", app_embedding=_vec(0, 1, 0))
    assert audit_identity_collisions(_Ctx(store)) == []


def test_audit_empty_with_one_person(tmp_path):
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    store.enroll("", "", _vec(1, 0, 0), person_id="guest-1")
    assert audit_identity_collisions(_Ctx(store)) == []


def test_audit_no_store_is_empty():
    assert audit_identity_collisions(_Ctx(None)) == []


def test_appearance_vectors_returns_only_enrolled_attire(tmp_path):
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    store.enroll("", "", _vec(1, 0, 0), person_id="guest-1", app_embedding=_vec(0, 1, 0))
    store.enroll("", "", _vec(0, 1, 0), person_id="guest-2")  # no attire enrolled
    av = store.appearance_vectors()
    assert set(av) == {"guest-1"}
    assert av["guest-1"] == pytest.approx(_vec(0, 1, 0), abs=1e-6)


# ---------------------------------------------------------------------------
# _gate_candidates (pure)
# ---------------------------------------------------------------------------


def test_gate_candidates_keeps_only_near_hint():
    hint = (270, 40, 370, 440)  # width 100, center (320, 240) -> radius 150
    near = ((300, 40, 360, 440), 1.0)  # center 330 -> dist 10
    far = ((500, 40, 600, 440), 2.0)  # center 550 -> dist 230
    assert _gate_candidates([near, far], hint, radius_scale=1.5) == [near]


def test_gate_candidates_no_hint_returns_all():
    cands = [((0, 0, 10, 10), None), ((20, 20, 30, 30), None)]
    assert _gate_candidates(cands, None, radius_scale=1.5) == cands


def test_gate_candidates_disabled_returns_all():
    hint = (0, 0, 10, 10)
    far = ((100, 100, 120, 120), None)  # would be excluded if gating ran
    assert _gate_candidates([far], hint, radius_scale=0.0) == [far]


def test_gate_candidates_empty_when_none_near():
    hint = (0, 0, 10, 10)  # width 10, center (5, 5) -> radius 15
    far = ((100, 100, 120, 120), None)
    assert _gate_candidates([far], hint, radius_scale=1.5) == []


# ---------------------------------------------------------------------------
# FollowSelector: spatial gating + face throttle + widen-on-miss
# ---------------------------------------------------------------------------


class _Snap:
    def __init__(self, img):
        self.img = img


class _FaceRecCounting:
    """Detects no faces (host's back is turned), but counts the calls."""

    def __init__(self):
        self.calls = 0

    def embed(self, img):
        self.calls += 1
        return []


class _AppearanceByWidth:
    """Maps a crop to a vector by its width, so the returned attire is keyed on
    *which* candidate box was embedded (host vs. decoy) regardless of call order;
    counts the calls so a test can assert how many embeds a tick paid for."""

    def __init__(self, by_width):
        self.by_width = by_width
        self.calls = 0

    def embed(self, crop):
        self.calls += 1
        return self.by_width[crop.width]


def _follow_ctx(tmp_path, pose_per_call, face, appearance):
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    # Host enrolled with a face vector and an attire vector; the attire is what
    # the follow loop matches against (the face pass never matches from behind).
    store.enroll("", "", _vec(1, 0, 0), person_id="host", app_embedding=_vec(0, 1, 0))
    ai = _AI(face_recognition=face, pose_estimation=_Pose(pose_per_call), appearance=appearance)
    return _Ctx(store, ai)


def _follow_env(monkeypatch, *, face_every_n="3"):
    monkeypatch.setenv("HRI_FOLLOW_PARALLEL", "0")  # serial path: deterministic counts
    monkeypatch.setenv("HRI_FOLLOW_GATE_RADIUS_SCALE", "1.5")
    monkeypatch.setenv("HRI_FOLLOW_FACE_EVERY_N", face_every_n)
    monkeypatch.setenv("HRI_FOLLOW_APPEARANCE_MAX_CANDIDATES", "0")  # keep order; no cap
    monkeypatch.setenv("HRI_FOLLOW_APPEARANCE_MARGIN", "0.05")


# host box: cxcywh (320, 240, 100, 400) -> xyxy (270, 40, 370, 440), width 100,
# center (320, 240). decoy is 180px away (center 500), outside the 150px gate.
_HOST_A = (320, 240, 100, 400)
_DECOY_A = (500, 240, 200, 400)  # width 200 -> non-matching attire


def test_follow_selector_gates_and_throttles(tmp_path, monkeypatch):
    _follow_env(monkeypatch, face_every_n="3")
    face = _FaceRecCounting()
    app = _AppearanceByWidth({100: _vec(0, 1, 0), 200: _vec(0, 0, 1)})  # 100=host, 200=decoy
    ctx = _follow_ctx(tmp_path, [[_pp(_HOST_A), _pp(_DECOY_A)]], face, app)
    sel = make_follow_selector("host")
    snap = _Snap(_img())

    ticks = []
    for _ in range(6):
        f0, a0 = face.calls, app.calls
        box = sel(ctx, snap)
        ticks.append((box is not None, face.calls - f0, app.calls - a0))

    # tick 0 re-acquires (no lock): face ON, full scan embeds BOTH candidates.
    assert ticks[0] == (True, 1, 2)
    # steady ticks: face throttled OFF, attire gated to the host box only (1 embed).
    assert ticks[1] == (True, 0, 1)
    assert ticks[2] == (True, 0, 1)
    # every-3rd tick: face ON again (re-check), still gated to the host (1 embed).
    assert ticks[3] == (True, 1, 1)
    assert ticks[4] == (True, 0, 1)
    assert ticks[5] == (True, 0, 1)


def test_follow_selector_widens_when_gated_candidate_fails(tmp_path, monkeypatch):
    _follow_env(monkeypatch, face_every_n="3")
    face = _FaceRecCounting()
    app = _AppearanceByWidth({100: _vec(0, 1, 0), 200: _vec(0, 0, 1)})  # 100=host, 200=decoy
    # tick 0: only the host (at A) is in view -> locks onto A.
    # tick 1: a non-matching decoy (width 200) sits where the host was (near A),
    #         while the real host (width 100) has moved far (out of the gate).
    host_b = (560, 240, 100, 400)  # center 560: 240px from A, outside the 150px gate
    decoy_at_a = (320, 240, 200, 400)  # width 200 (non-match), squatting on the old spot
    ctx = _follow_ctx(
        tmp_path,
        [[_pp(_HOST_A)], [_pp(decoy_at_a), _pp(host_b)]],
        face,
        app,
    )
    sel = make_follow_selector("host")
    snap = _Snap(_img())

    assert sel(ctx, snap) == cxcywh_to_xyxy(_HOST_A)  # tick 0: locked onto A

    a0 = app.calls
    box = sel(ctx, snap)  # tick 1: gated decoy misses -> widen to full frame
    # Re-acquired the real host (at B), not the decoy squatting on the old spot.
    assert box == cxcywh_to_xyxy(host_b)
    # 1 gated embed (the decoy, a miss) + 2 full-set embeds (decoy + host) = 3.
    assert app.calls - a0 == 3
