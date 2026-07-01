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
from walkie_world.people.store import _mean_unit
from tasks.HRI.identity import (
    _avg_unit,
    _dedup_person_id,
    _gate_candidates,
    _reject_outliers,
    audit_identity_collisions,
    enroll_guest_frames,
    locate_people,
    make_follow_selector,
    select_person_to_follow,
)
from tasks.skills import cxcywh_to_xyxy


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
    # These gating/throttle tests predate the lock hysteresis — lock on the first
    # good tick so their per-tick expectations are unchanged. The hysteresis tests
    # below set HRI_FOLLOW_LOCK_CONFIRM_TICKS explicitly.
    monkeypatch.setenv("HRI_FOLLOW_LOCK_CONFIRM_TICKS", "1")


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


# ---------------------------------------------------------------------------
# Follow F1: visible-candidate peakiness (return None on a near-tie)
# ---------------------------------------------------------------------------


def test_follow_returns_none_when_two_candidates_near_tie(tmp_path, monkeypatch):
    _follow_env(monkeypatch)
    monkeypatch.setenv("HRI_FOLLOW_VISIBLE_MARGIN", "0.06")
    face = _FaceRecCounting()  # no faces -> attire path
    # Two visible people whose attire is near-tied against the host's _vec(0,1,0).
    app = _AppearanceByWidth({100: _vec(0, 1, 0.05), 120: _vec(0, 1, 0.08)})
    b1, b2 = (320, 240, 100, 400), (520, 240, 120, 400)
    ctx = _follow_ctx(tmp_path, [[_pp(b1), _pp(b2)]], face, app)
    snap = _Snap(_img())
    # ambiguous: neither clearly the host -> coast (None)
    assert select_person_to_follow(ctx, snap, "host", run_face=False) is None
    # margin off -> the top scorer is taken (old behaviour)
    monkeypatch.setenv("HRI_FOLLOW_VISIBLE_MARGIN", "0")
    assert select_person_to_follow(ctx, snap, "host", run_face=False) == cxcywh_to_xyxy(b1)


def test_follow_picks_host_when_clear_winner(tmp_path, monkeypatch):
    _follow_env(monkeypatch)
    monkeypatch.setenv("HRI_FOLLOW_VISIBLE_MARGIN", "0.06")
    face = _FaceRecCounting()
    app = _AppearanceByWidth({100: _vec(0, 1, 0), 120: _vec(0, 1, 0.5)})  # b1 clearly host
    b1, b2 = (320, 240, 100, 400), (520, 240, 120, 400)
    ctx = _follow_ctx(tmp_path, [[_pp(b1), _pp(b2)]], face, app)
    box = select_person_to_follow(ctx, _Snap(_img()), "host", run_face=False)
    assert box == cxcywh_to_xyxy(b1)


# ---------------------------------------------------------------------------
# Follow F2: exclude a candidate whose box shows a clearly non-host face
# ---------------------------------------------------------------------------


def test_follow_excludes_box_showing_nonhost_face(tmp_path, monkeypatch):
    _follow_env(monkeypatch)
    monkeypatch.setenv("HRI_FOLLOW_VISIBLE_MARGIN", "0")  # isolate F2 from the peakiness gate
    host_box = (320, 240, 100, 400)  # width 100
    decoy_box = (520, 240, 120, 400)  # width 120 — a look-alike (even closer attire)
    app = _AppearanceByWidth({100: _vec(0, 1, 0.2), 120: _vec(0, 1, 0)})
    # One detected face, inside the decoy box, that is NOT the host's.
    nonhost = _fe(_vec(0, 0, 1), bbox=(500, 180, 540, 300), det=0.9)  # center (520, 240)
    ctx = _follow_ctx(tmp_path, [[_pp(host_box), _pp(decoy_box)]], _FaceRec([[nonhost]]), app)
    # F2 on: the decoy is ruled out by its non-host face, so the host wins.
    assert (
        select_person_to_follow(ctx, _Snap(_img()), "host", run_face=True)
        == cxcywh_to_xyxy(host_box)
    )
    # F2 off: nothing rules the decoy out and its attire out-scores the host.
    monkeypatch.setenv("HRI_FOLLOW_EXCLUDE_NONHOST_FACE", "0")
    ctx2 = _follow_ctx(
        tmp_path, [[_pp(host_box), _pp(decoy_box)]], _FaceRec([[nonhost]]), app
    )
    assert (
        select_person_to_follow(ctx2, _Snap(_img()), "host", run_face=True)
        == cxcywh_to_xyxy(decoy_box)
    )


# ---------------------------------------------------------------------------
# Follow F3: stricter re-acquisition floor
# ---------------------------------------------------------------------------


def test_reacquire_uses_higher_floor(tmp_path, monkeypatch):
    _follow_env(monkeypatch)  # HRI_FOLLOW_REACQUIRE_MIN_SCORE default 0.6
    face = _FaceRecCounting()
    v55 = _vec(0, 0.55, 0.835)  # ~0.55 cosine to the host attire _vec(0,1,0)
    app = _AppearanceByWidth({100: v55})
    box = (320, 240, 100, 400)
    ctx = _follow_ctx(tmp_path, [[_pp(box)]], face, app)
    snap = _Snap(_img())
    # re-acquiring: 0.55 < 0.6 -> no fresh lock granted
    assert select_person_to_follow(ctx, snap, "host", run_face=False, reacquiring=True) is None
    # maintaining: 0.55 >= steady-state 0.5 -> accepted
    assert (
        select_person_to_follow(ctx, snap, "host", run_face=False, reacquiring=False)
        == cxcywh_to_xyxy(box)
    )


# ---------------------------------------------------------------------------
# Follow F4: temporal hysteresis (lock confirmation + miss tolerance)
# ---------------------------------------------------------------------------


def test_lock_requires_k_confirming_ticks(tmp_path, monkeypatch):
    _follow_env(monkeypatch)
    monkeypatch.setenv("HRI_FOLLOW_LOCK_CONFIRM_TICKS", "2")
    app = _AppearanceByWidth({100: _vec(0, 1, 0)})
    box = (320, 240, 100, 400)
    ctx = _follow_ctx(tmp_path, [[_pp(box)]], _FaceRecCounting(), app)
    sel = make_follow_selector("host")
    snap = _Snap(_img())
    assert sel(ctx, snap) is None  # tick 0: confirm=1 -> coast
    assert sel(ctx, snap) == cxcywh_to_xyxy(box)  # tick 1: confirm=2 -> lock
    assert sel.locked is True


def test_lock_tolerates_brief_miss(tmp_path, monkeypatch):
    _follow_env(monkeypatch)
    monkeypatch.setenv("HRI_FOLLOW_LOCK_CONFIRM_TICKS", "1")
    monkeypatch.setenv("HRI_FOLLOW_LOCK_MISS_TOLERANCE", "1")
    app = _AppearanceByWidth({100: _vec(0, 1, 0)})
    box = (320, 240, 100, 400)
    # present, present, MISS, present, MISS, MISS
    ctx = _follow_ctx(
        tmp_path,
        [[_pp(box)], [_pp(box)], [], [_pp(box)], [], []],
        _FaceRecCounting(),
        app,
    )
    sel = make_follow_selector("host")
    snap = _Snap(_img())
    assert sel(ctx, snap) == cxcywh_to_xyxy(box)  # tick 0: K=1 -> lock now
    assert sel(ctx, snap) == cxcywh_to_xyxy(box)  # tick 1: held
    assert sel(ctx, snap) is None and sel.locked is True  # tick 2: miss=1, tolerated
    assert sel(ctx, snap) == cxcywh_to_xyxy(box)  # tick 3: re-seen, no re-confirm needed
    assert sel(ctx, snap) is None and sel.locked is True  # tick 4: miss=1
    assert sel(ctx, snap) is None and sel.locked is False  # tick 5: miss=2 > tol -> dropped


def test_face_runs_every_tick_until_locked(tmp_path, monkeypatch):
    _follow_env(monkeypatch, face_every_n="5")
    monkeypatch.setenv("HRI_FOLLOW_LOCK_CONFIRM_TICKS", "3")
    face = _FaceRecCounting()
    app = _AppearanceByWidth({100: _vec(0, 1, 0)})
    box = (320, 240, 100, 400)
    ctx = _follow_ctx(tmp_path, [[_pp(box)]], face, app)
    sel = make_follow_selector("host")
    snap = _Snap(_img())
    for _ in range(3):  # confirming ticks run face every tick despite every_n=5
        c0 = face.calls
        sel(ctx, snap)
        assert face.calls - c0 == 1
    assert sel.locked is True
    c0 = face.calls
    sel(ctx, snap)  # now locked, tick index 3 -> 3 % 5 != 0 -> face throttled off
    assert face.calls - c0 == 0


# ---------------------------------------------------------------------------
# Introduction B1/B2/B3/G1: optimal assignment, peakiness, votes, quality gate
# ---------------------------------------------------------------------------


def _locate_env(monkeypatch):
    monkeypatch.setenv("HRI_FACE_MIN_AREA_PX", "0")  # don't gate on synthetic face size
    monkeypatch.setenv("HRI_RECOG_MIN_DET_SCORE", "0")  # trust the synthetic faces
    monkeypatch.setenv("HRI_LOCATE_BOX_MARGIN", "0.06")
    monkeypatch.setenv("HRI_LOCATE_MIN_VOTES", "1")


def _locate_ctx(tmp_path, faces_per_frame, poses_per_frame, app_items, enroll, sub="p"):
    store = PeopleStore(persist_dir=tmp_path / sub, embedding_model="m")
    enroll(store)
    ai = _AI(
        face_recognition=_FaceRec(faces_per_frame),
        pose_estimation=_Pose(poses_per_frame),
        appearance=_Appearance(app_items),
    )
    return _Ctx(store, ai)


def test_locate_greedy_steal_fixed(tmp_path, monkeypatch):
    _locate_env(monkeypatch)
    g1, g2 = _vec(1, 0, 0), _vec(0, 1, 0)

    def enroll(store):  # face-only (no attire) so the score is purely the face cosine
        store.enroll("", "", g1, person_id="guest-1")
        store.enroll("", "", g2, person_id="guest-2")

    box_a, box_b = (200, 240, 100, 400), (450, 240, 100, 400)
    fa = _vec(0.52, 0.55, 0.6535)  # cos g1=0.52, g2=0.55 (greedy would give box_a to g2)
    fb = _vec(0.10, 0.90, 0.4243)  # cos g1=0.10, g2=0.90
    face_a = _fe(fa, bbox=(180, 150, 260, 280))  # center (220,215) inside box_a
    face_b = _fe(fb, bbox=(420, 150, 500, 280))  # center (460,215) inside box_b
    ctx = _locate_ctx(
        tmp_path,
        [[face_a, face_b]],
        [[_pp(box_a), _pp(box_b)]],
        [_vec(0, 0, 1)],
        enroll,
    )
    located = locate_people(ctx, [_img()], ["guest-1", "guest-2"])
    # Optimal assignment keeps guest-1 on box_a (its true box) even though guest-2
    # scores higher there, because guest-2 is best explained by box_b.
    assert located["guest-1"][1] == cxcywh_to_xyxy(box_a)
    assert located["guest-2"][1] == cxcywh_to_xyxy(box_b)


def test_locate_drops_ambiguous_box(tmp_path, monkeypatch):
    _locate_env(monkeypatch)
    g1, g2 = _vec(1, 0, 0), _vec(0, 1, 0)

    def enroll(store):
        store.enroll("", "", g1, person_id="guest-1")
        store.enroll("", "", g2, person_id="guest-2")

    box = (320, 240, 100, 400)
    fx = _vec(0.55, 0.54, 0.637)  # top-2 near-tied (0.55 vs 0.54)
    face = _fe(fx, bbox=(290, 150, 370, 280))  # center (330,215) inside box
    ctx = _locate_ctx(tmp_path, [[face]], [[_pp(box)]], [_vec(0, 0, 1)], enroll)
    # ambiguous -> not labeled (caller falls back to the stored seat)
    assert "guest-1" not in locate_people(ctx, [_img()], ["guest-1", "guest-2"])
    monkeypatch.setenv("HRI_LOCATE_BOX_MARGIN", "0")
    assert "guest-1" in locate_people(ctx, [_img()], ["guest-1", "guest-2"])


def test_locate_multiframe_vote(tmp_path, monkeypatch):
    _locate_env(monkeypatch)

    def enroll(store):
        store.enroll("", "", _vec(1, 0, 0), person_id="guest-1")

    box = (320, 240, 100, 400)
    face = _fe(_vec(1, 0, 0.05), bbox=(290, 150, 370, 280))  # strong guest-1 in frame 0 only
    frames = [_img(), _img(), _img()]
    ctx = _locate_ctx(
        tmp_path, [[face], [], []], [[_pp(box)], [], []], [_vec(0, 0, 1)], enroll, sub="p1"
    )
    assert "guest-1" in locate_people(ctx, frames, ["guest-1"])  # 1 vote suffices
    monkeypatch.setenv("HRI_LOCATE_MIN_VOTES", "2")
    ctx2 = _locate_ctx(
        tmp_path, [[face], [], []], [[_pp(box)], [], []], [_vec(0, 0, 1)], enroll, sub="p2"
    )
    assert "guest-1" not in locate_people(ctx2, frames, ["guest-1"])  # only 1 frame agrees


def test_locate_appearance_fallback_still_peaky(tmp_path, monkeypatch):
    _locate_env(monkeypatch)

    def enroll(store):
        store.enroll("", "", _vec(1, 0, 0), person_id="guest-1", app_embedding=_vec(0, 1, 0))
        store.enroll("", "", _vec(0, 1, 0), person_id="guest-2", app_embedding=_vec(0, 0, 1))

    box = (320, 240, 100, 400)
    av = _vec(0, 0.55, 0.54)  # attire near-tied between the two guests
    ctx = _locate_ctx(tmp_path, [[]], [[_pp(box)]], [av], enroll, sub="p1")  # faces turned away
    assert "guest-1" not in locate_people(ctx, [_img()], ["guest-1", "guest-2"])
    monkeypatch.setenv("HRI_LOCATE_BOX_MARGIN", "0")
    ctx2 = _locate_ctx(tmp_path, [[]], [[_pp(box)]], [av], enroll, sub="p2")
    assert "guest-1" in locate_people(ctx2, [_img()], ["guest-1", "guest-2"])


def test_locate_skips_low_det_face(tmp_path, monkeypatch):
    _locate_env(monkeypatch)
    monkeypatch.setenv("HRI_RECOG_MIN_DET_SCORE", "0.5")  # re-enable the det gate

    def enroll(store):
        store.enroll("", "", _vec(1, 0, 0), person_id="guest-1")

    box = (320, 240, 100, 400)
    fx = _vec(1, 0, 0.05)  # a strong guest-1 face
    low = _fe(fx, bbox=(290, 150, 370, 280), det=0.3)  # but below the det gate
    ctx = _locate_ctx(tmp_path, [[low]], [[_pp(box)]], [_vec(0, 0, 1)], enroll, sub="p1")
    assert "guest-1" not in locate_people(ctx, [_img()], ["guest-1"])
    good = _fe(fx, bbox=(290, 150, 370, 280), det=0.9)
    ctx2 = _locate_ctx(tmp_path, [[good]], [[_pp(box)]], [_vec(0, 0, 1)], enroll, sub="p2")
    assert "guest-1" in locate_people(ctx2, [_img()], ["guest-1"])


def test_follow_maintenance_not_area_gated(tmp_path, monkeypatch):
    _follow_env(monkeypatch)
    monkeypatch.setenv("HRI_FACE_MIN_AREA_PX", "10000")
    box = (320, 240, 100, 400)
    # a SMALL host face (area 60*80 = 4800 < 10000) that matches the host
    small = _fe(_vec(1, 0, 0), bbox=(300, 160, 360, 240), det=0.9)  # center (330,200) inside box
    app = _AppearanceByWidth({100: _vec(0, 0, 1)})  # attire deliberately non-matching
    ctx = _follow_ctx(tmp_path, [[_pp(box)]], _FaceRec([[small]]), app)
    # maintenance (reacquiring=False -> hard=False): the small face still holds the lock
    assert (
        select_person_to_follow(ctx, _Snap(_img()), "host", run_face=True, reacquiring=False)
        == cxcywh_to_xyxy(box)
    )


# ---------------------------------------------------------------------------
# G2: pre-enrollment dedup helper
# ---------------------------------------------------------------------------


def _dedup_store(tmp_path):
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    store.enroll("", "", _vec(1, 0, 0), person_id="guest-1", app_embedding=_vec(0, 1, 0))
    return store


def test_dedup_reuses_id_when_unpinned_strong_match(tmp_path):
    ctx = _Ctx(_dedup_store(tmp_path))
    rid = _dedup_person_id(ctx, _vec(1, 0, 0.05), _vec(0, 1, 0.05), 0.9, "new", pinned=False)
    assert rid == "guest-1"


def test_dedup_warns_keeps_pinned_id_on_collision(tmp_path, capsys):
    ctx = _Ctx(_dedup_store(tmp_path))
    rid = _dedup_person_id(ctx, _vec(1, 0, 0.05), _vec(0, 1, 0.05), 0.9, "guest-2", pinned=True)
    assert rid == "guest-2"  # distinct pinned guests are never auto-merged
    assert "WARNING" in capsys.readouterr().out


def test_dedup_creates_new_on_weak_match(tmp_path):
    ctx = _Ctx(_dedup_store(tmp_path))
    rid = _dedup_person_id(ctx, _vec(0, 0, 1), _vec(0, 0, 1), 0.9, "guest-2", pinned=False)
    assert rid == "guest-2"


def test_dedup_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HRI_ENROLL_DEDUP", "0")
    ctx = _Ctx(_dedup_store(tmp_path))
    rid = _dedup_person_id(ctx, _vec(1, 0, 0.05), _vec(0, 1, 0.05), 0.9, "new", pinned=False)
    assert rid == "new"
