"""Guest identity: enroll faces/appearance into ctx.people, find them again.

Enrollment happens at the door (the guest stands alone in front of the
camera — the best face shot of the whole run); recognition happens in the
living room at introduction time, where guests may have switched seats.

Everything here is best-effort: any AI-client failure logs and degrades to a
partial result — never raises (a missed enrollment costs points, a crashed
task costs the run).
"""

from __future__ import annotations

import math
import os
import time

from PIL import Image

from client.face_recognition import FaceEmbedding
from client.pose_estimation import PersonPose
from tasks.base import TaskContext

from .skills import BBox, cxcywh_to_xyxy, lift_bbox_world_xy


def _expand_face_to_person(face_xyxy: BBox, img_w: int, img_h: int) -> BBox:
    """Approximate a person bbox from a face bbox (torso-heavy crop).

    Used when pose estimation found no person for that face: one face width of
    margin per side, half a face height above, four below — enough body for
    the OSNet attire embedding without grabbing the neighbours.
    """
    x1, y1, x2, y2 = face_xyxy
    fw, fh = x2 - x1, y2 - y1
    return (
        max(0.0, x1 - fw),
        max(0.0, y1 - 0.5 * fh),
        min(float(img_w), x2 + fw),
        min(float(img_h), y2 + 4.0 * fh),
    )


def _person_bbox_for_face(
    face: FaceEmbedding, persons_xyxy: list[BBox], img_w: int, img_h: int
) -> BBox:
    """The pose-detected person containing this face, else an expanded face box."""
    fx = (face.bbox_xyxy[0] + face.bbox_xyxy[2]) / 2
    fy = (face.bbox_xyxy[1] + face.bbox_xyxy[3]) / 2
    containing = [
        b for b in persons_xyxy if b[0] <= fx <= b[2] and b[1] <= fy <= b[3]
    ]
    if containing:
        # Smallest containing box — a huge box swallowing several people loses
        # to the snug one around this person.
        return min(containing, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return _expand_face_to_person(face.bbox_xyxy, img_w, img_h)


def _appearance_embedding(ctx: TaskContext, img: Image.Image, person_xyxy: BBox):
    """OSNet attire embedding of the person crop; None on any failure."""
    try:
        x1, y1, x2, y2 = (int(v) for v in person_xyxy)
        crop = img.crop((max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)))
        return ctx.walkieAI.appearance.embed(crop)
    except Exception as exc:
        print(f"[identity] appearance embed failed ({exc})")
        return None


def _detect_faces(ctx: TaskContext, img: Image.Image) -> list[FaceEmbedding]:
    try:
        return ctx.walkieAI.face_recognition.embed(img)
    except Exception as exc:
        print(f"[identity] face embed failed ({exc})")
        return []


def _detect_persons(ctx: TaskContext, img: Image.Image) -> list[PersonPose]:
    """Pose-detected people in *img* (with keypoints); [] on any failure."""
    try:
        return ctx.walkieAI.pose_estimation.estimate(img)
    except Exception as exc:
        print(f"[identity] pose estimate failed ({exc})")
        return []


def _detect_persons_xyxy(ctx: TaskContext, img: Image.Image) -> list[BBox]:
    return [cxcywh_to_xyxy(p.bbox) for p in _detect_persons(ctx, img)]


# COCO keypoint indices for the body-scale proximity proxy (see _person_scale_px).
_KP_L_SHOULDER, _KP_R_SHOULDER = 5, 6
_KP_L_HIP, _KP_R_HIP = 11, 12


def _person_scale_px(pose: PersonPose, *, min_kp_conf: float) -> float | None:
    """A distance-monotonic pixel size for *pose*, used as a "closeness" proxy.

    Torso length (shoulder midpoint → hip midpoint) in pixels: it grows as the
    person nears the camera and, unlike the raw bbox, is unmoved by a raised arm
    and survives the legs being clipped at the frame edge. Falls back to shoulder
    width, then bbox height, when the hips/shoulders aren't confidently seen.
    Only keypoints above *min_kp_conf* are trusted; ``None`` when too little of
    the body is measurable (the box then carries no proximity bonus).
    """
    kp = {k.index: k for k in pose.keypoints if k.confidence >= min_kp_conf}

    def midpoint(a: int, b: int) -> tuple[float, float] | None:
        if a in kp and b in kp:
            return (kp[a].x + kp[b].x) / 2, (kp[a].y + kp[b].y) / 2
        return None

    shoulders = midpoint(_KP_L_SHOULDER, _KP_R_SHOULDER)
    hips = midpoint(_KP_L_HIP, _KP_R_HIP)
    if shoulders is not None and hips is not None:
        return math.hypot(shoulders[0] - hips[0], shoulders[1] - hips[1])
    if _KP_L_SHOULDER in kp and _KP_R_SHOULDER in kp:
        ls, rs = kp[_KP_L_SHOULDER], kp[_KP_R_SHOULDER]
        return math.hypot(ls.x - rs.x, ls.y - rs.y)
    _cx, _cy, _w, h = pose.bbox
    return float(h) if h > 0 else None


def _store_enrollment(
    ctx: TaskContext,
    img: Image.Image,
    person_id: str,
    face: FaceEmbedding | None,
    person_box: BBox,
    *,
    name: str,
    drink: str,
    attributes: str,
) -> bool:
    """Embed the attire crop and write the record; whichever modalities exist."""
    app_emb = _appearance_embedding(ctx, img, person_box)
    if face is None and app_emb is None:
        return False
    try:
        # The store keys records on a face vector; an attire-only sighting
        # enrolls a zero face vector, which the store treats as "no face known"
        # (never face-matched; recognize_fused goes appearance-only).
        ctx.people.enroll(
            name,
            drink,
            face.embedding if face is not None else [0.0] * 512,
            person_id=person_id,
            attributes=attributes,
            app_embedding=app_emb,
            frame=img,
            face_bbox_xyxy=face.bbox_xyxy if face is not None else None,
        )
        modal = "face+appearance" if (face and app_emb) else ("face" if face else "appearance")
        print(f"[identity] enrolled {person_id} ({modal})")
        return True
    except Exception as exc:
        print(f"[identity] enroll {person_id} failed ({exc})")
        return False


def enroll_guest(
    ctx: TaskContext,
    img: Image.Image,
    person_id: str,
    *,
    name: str = "",
    drink: str = "",
    attributes: str = "",
) -> bool:
    """Remember the person in front of the camera under a stable id.

    Embeds the largest (nearest) face plus an attire vector from their person
    crop and enrolls both into ``ctx.people``. Enrolls with whichever
    modalities succeeded; returns False when neither did (or no store).
    """
    if ctx.people is None:
        return False
    faces = _detect_faces(ctx, img)
    face = max(faces, key=lambda f: f.area()) if faces else None
    persons_xyxy = _detect_persons_xyxy(ctx, img)

    if face is not None:
        person_box = _person_bbox_for_face(face, persons_xyxy, img.width, img.height)
    elif len(persons_xyxy) == 1:
        person_box = persons_xyxy[0]  # back turned / face hidden: attire only
    else:
        print(f"[identity] enroll {person_id}: no face and no unique person; skipping")
        return False
    return _store_enrollment(
        ctx, img, person_id, face, person_box, name=name, drink=drink, attributes=attributes
    )


def enroll_person_in_box(
    ctx: TaskContext,
    img: Image.Image,
    person_xyxy: BBox,
    person_id: str,
    *,
    name: str = "",
    drink: str = "",
    attributes: str = "",
) -> bool:
    """Enroll the specific person inside *person_xyxy* (e.g. the seated host).

    Unlike :func:`enroll_guest` this never grabs the largest face in the frame
    — only a face whose center lies inside the given box counts, so a guest
    standing next to the robot can't be enrolled as the host.
    """
    if ctx.people is None:
        return False
    x1, y1, x2, y2 = person_xyxy
    inside = [
        f
        for f in _detect_faces(ctx, img)
        if x1 <= (f.bbox_xyxy[0] + f.bbox_xyxy[2]) / 2 <= x2
        and y1 <= (f.bbox_xyxy[1] + f.bbox_xyxy[3]) / 2 <= y2
    ]
    face = max(inside, key=lambda f: f.area()) if inside else None
    return _store_enrollment(
        ctx, img, person_id, face, person_xyxy, name=name, drink=drink, attributes=attributes
    )


def locate_people(
    ctx: TaskContext, frames: list[Image.Image], person_ids: list[str]
) -> dict[str, tuple[int, BBox]]:
    """Find enrolled people across one or more frames: {id: (frame_index, bbox)}.

    Face FIRST, across every frame: each detected face is scored against the
    whole store with ``recognize_fused`` (face + attire of the same person
    crop), then matches from all frames are greedily assigned to identities by
    similarity — so a guest whose face shows in any view wins, and the host
    (also enrolled) acts as a distractor a guest match must beat. Only the ids
    still missing then get an appearance-only fallback pass over the remaining
    pose-detected persons in every frame (covers a guest turned away in all
    views). The returned frame index ties each match to the snapshot it came
    from, so the caller can lift the bbox against that frame's capture-time
    geometry. Partial results on any failure.
    """
    if ctx.people is None or ctx.people.count() == 0:
        return {}
    wanted = set(person_ids)
    min_score = ctx.people.fused_min_score()

    # Per-frame pose boxes, computed once and reused by both passes; face
    # candidates tagged with the frame they were seen in.
    persons_by_frame: list[list[BBox]] = []
    candidates: list[tuple[float, str, int, BBox]] = []  # (sim, id, frame, box)
    for fi, img in enumerate(frames):
        faces = _detect_faces(ctx, img)
        persons_xyxy = _detect_persons_xyxy(ctx, img)
        persons_by_frame.append(persons_xyxy)
        for face in faces:
            person_box = _person_bbox_for_face(face, persons_xyxy, img.width, img.height)
            app_emb = _appearance_embedding(ctx, img, person_box)
            rec = ctx.people.recognize_fused(
                face.embedding,
                app_emb,
                face_confidence=face.det_score,
                min_score=min_score,
            )
            if rec is not None and rec.similarity is not None:
                candidates.append((rec.similarity, rec.id, fi, person_box))

    located: dict[str, tuple[int, BBox]] = {}
    used: set[tuple[int, BBox]] = set()  # (frame, box) claimed by a match
    for sim, rid, fi, box in sorted(candidates, key=lambda c: c[0], reverse=True):
        if rid in located or (fi, box) in used:
            continue
        located[rid] = (fi, box)
        used.add((fi, box))
        print(f"[identity] located {rid} by face (frame {fi}, similarity {sim:.2f})")

    # Appearance-only fallback for ids still missing (face turned away in every
    # frame) — scan the unclaimed pose boxes across all frames.
    missing = wanted - set(located)
    for fi, persons_xyxy in enumerate(persons_by_frame):
        if not missing:
            break
        for box in persons_xyxy:
            if (fi, box) in used:
                continue
            app_emb = _appearance_embedding(ctx, frames[fi], box)
            if app_emb is None:
                continue
            rec = ctx.people.recognize_fused(None, app_emb, min_score=min_score)
            if rec is not None and rec.id in missing:
                located[rec.id] = (fi, box)
                used.add((fi, box))
                missing.discard(rec.id)
                print(f"[identity] located {rec.id} by appearance only (frame {fi})")
            if not missing:
                break

    return {rid: v for rid, v in located.items() if rid in wanted}


def _candidate_persons(
    ctx: TaskContext, img: Image.Image
) -> list[tuple[BBox, float | None]]:
    """Person boxes (xyxy) paired with a pose-keypoint closeness proxy.

    To bound the per-tick embed cost in a crowd, only the largest
    ``HRI_FOLLOW_APPEARANCE_MAX_CANDIDATES`` person boxes are kept; 0 (the
    default) keeps all, so a host who has walked off and shrunk below a guest
    isn't dropped from the candidate set.
    """
    persons = _detect_persons(ctx, img)
    max_cand = int(os.getenv("HRI_FOLLOW_APPEARANCE_MAX_CANDIDATES", "0"))
    if max_cand > 0:
        persons = sorted(
            persons, key=lambda p: p.bbox[2] * p.bbox[3], reverse=True
        )[:max_cand]
    min_kp_conf = float(os.getenv("HRI_FOLLOW_POSE_KP_CONF", "0.3"))
    return [
        (cxcywh_to_xyxy(p.bbox), _person_scale_px(p, min_kp_conf=min_kp_conf))
        for p in persons
    ]


def _cand_for_face(
    face: FaceEmbedding,
    cands: list[tuple[BBox, float | None]],
    img_w: int,
    img_h: int,
) -> tuple[BBox, float | None]:
    """The candidate ``(box, scale)`` whose box contains *face*'s center (the
    smallest such), else an expanded-face box with unknown scale — pose missed
    that person, but their recognized face must still be followable."""
    fx = (face.bbox_xyxy[0] + face.bbox_xyxy[2]) / 2
    fy = (face.bbox_xyxy[1] + face.bbox_xyxy[3]) / 2
    containing = [(b, s) for (b, s) in cands if b[0] <= fx <= b[2] and b[1] <= fy <= b[3]]
    if containing:
        return min(containing, key=lambda bs: (bs[0][2] - bs[0][0]) * (bs[0][3] - bs[0][1]))
    return _expand_face_to_person(face.bbox_xyxy, img_w, img_h), None


def _pick_closest_qualifier(
    qualifying: list[tuple[float, BBox, float | None]],
    person_id: str,
    modality: str,
) -> BBox | None:
    """Among identity-qualifying boxes, the best by similarity PLUS a proximity
    bonus (``HRI_FOLLOW_PROXIMITY_WEIGHT`` × the box's pose scale relative to the
    nearest qualifier). The proximity term only breaks ties between genuine
    matches — every box here already cleared the identity gate, so a near
    bystander can never win. Boxes with no measurable scale carry no bonus.
    ``None`` when nothing qualifies.
    """
    if not qualifying:
        return None
    weight = float(os.getenv("HRI_FOLLOW_PROXIMITY_WEIGHT", "0.15"))
    max_scale = max((s for _sim, _b, s in qualifying if s is not None), default=0.0)
    best: tuple[float, float, BBox, float] | None = None  # (combined, sim, box, prox)
    for sim, box, scale in qualifying:
        prox = scale / max_scale if (scale is not None and max_scale > 0) else 0.0
        combined = sim + weight * prox
        if best is None or combined > best[0]:
            best = (combined, sim, box, prox)
    _combined, sim, box, prox = best
    print(
        f"[identity] follow: matched {person_id} by {modality} "
        f"(similarity {sim:.2f}, proximity {prox:.2f})"
    )
    return box


def _select_by_face(
    ctx: TaskContext,
    img: Image.Image,
    cands: list[tuple[BBox, float | None]],
    person_id: str,
) -> BBox | None:
    """Face pass: the closest box whose face is recognized as *person_id*.

    Faces are far more discriminative than clothing, so this locks onto the
    right person even when the crowd dresses alike. Each detected face is scored
    (face only) against every enrolled person; a box qualifies only when
    *person_id* clears the fused-match floor AND out-scores every OTHER enrolled
    person by ``HRI_FOLLOW_FACE_MARGIN``. ``None`` when no face in view is the
    target (turned away / too far / occluded, or *person_id* enrolled without a
    face) — the caller then falls back to attire.
    """
    faces = _detect_faces(ctx, img)
    if not faces:
        return None
    min_score = ctx.people.fused_min_score()
    margin = float(os.getenv("HRI_FOLLOW_FACE_MARGIN", "0.05"))
    qualifying: list[tuple[float, BBox, float | None]] = []
    for face in faces:
        box, scale = _cand_for_face(face, cands, img.width, img.height)
        scores = ctx.people.fused_scores(face.embedding)  # face-only path
        host_sim = scores.get(person_id)
        if host_sim is None or host_sim < min_score:
            continue
        best_other = max((s for rid, s in scores.items() if rid != person_id), default=0.0)
        if host_sim - best_other < margin:
            continue
        qualifying.append((host_sim, box, scale))
    return _pick_closest_qualifier(qualifying, person_id, "face")


def _select_by_attire(
    ctx: TaskContext,
    img: Image.Image,
    cands: list[tuple[BBox, float | None]],
    person_id: str,
) -> BBox | None:
    """Attire fallback: the closest box whose OSNet attire vector is *person_id*'s.

    Used when no face was recognized this tick. Embeds each candidate crop and
    scores it against every enrolled person (appearance-only fusion). A box
    qualifies only when *person_id* clears the match floor AND beats the best
    OTHER enrolled person by ``HRI_FOLLOW_APPEARANCE_MARGIN`` — at a party people
    dress alike, so a bare top-1 lead flips onto a guest on embedding noise; the
    margin demands the host clearly win. ``None`` when nobody qualifies.
    """
    min_score = ctx.people.fused_min_score()
    margin = float(os.getenv("HRI_FOLLOW_APPEARANCE_MARGIN", "0.05"))
    qualifying: list[tuple[float, BBox, float | None]] = []
    for box, scale in cands:
        app_emb = _appearance_embedding(ctx, img, box)
        if app_emb is None:
            continue
        scores = ctx.people.fused_scores(None, app_emb)
        host_sim = scores.get(person_id)
        if host_sim is None or host_sim < min_score:
            continue
        best_other = max((s for rid, s in scores.items() if rid != person_id), default=0.0)
        if host_sim - best_other < margin:
            continue
        qualifying.append((host_sim, box, scale))
    return _pick_closest_qualifier(qualifying, person_id, "appearance")


def select_person_to_follow(ctx: TaskContext, snap, person_id: str) -> BBox | None:
    """:func:`skills.follow_person` selector: the person box matching *person_id*,
    FACE first then ATTIRE, biased toward the nearest qualifying candidate.

    Per tick: pose-estimate the people in view, each box paired with a
    pose-keypoint closeness proxy (:func:`_person_scale_px`). A FACE pass runs
    first (:func:`_select_by_face`) — faces are far more discriminative than
    clothing, so a face match holds the right person in a similarly-dressed
    crowd. Only when no face in view is recognized as *person_id* this tick
    (turned away, too far, occluded, or enrolled without a face) does it fall
    back to an ATTIRE pass (:func:`_select_by_attire`, the previous behaviour).
    Either pass returns the qualifying box with the highest
    similarity-plus-proximity score, so among equally good identity matches the
    nearest body wins. ``None`` when nobody qualifies — the follow loop then
    coasts on its motion prediction.

    *person_id* must already be enrolled (the host is, at ``OfferSeat``); the
    face pass needs a face embedding, the attire fallback an attire one.
    """
    if snap is None or ctx.people is None or ctx.people.count() == 0:
        return None
    img = snap.img
    cands = _candidate_persons(ctx, img)
    if not cands:
        return None
    box = _select_by_face(ctx, img, cands, person_id)
    if box is not None:
        return box
    return _select_by_attire(ctx, img, cands, person_id)


def wait_until_seated(
    ctx: TaskContext,
    person_id: str,
    *,
    dwell_sec: float | None = None,
    timeout_sec: float | None = None,
    poll_sec: float | None = None,
) -> tuple[bool, tuple[float, float] | None]:
    """Block until *person_id* has been in view continuously for *dwell_sec*.

    Polls snapshots, recognizing the (already enrolled) person each tick with
    :func:`locate_people` — face first, attire fallback, so a guest turned away
    while settling into a seat still counts. A continuous run of sightings that
    spans *dwell_sec* is taken as "they have sat down". Returns
    ``(seated, last_world_xy)`` — the map-frame point of the last sighting,
    lifted against its snapshot, so the caller can persist where the guest
    actually ended up (which may differ from the offered seat). On *timeout_sec*
    returns ``(False, last_xy_or_None)``; the caller proceeds regardless, since a
    no-show must not stall the run. Defaults come from the ``HRI_SEATED_*`` env.
    """
    if dwell_sec is None:
        dwell_sec = float(os.getenv("HRI_SEATED_DWELL_SEC", "3"))
    if timeout_sec is None:
        timeout_sec = float(os.getenv("HRI_SEATED_WAIT_TIMEOUT_SEC", "20"))
    if poll_sec is None:
        poll_sec = float(os.getenv("HRI_SEATED_POLL_SEC", "0.5"))
    deadline = time.monotonic() + timeout_sec
    seen_since: float | None = None  # start of the current uninterrupted run
    last_xy: tuple[float, float] | None = None
    while time.monotonic() < deadline:
        snap = ctx.snapshot()
        found = (
            locate_people(ctx, [snap.img], [person_id]).get(person_id)
            if snap is not None else None
        )
        now = time.monotonic()
        if found is not None:
            _fi, box = found
            xy = lift_bbox_world_xy(ctx, snap, box)
            if xy is not None:
                last_xy = xy
            if seen_since is None:
                seen_since = now
            if now - seen_since >= dwell_sec:
                print(f"[identity] {person_id} seated (in view {dwell_sec:.0f}s)")
                return True, last_xy
        else:
            seen_since = None  # lost sight: the dwell must be continuous, restart
        time.sleep(poll_sec)
    print(f"[identity] {person_id} not confirmed seated before timeout; proceeding")
    return False, last_xy
