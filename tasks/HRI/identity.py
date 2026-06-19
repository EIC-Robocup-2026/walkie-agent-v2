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
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from client import FaceEmbedding, PersonPose
from perception.people_store import _cosine_sim, _mean_unit
from tasks.base import TaskContext

from .skills import BBox, cxcywh_to_xyxy, lift_bbox_world_xy


def _avg_unit(vectors: list[list[float]]) -> list[float] | None:
    """L2-normalized mean of unit vectors, or None when the list is empty.

    Thin empty-guarded wrapper over ``people_store._mean_unit`` (which assumes a
    non-empty list), so callers can average a best-effort burst that may have
    yielded nothing.
    """
    if not vectors:
        return None
    return _mean_unit(vectors)


def _outlier_rejection_enabled() -> bool:
    return os.getenv("HRI_BURST_OUTLIER_REJECT", "1").lower() in ("1", "true", "yes")


def _reject_outliers(vectors: list[list[float]]) -> list[list[float]]:
    """Drop burst frames whose embedding is cosine-far from the burst centroid.

    A multi-frame capture can catch a bystander, a half-turned head, or a
    motion-blurred body in ONE frame; averaging that in corrupts the stored
    vector. So: compute the unit-mean centroid, score each vector's cosine
    similarity to it, and drop vectors below BOTH an absolute floor
    (``HRI_BURST_OUTLIER_MIN_SIM``) AND a robust ``median - K*MAD`` gate
    (``HRI_BURST_OUTLIER_MAD_K``; MAD, not std, so the very outlier we want gone
    can't inflate the spread). Conservative: only clear outliers go, at least one
    vector (the closest to the centroid) is always kept, and it is a no-op for
    fewer than 3 vectors (too few to tell signal from outlier) or when disabled.
    """
    if not vectors:
        return vectors
    if len(vectors) < 3 or not _outlier_rejection_enabled():
        return list(vectors)
    centroid = _mean_unit(vectors)
    sims = [_cosine_sim(v, centroid) for v in vectors]
    srt = sorted(sims)
    med = srt[len(srt) // 2]
    devs = sorted(abs(s - med) for s in sims)
    mad = devs[len(devs) // 2]
    k = float(os.getenv("HRI_BURST_OUTLIER_MAD_K", "3.0"))
    min_sim = float(os.getenv("HRI_BURST_OUTLIER_MIN_SIM", "0.5"))
    mad_gate = med - k * mad
    keep = [v for v, s in zip(vectors, sims) if s >= min_sim and s >= mad_gate]
    if not keep:  # never strip a modality to nothing — keep the most central frame
        best_i = max(range(len(sims)), key=lambda i: sims[i])
        return [vectors[best_i]]
    if len(keep) < len(vectors):
        print(f"[identity] burst outlier rejection: kept {len(keep)}/{len(vectors)}")
    return keep


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
        return ctx.walkieAI.image.appearance(crop)
    except Exception as exc:
        print(f"[identity] appearance embed failed ({exc})")
        return None


def _detect_faces(ctx: TaskContext, img: Image.Image) -> list[FaceEmbedding]:
    try:
        return ctx.walkieAI.image.faces(img)
    except Exception as exc:
        print(f"[identity] face embed failed ({exc})")
        return []


def _detect_persons(ctx: TaskContext, img: Image.Image) -> list[PersonPose]:
    """Pose-detected people in *img* (with keypoints); [] on any failure."""
    try:
        return ctx.walkieAI.image.estimate_poses(img)
    except Exception as exc:
        print(f"[identity] pose estimate failed ({exc})")
        return []


def _detect_persons_xyxy(ctx: TaskContext, img: Image.Image) -> list[BBox]:
    return [cxcywh_to_xyxy(p.bbox) for p in _detect_persons(ctx, img)]


def _largest_person_box(ctx: TaskContext, img: Image.Image) -> BBox | None:
    """The largest pose-detected person box in *img*, or None.

    Used for the attire crop when the head is tilted DOWN and the face may be out
    of frame, so the box can't be anchored on a face: the nearest (largest) body
    is the guest standing right in front of the robot.
    """
    boxes = _detect_persons_xyxy(ctx, img)
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


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


def enroll_guest_frames(
    ctx: TaskContext,
    face_imgs: list[Image.Image],
    app_imgs: list[Image.Image],
    person_id: str,
    *,
    name: str = "",
    drink: str = "",
    attributes: str = "",
) -> bool:
    """Enroll one person from MULTIPLE frames, averaging embeddings, ONCE.

    *face_imgs* are head-level frames (the most reliable face shot); *app_imgs*
    are head-down frames that frame the body (the best OSNet attire crop). The
    largest face per face-frame is embedded and the unit-mean of those vectors is
    stored; the largest pose-detected person box per app-frame is embedded and
    the unit-mean of those attire vectors is stored. Averaging WITHIN a single
    capture burst (same outfit, same moment) is safe — it does not blur
    identities the way the store's cross-session appearance averaging would — so
    we enroll ONCE with the burst-averaged vectors instead of N latest-wins
    overwrites. Outlier frames are dropped per modality first (:func:`_reject_outliers`).

    Degrades on every failure (never raises): no face in any face-frame → enroll
    attire-only (zero face vector, treated by the store as "no face known"); no
    person box in any app-frame → fall back to the best face-frame's person box.
    Returns False only when neither modality produced any vector (or no store).
    """
    if ctx.people is None:
        return False

    # Face: largest face per face frame, averaged.
    face_embs: list[list[float]] = []
    best_face: FaceEmbedding | None = None  # kept for the thumbnail bbox
    best_face_img: Image.Image | None = None
    best_face_area = -1.0
    for img in face_imgs:
        faces = _detect_faces(ctx, img)
        if not faces:
            continue
        f = max(faces, key=lambda f: f.area())
        face_embs.append(list(f.embedding))
        if f.area() > best_face_area:
            best_face, best_face_img, best_face_area = f, img, f.area()
    face_embs = _reject_outliers(face_embs)
    avg_face = _avg_unit(face_embs)

    # Appearance: largest person box per app frame, averaged.
    app_embs: list[list[float]] = []
    for img in app_imgs:
        box = _largest_person_box(ctx, img)
        if box is None:
            continue
        emb = _appearance_embedding(ctx, img, box)
        if emb is not None:
            app_embs.append(list(emb))
    # Fallback: no body box in any app frame — use the best face frame's person
    # box (which itself falls back to an expanded-face crop), so attire is never
    # empty when we at least saw a face.
    if not app_embs and best_face is not None and best_face_img is not None:
        box = _person_bbox_for_face(
            best_face,
            _detect_persons_xyxy(ctx, best_face_img),
            best_face_img.width, best_face_img.height,
        )
        emb = _appearance_embedding(ctx, best_face_img, box)
        if emb is not None:
            app_embs.append(list(emb))
    app_embs = _reject_outliers(app_embs)
    avg_app = _avg_unit(app_embs)

    if avg_face is None and avg_app is None:
        print(f"[identity] enroll_guest_frames {person_id}: no usable vectors; skipping")
        return False

    # Thumbnail: the best face frame/bbox when we have a face, else the first app
    # frame (no face bbox → the archive step is skipped, which is fine).
    thumb_img = best_face_img if best_face is not None else (app_imgs[0] if app_imgs else None)
    thumb_bbox = best_face.bbox_xyxy if best_face is not None else None
    try:
        ctx.people.enroll(
            name, drink,
            avg_face if avg_face is not None else [0.0] * 512,
            person_id=person_id,
            attributes=attributes,
            app_embedding=avg_app,
            frame=thumb_img,
            face_bbox_xyxy=thumb_bbox,
        )
        modal = (
            "face+appearance" if (avg_face and avg_app)
            else ("face" if avg_face else "appearance")
        )
        print(
            f"[identity] enrolled {person_id} from {len(face_imgs)}f/{len(app_imgs)}a frames "
            f"({modal}; faces={len(face_embs)}, attire={len(app_embs)})"
        )
        return True
    except Exception as exc:
        print(f"[identity] enroll_guest_frames {person_id} failed ({exc})")
        return False


def audit_identity_collisions(ctx: TaskContext) -> list[tuple[str, str, str, float]]:
    """Flag pairs of enrolled people whose face OR attire vectors are too alike.

    Compares every pair among the enrolled ids (host, guest-1, guest-2, ...) on
    both modalities with the store's own cosine metric. A pair at/above
    ``HRI_DUP_FACE_SIM`` (face) or ``HRI_DUP_APP_SIM`` (attire) is reported as
    ``(id_a, id_b, "face"|"appearance", sim)`` and logged as a WARNING. Two
    DISTINCT guests cannot be merged (that would lose a real identity), so this
    only DETECTS — the caller decides what to do (e.g. widen the attire margin so
    recognition leans on the more discriminative face). Read-only on the store;
    best-effort (returns [] on any failure or with fewer than 2 people).
    """
    if ctx.people is None:
        return []
    try:
        people = ctx.people.list_people()
    except Exception as exc:
        print(f"[identity] collision audit: list_people failed ({exc})")
        return []
    if len(people) < 2:
        return []
    try:
        app_by_id = ctx.people.appearance_vectors()
    except Exception as exc:
        print(f"[identity] collision audit: appearance read failed ({exc})")
        app_by_id = {}

    face_gate = float(os.getenv("HRI_DUP_FACE_SIM", "0.75"))
    app_gate = float(os.getenv("HRI_DUP_APP_SIM", "0.85"))
    collisions: list[tuple[str, str, str, float]] = []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = people[i], people[j]
            fa, fb = list(a.embedding), list(b.embedding)
            if fa and fb and any(fa) and any(fb):
                fs = _cosine_sim(fa, fb)
                if fs >= face_gate:
                    print(f"[identity] WARNING near-duplicate FACE: {a.id} vs {b.id} (sim {fs:.2f})")
                    collisions.append((a.id, b.id, "face", fs))
            aa, ab = app_by_id.get(a.id), app_by_id.get(b.id)
            if aa is not None and ab is not None:
                aps = _cosine_sim(aa, ab)
                if aps >= app_gate:
                    print(f"[identity] WARNING near-duplicate ATTIRE: {a.id} vs {b.id} (sim {aps:.2f})")
                    collisions.append((a.id, b.id, "appearance", aps))
    return collisions


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


def _persons_to_candidates(
    persons: list[PersonPose],
) -> list[tuple[BBox, float | None]]:
    """Pair each pose with its closeness proxy, capped by config (pure, no I/O).

    The local half of :func:`_candidate_persons`, split out so the parallel
    follow path can build candidates from a pose result fetched on a worker
    thread without re-detecting.
    """
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


def _candidate_persons(
    ctx: TaskContext, img: Image.Image
) -> list[tuple[BBox, float | None]]:
    """Person boxes (xyxy) paired with a pose-keypoint closeness proxy.

    To bound the per-tick embed cost in a crowd, only the largest
    ``HRI_FOLLOW_APPEARANCE_MAX_CANDIDATES`` person boxes are kept; 0 (the
    default) keeps all, so a host who has walked off and shrunk below a guest
    isn't dropped from the candidate set.
    """
    return _persons_to_candidates(_detect_persons(ctx, img))


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


def _match_by_face(
    ctx: TaskContext,
    img: Image.Image,
    faces: list[FaceEmbedding],
    cands: list[tuple[BBox, float | None]],
    person_id: str,
) -> BBox | None:
    """Score pre-detected *faces* against the store and pick *person_id*'s box.

    The local (no-I/O) half of :func:`_select_by_face`, split out so the parallel
    follow path can hand in faces fetched concurrently with the pose detection.
    """
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
    return _match_by_face(ctx, img, _detect_faces(ctx, img), cands, person_id)


def _score_attire_candidates(
    ctx: TaskContext,
    cands: list[tuple[BBox, float | None]],
    embeds: list[list[float] | None],
    person_id: str,
) -> BBox | None:
    """Pick *person_id*'s box from candidate attire *embeds* aligned with *cands*.

    The local (no-I/O) half of :func:`_select_by_attire`, split out so the
    parallel follow path can fan the per-candidate OSNet embeds across threads
    and hand the results in. ``embeds[i]`` is the vector for ``cands[i]`` (or
    ``None`` if that embed failed). Scoring (``fused_scores``) runs here on the
    calling thread, so the local Chroma store is never touched concurrently.
    """
    min_score = ctx.people.fused_min_score()
    margin = float(os.getenv("HRI_FOLLOW_APPEARANCE_MARGIN", "0.05"))
    qualifying: list[tuple[float, BBox, float | None]] = []
    for (box, scale), app_emb in zip(cands, embeds):
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
    embeds = [_appearance_embedding(ctx, img, box) for box, _scale in cands]
    return _score_attire_candidates(ctx, cands, embeds, person_id)


def _follow_parallel_enabled() -> bool:
    return os.getenv("HRI_FOLLOW_PARALLEL", "0").lower() in ("1", "true", "yes")


def _box_center(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _gate_candidates(
    cands: list[tuple[BBox, float | None]],
    hint_box: BBox | None,
    *,
    radius_scale: float,
) -> list[tuple[BBox, float | None]]:
    """Candidates whose box center is near *hint_box* (the last tracked host box).

    While following, the host is in nearly the same image spot tick-to-tick, so
    only the box(es) within ``radius_scale × hint_box_width`` of *hint_box*'s
    center need the costly per-candidate OSNet embed — usually just one. Returns
    *cands* unchanged when there's no hint (re-acquiring) or gating is disabled
    (``radius_scale <= 0``). May return fewer (or zero) when the host has moved /
    is occluded; the caller then falls back to the full set so the lock is never
    silently dropped. Pure (no I/O), so it's shared by the serial and parallel
    paths.
    """
    if hint_box is None or radius_scale <= 0:
        return cands
    hx, hy = _box_center(hint_box)
    radius = radius_scale * (hint_box[2] - hint_box[0])
    return [
        (box, scale)
        for (box, scale) in cands
        if math.hypot(_box_center(box)[0] - hx, _box_center(box)[1] - hy) <= radius
    ]


def _gate_radius_scale() -> float:
    return float(os.getenv("HRI_FOLLOW_GATE_RADIUS_SCALE", "1.5"))


def _attire_pass(
    ctx: TaskContext,
    img: Image.Image,
    cands: list[tuple[BBox, float | None]],
    person_id: str,
    hint_box: BBox | None,
) -> BBox | None:
    """Serial attire fallback, gated to *hint_box* then widened on a miss.

    Embeds only the candidates near the last tracked box first (typically one),
    and only if none of those qualify — and gating actually narrowed the set —
    re-embeds the full candidate list to re-acquire. The identity margin gate in
    :func:`_select_by_attire` runs on every embedded box, so the gated shortcut
    can never lock onto the wrong person; it just saves the embeds for the boxes
    that obviously aren't the host.
    """
    gated = _gate_candidates(cands, hint_box, radius_scale=_gate_radius_scale())
    box = _select_by_attire(ctx, img, gated, person_id)
    if box is None and len(gated) != len(cands):
        box = _select_by_attire(ctx, img, cands, person_id)
    return box


def _embed_and_score_attire_parallel(
    ex: ThreadPoolExecutor,
    ctx: TaskContext,
    img: Image.Image,
    cands: list[tuple[BBox, float | None]],
    person_id: str,
) -> BBox | None:
    """Embed *cands*' crops concurrently on *ex* and score them (parallel attire).

    ``ex.map`` preserves order, so ``embeds[i]`` aligns with ``cands[i]``;
    scoring (:meth:`fused_scores`) stays on the calling thread so the local
    Chroma store is never touched concurrently. ``None`` for an empty *cands*.
    """
    if not cands:
        return None
    embeds = list(ex.map(lambda bs: _appearance_embedding(ctx, img, bs[0]), cands))
    return _score_attire_candidates(ctx, cands, embeds, person_id)


def _select_person_to_follow_parallel(
    ctx: TaskContext,
    img: Image.Image,
    person_id: str,
    *,
    hint_box: BBox | None,
    run_face: bool,
) -> BBox | None:
    """Overlap one follow tick's independent server round-trips (HRI_FOLLOW_PARALLEL=1).

    Pose estimation and face detection each need only the full frame and hit
    different sub-clients (separate ``requests.Session``s, designed for
    concurrent use), so they run on worker threads at the same time. The
    attire-fallback embeds — one OSNet call per candidate box, the costly part in
    a crowd — are likewise fanned out, but only over the candidates *gated* to
    *hint_box*. All identity scoring (:meth:`fused_scores`, which reads the local
    Chroma store) stays on the calling thread, so no Chroma access is concurrent.
    Returns exactly the box the serial path would for the same *hint_box* /
    *run_face*.

    Whether this lowers per-tick latency depends on the server handling
    concurrent requests; with one GPU serializing inference the gain is limited
    to overlapping encode/transfer/pre-post. Measure with HRI_FOLLOW_TRACK_DEBUG.
    """
    workers = max(2, int(os.getenv("HRI_FOLLOW_PARALLEL_WORKERS", "8")))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_persons = ex.submit(_detect_persons, ctx, img)  # pose, own session
        # Skip the face round-trip on throttled ticks (the host's back is to the
        # camera while following, so the face pass matches ~never — see
        # HRI_FOLLOW_FACE_EVERY_N).
        fut_faces = ex.submit(_detect_faces, ctx, img) if run_face else None
        cands = _persons_to_candidates(fut_persons.result())
        if not cands:
            return None
        if fut_faces is not None:
            box = _match_by_face(ctx, img, fut_faces.result(), cands, person_id)
            if box is not None:
                return box  # face match: never pay for the attire embeds (as serial)
        # Attire fallback: gate to the last tracked box, widen on a miss.
        gated = _gate_candidates(cands, hint_box, radius_scale=_gate_radius_scale())
        box = _embed_and_score_attire_parallel(ex, ctx, img, gated, person_id)
        if box is None and len(gated) != len(cands):
            box = _embed_and_score_attire_parallel(ex, ctx, img, cands, person_id)
        return box


def select_person_to_follow(
    ctx: TaskContext,
    snap,
    person_id: str,
    *,
    hint_box: BBox | None = None,
    run_face: bool = True,
) -> BBox | None:
    """:func:`skills.follow_person` selector: the person box matching *person_id*,
    FACE first then ATTIRE, biased toward the nearest qualifying candidate.

    Per tick: pose-estimate the people in view, each box paired with a
    pose-keypoint closeness proxy (:func:`_person_scale_px`). A FACE pass runs
    first (:func:`_select_by_face`, only when *run_face*) — faces are far more
    discriminative than clothing, so a face match holds the right person in a
    similarly-dressed crowd. Only when no face in view is recognized as
    *person_id* this tick (turned away, too far, occluded, enrolled without a
    face, or the face pass throttled off) does it fall back to an ATTIRE pass
    (:func:`_attire_pass`). Either pass returns the qualifying box with the
    highest similarity-plus-proximity score, so among equally good identity
    matches the nearest body wins. ``None`` when nobody qualifies — the follow
    loop then coasts on its motion prediction.

    *hint_box* is the last tracked host box: when given, the attire fallback only
    embeds the candidate(s) near it (re-embedding the full set on a miss), which
    is what cuts the per-tick OSNet cost from N to ~1. *run_face* throttles the
    (usually fruitless, while following) face round-trip. Both default to the
    pre-tracking behaviour (no hint, face every tick), so direct callers and the
    enrollment paths are unaffected — the stateful :class:`FollowSelector`
    supplies them during a follow.

    With ``HRI_FOLLOW_PARALLEL=1`` the independent per-tick server calls run
    concurrently (:func:`_select_person_to_follow_parallel`); the result is
    identical, only the latency differs.

    *person_id* must already be enrolled (the host is, at ``OfferSeat``); the
    face pass needs a face embedding, the attire fallback an attire one.
    """
    if snap is None or ctx.people is None or ctx.people.count() == 0:
        return None
    img = snap.img
    if _follow_parallel_enabled():
        return _select_person_to_follow_parallel(
            ctx, img, person_id, hint_box=hint_box, run_face=run_face
        )
    cands = _candidate_persons(ctx, img)
    if not cands:
        return None
    if run_face:
        box = _select_by_face(ctx, img, cands, person_id)
        if box is not None:
            return box
    return _attire_pass(ctx, img, cands, person_id, hint_box)


class FollowSelector:
    """Stateful :func:`skills.follow_person` selector: tracks the host tick-to-tick.

    The follow loop calls ``select(ctx, snap)`` as a plain 2-arg callable; this
    object satisfies that while carrying the little state that makes a follow
    cheap — the last box it locked onto (the attire-gate hint) and a tick counter
    (the face-pass throttle). Each call:

    * runs the FACE pass only when re-acquiring (no current lock) or every
      ``HRI_FOLLOW_FACE_EVERY_N`` ticks — while following, the host's back is to
      the camera so the face pass otherwise burns a full-frame inference every
      tick for ~no matches;
    * gates the attire embed to the last box via ``hint_box`` (full scan on a
      miss);
    * remembers the returned box (``None`` clears the lock, so the *next* tick
      re-acquires at full power — face on, no gate).

    One instance per follow; not thread-safe (the loop is single-threaded).
    """

    def __init__(self, person_id: str) -> None:
        self.person_id = person_id
        self.last_box: BBox | None = None
        self.tick = 0

    def _run_face_this_tick(self) -> bool:
        if self.last_box is None:
            return True  # re-acquiring: use every modality
        every_n = max(1, int(os.getenv("HRI_FOLLOW_FACE_EVERY_N", "5")))
        return self.tick % every_n == 0

    def __call__(self, ctx: TaskContext, snap) -> BBox | None:
        box = select_person_to_follow(
            ctx,
            snap,
            self.person_id,
            hint_box=self.last_box,
            run_face=self._run_face_this_tick(),
        )
        self.last_box = box
        self.tick += 1
        return box


def make_follow_selector(person_id: str) -> FollowSelector:
    """A fresh stateful follow selector for *person_id* (see :class:`FollowSelector`)."""
    return FollowSelector(person_id)


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
