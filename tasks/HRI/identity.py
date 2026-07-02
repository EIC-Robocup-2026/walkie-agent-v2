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
from walkie_world.people.store import _cosine_sim, _mean_unit
from tasks.base import TaskContext

from tasks.skills import BBox, cxcywh_to_xyxy, lift_bbox_world_xy

from . import prompts


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


def distill_appearance_caption(ctx: TaskContext, raw: str | None) -> str | None:
    """Distill a rambling VLM caption down to the PERSON's appearance only.

    The server's caption model narrates the whole scene ("The image shows a
    woman standing in front of a projector screen in a room with chairs...")
    regardless of the prompt, so the LLM extracts just the person's own details
    (clothing colors, hair, glasses, ...) before the text is stored or spoken.
    Gated by ``HRI_APPEARANCE_DISTILL`` (default on); best-effort — any failure,
    a ctx without an extractor (unit-test fakes), or a null extraction returns
    *raw* unchanged.
    """
    if not raw or not raw.strip():
        return raw
    if os.getenv("HRI_APPEARANCE_DISTILL", "1").lower() not in ("1", "true", "yes"):
        return raw
    extract = getattr(ctx, "extract", None)
    if extract is None:
        return raw
    try:
        out = extract(
            prompts.PersonAppearance, prompts.APPEARANCE_DISTILL_INSTRUCTIONS, raw
        )
    except Exception as exc:
        print(f"[identity] appearance distill failed ({exc}); keeping raw caption")
        return raw
    desc = (out.description or "").strip() if out is not None else ""
    if not desc:
        return raw
    if desc != raw.strip():
        print(f"[identity] appearance distilled: {desc!r} (raw: {raw!r})")
    return desc


def caption_person_appearance(
    ctx: TaskContext, img: Image.Image, person_xyxy: BBox | None = None
) -> str | None:
    """TEXT appearance description of the person in *img*; None on any failure.

    Crops to *person_xyxy* (with a small margin so clothing isn't clipped at the
    bbox edge) when given, else captions the whole frame and lets the prompt
    single out the person. The caption is the spoken/introduction-facing sibling
    of the OSNet attire embedding: clothing + colors, hair, glasses, distinctive
    features (``prompts.APPEARANCE_CAPTION_PROMPT``). The raw caption is then
    LLM-distilled to person-only details (:func:`distill_appearance_caption`),
    since the caption model narrates the whole scene no matter the prompt.
    Best-effort — never raises.
    """
    crop = img
    if person_xyxy is not None:
        x1, y1, x2, y2 = person_xyxy
        m = 20  # px padding so clothing isn't clipped at the bbox edge
        crop = img.crop((
            max(0, int(x1 - m)), max(0, int(y1 - m)),
            min(img.width, int(x2 + m)), min(img.height, int(y2 + m)),
        ))
    try:
        raw = ctx.walkieAI.image.caption(
            crop, prompt=prompts.APPEARANCE_CAPTION_PROMPT
        )
    except Exception as exc:
        print(f"[identity] appearance caption failed ({exc})")
        return None
    return distill_appearance_caption(ctx, raw)


def _detect_faces(ctx: TaskContext, img: Image.Image) -> list[FaceEmbedding]:
    try:
        return ctx.walkieAI.image.faces(img)
    except Exception as exc:
        print(f"[identity] face embed failed ({exc})")
        return []


def _face_trusted(face: FaceEmbedding, *, hard: bool) -> bool:
    """Quality gate before trusting a face for an identity decision.

    Always requires the detector's own confidence to clear
    ``HRI_RECOG_MIN_DET_SCORE`` (default 0.5, the ``face_conf_med`` fusion knee —
    below it the embedding is too noisy to *label* someone). Only for a HARD
    decision (an introduction label, or re-acquiring a lost follow lock) does it
    also demand a minimum on-screen face area (``HRI_RECOG_MIN_FACE_AREA_PX``,
    default ~50×50 px — a person across the living room still clears it; NOT the
    greeter's ``HRI_FACE_MIN_AREA_PX``, which is sized for someone standing right
    at the door): while merely *maintaining* a lock we must not drop a small,
    far-away host face. Frontalness (``HRI_RECOG_REQUIRE_FRONTAL``) is an opt-in
    hook, off by default because the face box arrives without its pose keypoints.
    """
    min_det = float(os.getenv("HRI_RECOG_MIN_DET_SCORE", "0.5"))
    if face.det_score < min_det:
        return False
    if hard:
        min_area = float(os.getenv("HRI_RECOG_MIN_FACE_AREA_PX", "2500"))
        if face.area() < min_area:
            return False
    return True


def _dedup_person_id(
    ctx: TaskContext,
    face_emb,
    app_emb,
    det_score: float,
    requested_id: str,
    *,
    pinned: bool = True,
) -> str:
    """Guard against enrolling an already-known person as a brand-new record.

    Before writing, check whether the person about to be enrolled STRONGLY matches
    someone already in the store (``recognize_fused`` at
    ``HRI_ENROLL_DEDUP_MIN_SCORE``, default 0.7 — deliberately higher than the 0.5
    recognition floor, so only a confident match reuses/flags an id and a weak one
    safely creates a fresh person). Policy:

    * strong match to the SAME id → no-op (a re-enroll refresh; the store folds the
      centroid);
    * caller pinned no id and a strong match exists → return the matched id (reuse
      the record instead of duplicating it);
    * caller PINNED an id ("guest-2") that strongly matches a DIFFERENT existing id
      → keep the requested id but log a WARNING — two distinct competition guests
      must never be auto-merged (that would silently drop one), so this only
      surfaces the collision (it also feeds the ``HRI_DUP_*`` AuditIdentities step).

    Best-effort: disabled with ``HRI_ENROLL_DEDUP=0``, and any store error or empty
    store returns the requested id unchanged.
    """
    if os.getenv("HRI_ENROLL_DEDUP", "1").lower() not in ("1", "true", "yes"):
        return requested_id
    if ctx.people is None or ctx.people.count() == 0:
        return requested_id
    min_score = float(os.getenv("HRI_ENROLL_DEDUP_MIN_SCORE", "0.7"))
    try:
        rec = ctx.people.recognize_fused(
            face_emb, app_emb, face_confidence=det_score, min_score=min_score
        )
    except Exception as exc:
        print(f"[identity] enroll dedup check failed ({exc})")
        return requested_id
    if rec is None or rec.id == requested_id:
        return requested_id
    sim = rec.similarity if rec.similarity is not None else float("nan")
    if pinned:
        print(
            f"[identity] WARNING enroll {requested_id} strongly matches existing "
            f"{rec.id} (sim {sim:.2f}) — possible duplicate; keeping {requested_id}"
        )
        return requested_id
    print(f"[identity] enroll: reusing existing id {rec.id} (sim {sim:.2f})")
    return rec.id


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
    person_id = _dedup_person_id(
        ctx,
        face.embedding if face is not None else None,
        app_emb,
        face.det_score if face is not None else 0.0,
        person_id,
    )
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
    When *attributes* is empty, a TEXT appearance caption of the best frame
    (:func:`caption_person_appearance`) is stored in its place, so every
    enrollment carries a spoken-introduction-ready description.

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

    person_id = _dedup_person_id(
        ctx,
        avg_face,
        avg_app,
        best_face.det_score if best_face is not None else 0.0,
        person_id,
    )

    # TEXT appearance: when the caller didn't supply one, caption the best face
    # frame's person crop (head-level — it carries hair/glasses/face, unlike the
    # tilted-down attire frames), falling back to the first app frame's largest
    # body. Stored as the record's attributes, so the introduction step can
    # describe every enrolled person in detail. Best-effort — a caption failure
    # enrolls without attributes exactly as before.
    if not attributes:
        cap_img, cap_box = None, None
        if best_face is not None and best_face_img is not None:
            cap_img = best_face_img
            cap_box = _person_bbox_for_face(
                best_face,
                _detect_persons_xyxy(ctx, best_face_img),
                best_face_img.width, best_face_img.height,
            )
        elif app_imgs:
            cap_img = app_imgs[0]
            cap_box = _largest_person_box(ctx, app_imgs[0])
        if cap_img is not None:
            attributes = caption_person_appearance(ctx, cap_img, cap_box) or ""

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


def _best_injective_assignment(
    wanted: list[str], feasible: dict[str, dict[int, float]], n_boxes: int
) -> dict[str, int]:
    """Assign at most one box to each wanted id (and each box to at most one id)
    so that total similarity is maximized.

    Brute force over the tiny id/box sets (Receptionist has host + <=2 guests);
    falls back to a plain greedy assignment above a safety size so it can never
    blow up on pathological input. ``feasible[id]`` maps box-index -> score for the
    boxes that cleared ``min_score`` for that id.
    """
    if len(wanted) > 6 or n_boxes > 6:
        # Unreachable in HRI; safe greedy fallback for pathological inputs.
        chosen: dict[str, int] = {}
        used: set[int] = set()
        for _s, rid, k in sorted(
            ((s, rid, k) for rid in wanted for k, s in feasible[rid].items()),
            reverse=True,
        ):
            if rid in chosen or k in used:
                continue
            chosen[rid] = k
            used.add(k)
        return chosen

    best: dict[str, int] = {}
    best_total = -1.0

    def rec(i: int, used: frozenset, cur: dict[str, int], total: float) -> None:
        nonlocal best, best_total
        if i == len(wanted):
            if total > best_total:
                best_total, best = total, dict(cur)
            return
        rid = wanted[i]
        rec(i + 1, used, cur, total)  # leave this id unassigned
        for k, s in feasible[rid].items():
            if k in used:
                continue
            cur[rid] = k
            rec(i + 1, used | {k}, cur, total + s)
            del cur[rid]

    rec(0, frozenset(), {}, 0.0)
    return best


def _assign_boxes(
    boxscores: list[tuple[BBox, dict[str, float]]],
    wanted: list[str],
    min_score: float,
    box_margin: float,
) -> dict[str, tuple[BBox, float]]:
    """Globally-optimal, peakiness-gated id<->box assignment within one frame.

    *boxscores* pairs each person box with its FULL per-id fused scores (every
    enrolled id, so an un-wanted host acts as a distractor). Returns
    ``{id: (box, sim)}`` for the confident assignments only:

    1. Over the *wanted* ids, pick the injective assignment maximizing total
       similarity (:func:`_best_injective_assignment`). This removes the greedy
       "steal" where a higher *wrong* pairing claims another guest's box: the
       global optimum keeps each guest on the box that best explains it.
    2. Confidence gate: an assigned ``(id, box)`` survives only if *id*'s score on
       that box beats, by *box_margin*, the best score of any OTHER enrolled id
       that would *prefer* this box (scores higher here than at its own
       assignment). A genuinely ambiguous box (near-tied ids with no better home
       elsewhere) is dropped rather than mislabeled.
    """
    boxes = [box for box, _s in boxscores]
    feasible: dict[str, dict[int, float]] = {}
    for rid in wanted:
        feasible[rid] = {
            k: sc[rid]
            for k, (_b, sc) in enumerate(boxscores)
            if sc.get(rid) is not None and sc[rid] >= min_score
        }
    assign = _best_injective_assignment(wanted, feasible, len(boxes))

    result: dict[str, tuple[BBox, float]] = {}
    for rid, k in assign.items():
        sim = feasible[rid][k]
        runner_up = 0.0
        for other, s_here in boxscores[k][1].items():
            if other == rid:
                continue
            ok = assign.get(other)
            other_home = feasible.get(other, {}).get(ok, 0.0) if ok is not None else 0.0
            if s_here > other_home:  # `other` would rather have this box
                runner_up = max(runner_up, s_here)
        if sim - runner_up >= box_margin:
            result[rid] = (boxes[k], sim)
    return result


def locate_people(
    ctx: TaskContext, frames: list[Image.Image], person_ids: list[str]
) -> dict[str, tuple[int, BBox]]:
    """Find enrolled people across one or more frames: {id: (frame_index, bbox)}.

    Face FIRST, per frame: each trusted detected face (:func:`_face_trusted`) is
    scored against the whole store with ``fused_scores`` (face + attire of the
    same person crop, all ids), then boxes are matched to identities by a
    GLOBALLY-OPTIMAL, peakiness-gated assignment (:func:`_assign_boxes`) rather
    than greedily — so a higher *wrong* pairing can no longer steal a guest's box,
    and a genuinely ambiguous box is left unlabeled instead of mis-labeled. The
    per-frame assignments are then voted across the sweep (``HRI_LOCATE_MIN_VOTES``
    frames must agree before a match is trusted). Only ids still missing get an
    appearance-only fallback over the remaining pose boxes (covers a guest turned
    away in all views), which keeps the same peakiness so it can't mislabel
    either. The returned frame index ties each match to the snapshot it came from,
    so the caller can lift the bbox against that frame's capture-time geometry.
    Partial results on any failure; an unmatched guest is simply absent (the
    caller falls back to their stored seat rather than risk a wrong label).
    """
    if ctx.people is None or ctx.people.count() == 0:
        return {}
    wanted = list(dict.fromkeys(person_ids))
    wanted_set = set(wanted)
    min_score = ctx.people.fused_min_score()
    box_margin = float(os.getenv("HRI_LOCATE_BOX_MARGIN", "0.06"))
    min_votes = max(1, int(os.getenv("HRI_LOCATE_MIN_VOTES", "1")))

    # Per frame: the pose boxes (reused by the fallback) and, for each trusted
    # face, its person box paired with the full per-id fused scores.
    persons_by_frame: list[list[BBox]] = []
    per_frame_boxscores: list[list[tuple[BBox, dict[str, float]]]] = []
    for img in frames:
        faces = _detect_faces(ctx, img)
        persons_xyxy = _detect_persons_xyxy(ctx, img)
        persons_by_frame.append(persons_xyxy)
        boxscores: list[tuple[BBox, dict[str, float]]] = []
        for face in faces:
            if not _face_trusted(face, hard=True):
                continue
            person_box = _person_bbox_for_face(face, persons_xyxy, img.width, img.height)
            app_emb = _appearance_embedding(ctx, img, person_box)
            scores = ctx.people.fused_scores(
                face.embedding, app_emb, face_confidence=face.det_score
            )
            if scores:
                boxscores.append((person_box, scores))
        per_frame_boxscores.append(boxscores)

    # Optimal per-frame assignment, then vote across the sweep frames.
    by_id: dict[str, list[tuple[float, int, BBox]]] = {}
    for fi, boxscores in enumerate(per_frame_boxscores):
        for rid, (box, sim) in _assign_boxes(
            boxscores, wanted, min_score, box_margin
        ).items():
            by_id.setdefault(rid, []).append((sim, fi, box))

    located: dict[str, tuple[int, BBox]] = {}
    used: set[tuple[int, BBox]] = set()  # (frame, box) claimed by a match
    ranked: list[tuple[float, str, list[tuple[float, int, BBox]]]] = []
    for rid, sightings in by_id.items():
        if len(sightings) < min_votes:
            continue
        sightings.sort(reverse=True)  # best similarity first
        ranked.append((sightings[0][0], rid, sightings))
    for _best, rid, sightings in sorted(ranked, key=lambda r: r[0], reverse=True):
        for sim, fi, box in sightings:
            if rid in located or (fi, box) in used:
                continue
            located[rid] = (fi, box)
            used.add((fi, box))
            print(f"[identity] located {rid} by face (frame {fi}, similarity {sim:.2f})")
            break

    # Appearance-only fallback for ids still missing (face turned away in every
    # frame) — scan the unclaimed pose boxes, keeping the per-box peakiness (best
    # missing id must beat the best other enrolled distractor) so it can't
    # mislabel either.
    missing = [rid for rid in wanted if rid not in located]
    if missing:
        fb: list[tuple[float, str, int, BBox]] = []
        for fi, persons_xyxy in enumerate(persons_by_frame):
            for box in persons_xyxy:
                if (fi, box) in used:
                    continue
                app_emb = _appearance_embedding(ctx, frames[fi], box)
                if app_emb is None:
                    continue
                scores = ctx.people.fused_scores(None, app_emb)
                ranked_missing = sorted(
                    (
                        (scores[rid], rid)
                        for rid in missing
                        if rid in scores and scores[rid] >= min_score
                    ),
                    reverse=True,
                )
                if not ranked_missing:
                    continue
                best_sim, best_id = ranked_missing[0]
                other_best = max(
                    (s for rid2, s in scores.items() if rid2 != best_id), default=0.0
                )
                if best_sim - other_best < box_margin:
                    continue  # ambiguous vs another id on this box
                fb.append((best_sim, best_id, fi, box))
        for sim, rid, fi, box in sorted(fb, key=lambda c: c[0], reverse=True):
            if rid in located or (fi, box) in used:
                continue
            located[rid] = (fi, box)
            used.add((fi, box))
            print(f"[identity] located {rid} by appearance only (frame {fi}, sim {sim:.2f})")

    return {rid: v for rid, v in located.items() if rid in wanted_set}


def refresh_person_attire(ctx: TaskContext, person_id: str, img: Image.Image) -> bool:
    """Re-embed *person_id*'s attire from *img* and refresh the stored vector.

    The follow loop matches attire against the enrollment-time vector, but the
    host is enrolled from ONE frame while SEATED (``OfferSeat``) and then followed
    while STANDING — a posture/crop mismatch that drags every follow-tick attire
    similarity down. Called right before the follow starts (the robot is facing
    the host, who is getting up to lead): locate the person face-first in the
    current frame (:func:`locate_people`, so the refreshed box is
    identity-verified), embed its attire crop, and re-enroll with only the
    appearance vector updated — latest-wins is the store's own contract for
    attire, and re-passing the stored face vector folds a centroid of itself
    (unchanged). Gated by ``HRI_FOLLOW_REFRESH_ATTIRE``; best-effort — any miss
    or failure logs and returns False, leaving the stored vector as it was.
    """
    if os.getenv("HRI_FOLLOW_REFRESH_ATTIRE", "1").lower() not in ("1", "true", "yes"):
        return False
    if ctx.people is None or ctx.people.count() == 0:
        return False
    try:
        found = locate_people(ctx, [img], [person_id]).get(person_id)
        if found is None:
            print(f"[identity] attire refresh: {person_id} not located; skipping")
            return False
        _fi, box = found
        app_emb = _appearance_embedding(ctx, img, box)
        if app_emb is None:
            return False
        rec = ctx.people.get(person_id)
        if rec is None:
            return False
        ctx.people.enroll(
            rec.name or "",
            rec.drink or "",
            list(rec.embedding) if rec.embedding else [0.0] * 512,
            person_id=person_id,
            app_embedding=app_emb,
        )
        print(f"[identity] refreshed {person_id}'s attire from the current frame")
        return True
    except Exception as exc:
        print(f"[identity] attire refresh for {person_id} failed ({exc})")
        return False


def _persons_to_candidates(
    persons: list[PersonPose],
) -> list[tuple[BBox, float | None]]:
    """Pair each pose with its closeness proxy, capped by config (pure, no I/O).

    Builds the follow candidate list from a pose result already fetched by the
    unified :func:`_detect_pose_and_face` call, so the selection logic never
    re-detects.
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


def _follow_debug_enabled() -> bool:
    return os.getenv("HRI_FOLLOW_TRACK_DEBUG", "0").lower() in ("1", "true", "yes")


def _log_no_qualifier(
    modality: str, best_sim: float | None, min_score: float, reacquiring: bool
) -> None:
    """One HRI_FOLLOW_TRACK_DEBUG line when a pass scored candidates but none
    qualified — reports the best host similarity seen against the active floor,
    which is exactly what's needed to tune the floors to a real arena."""
    if best_sim is None or not _follow_debug_enabled():
        return
    print(
        f"[identity] follow: no {modality} qualifier "
        f"(best host sim {best_sim:.2f}, floor {min_score:.2f}"
        f"{', reacquiring' if reacquiring else ''})"
    )


def _match_floor(ctx: TaskContext, *, reacquiring: bool) -> float:
    """Fused-match floor for a follow tick: the perception-wide
    ``APPEARANCE_MATCH_THRESHOLD`` while maintaining a lock, raised to
    ``HRI_FOLLOW_REACQUIRE_MIN_SCORE`` when re-acquiring — a fresh lock is riskier
    than holding one, so a look-alike must clear a higher bar to grab the host."""
    base = ctx.people.fused_min_score()
    if not reacquiring:
        return base
    return max(base, float(os.getenv("HRI_FOLLOW_REACQUIRE_MIN_SCORE", "0.55")))


def _visible_margin(*, reacquiring: bool) -> float:
    """How much the chosen host box must beat the runner-up VISIBLE candidate's
    host-similarity by — stricter while re-acquiring (``HRI_FOLLOW_REACQUIRE_MARGIN``)
    than while maintaining a lock (``HRI_FOLLOW_VISIBLE_MARGIN``)."""
    key, default = (
        ("HRI_FOLLOW_REACQUIRE_MARGIN", "0.10")
        if reacquiring
        else ("HRI_FOLLOW_VISIBLE_MARGIN", "0.06")
    )
    return float(os.getenv(key, default))


def _pick_closest_qualifier(
    qualifying: list[tuple[float, BBox, float | None]],
    person_id: str,
    modality: str,
    *,
    reacquiring: bool = False,
) -> BBox | None:
    """Among identity-qualifying boxes, the best by similarity PLUS a proximity
    bonus (``HRI_FOLLOW_PROXIMITY_WEIGHT`` × the box's pose scale relative to the
    nearest qualifier). The proximity term only breaks ties between genuine
    matches — every box here already cleared the identity gate, so a near
    bystander can never win. Boxes with no measurable scale carry no bonus.

    Then a VISIBLE-candidate peakiness test: the winner must beat the runner-up
    *visible* candidate's raw host-similarity by :func:`_visible_margin`, else the
    tick is ambiguous (two people about equally like the host — e.g. a look-alike
    standing where the host just was) and we return ``None`` so ``follow_person``
    coasts on its motion prediction rather than risk locking the wrong person.
    ``None`` when nothing qualifies or the top match is ambiguous.
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
    margin = _visible_margin(reacquiring=reacquiring)
    if margin > 0 and len(qualifying) >= 2:
        runner_up = max((s for s, b, _sc in qualifying if b != box), default=None)
        if runner_up is not None and sim - runner_up < margin:
            print(
                f"[identity] follow: ambiguous {modality} "
                f"(top {sim:.2f} vs runner-up {runner_up:.2f}); coasting"
            )
            return None
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
    *,
    reacquiring: bool = False,
) -> BBox | None:
    """Score pre-detected *faces* against the store and pick *person_id*'s box.

    Pure (no I/O of its own): the faces arrive alongside the poses from the
    single :func:`_detect_pose_and_face` round-trip. Faces are far more
    discriminative than clothing, so a face match holds the right person even in
    a similarly-dressed crowd. Each detected face (that clears :func:`_face_trusted`
    — a low-confidence detection, or a tiny one when re-acquiring, is not trusted
    to establish identity) is scored (face only) against every enrolled person; a
    box qualifies only when *person_id* clears the (re-acquire-aware)
    :func:`_match_floor` AND out-scores every OTHER enrolled person by
    ``HRI_FOLLOW_FACE_MARGIN``, and the winner must additionally clear the
    visible-candidate peakiness test in :func:`_pick_closest_qualifier`. ``None``
    when no face in view is the target (turned away / too far / occluded, or
    *person_id* enrolled without a face) — the caller then falls back to attire.
    """
    if not faces:
        return None
    min_score = _match_floor(ctx, reacquiring=reacquiring)
    margin = float(os.getenv("HRI_FOLLOW_FACE_MARGIN", "0.05"))
    qualifying: list[tuple[float, BBox, float | None]] = []
    best_seen: float | None = None
    for face in faces:
        if not _face_trusted(face, hard=reacquiring):
            continue
        box, scale = _cand_for_face(face, cands, img.width, img.height)
        scores = ctx.people.fused_scores(face.embedding)  # face-only path
        host_sim = scores.get(person_id)
        if host_sim is None:
            continue
        best_seen = host_sim if best_seen is None else max(best_seen, host_sim)
        if host_sim < min_score:
            continue
        best_other = max((s for rid, s in scores.items() if rid != person_id), default=0.0)
        if host_sim - best_other < margin:
            continue
        qualifying.append((host_sim, box, scale))
    if not qualifying:
        _log_no_qualifier("face", best_seen, min_score, reacquiring)
    return _pick_closest_qualifier(qualifying, person_id, "face", reacquiring=reacquiring)


def _score_attire_candidates(
    ctx: TaskContext,
    cands: list[tuple[BBox, float | None]],
    embeds: list[list[float] | None],
    person_id: str,
    *,
    reacquiring: bool = False,
) -> BBox | None:
    """Pick *person_id*'s box from candidate attire *embeds* aligned with *cands*.

    The local (no-I/O) half of :func:`_select_by_attire`, split out so the
    parallel follow path can fan the per-candidate OSNet embeds across threads
    and hand the results in. ``embeds[i]`` is the vector for ``cands[i]`` (or
    ``None`` if that embed failed). Scoring (``fused_scores``) runs here on the
    calling thread, so the local Chroma store is never touched concurrently. Uses
    the re-acquire-aware :func:`_match_floor` and visible-candidate peakiness
    (:func:`_pick_closest_qualifier`); an optional ``HRI_RECOG_MIN_CROP_PX`` gate
    skips a too-small crop before trusting its embedding.
    """
    min_score = _match_floor(ctx, reacquiring=reacquiring)
    margin = float(os.getenv("HRI_FOLLOW_APPEARANCE_MARGIN", "0.05"))
    min_crop = float(os.getenv("HRI_RECOG_MIN_CROP_PX", "0"))
    qualifying: list[tuple[float, BBox, float | None]] = []
    best_seen: float | None = None
    for (box, scale), app_emb in zip(cands, embeds):
        if app_emb is None:
            continue
        if min_crop > 0 and (box[2] - box[0]) * (box[3] - box[1]) < min_crop:
            continue
        scores = ctx.people.fused_scores(None, app_emb)
        host_sim = scores.get(person_id)
        if host_sim is None:
            continue
        best_seen = host_sim if best_seen is None else max(best_seen, host_sim)
        if host_sim < min_score:
            continue
        best_other = max((s for rid, s in scores.items() if rid != person_id), default=0.0)
        if host_sim - best_other < margin:
            continue
        qualifying.append((host_sim, box, scale))
    if not qualifying:
        _log_no_qualifier("attire", best_seen, min_score, reacquiring)
    return _pick_closest_qualifier(qualifying, person_id, "appearance", reacquiring=reacquiring)


def _select_by_attire(
    ctx: TaskContext,
    img: Image.Image,
    cands: list[tuple[BBox, float | None]],
    person_id: str,
    *,
    reacquiring: bool = False,
) -> BBox | None:
    """Attire fallback: the closest box whose OSNet attire vector is *person_id*'s.

    Used when no face was recognized this tick. Embeds each candidate crop and
    scores it against every enrolled person (appearance-only fusion). A box
    qualifies only when *person_id* clears the (re-acquire-aware) match floor AND
    beats the best OTHER enrolled person by ``HRI_FOLLOW_APPEARANCE_MARGIN`` — at a
    party people dress alike, so a bare top-1 lead flips onto a guest on embedding
    noise; the margin demands the host clearly win. ``None`` when nobody qualifies.
    """
    embeds = [_appearance_embedding(ctx, img, box) for box, _scale in cands]
    return _score_attire_candidates(ctx, cands, embeds, person_id, reacquiring=reacquiring)


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
    *,
    reacquiring: bool = False,
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
    box = _select_by_attire(ctx, img, gated, person_id, reacquiring=reacquiring)
    if box is None and len(gated) != len(cands):
        box = _select_by_attire(ctx, img, cands, person_id, reacquiring=reacquiring)
    return box


def _embed_and_score_attire_parallel(
    ex: ThreadPoolExecutor,
    ctx: TaskContext,
    img: Image.Image,
    cands: list[tuple[BBox, float | None]],
    person_id: str,
    *,
    reacquiring: bool = False,
) -> BBox | None:
    """Embed *cands*' crops concurrently on *ex* and score them (parallel attire).

    ``ex.map`` preserves order, so ``embeds[i]`` aligns with ``cands[i]``;
    scoring (:meth:`fused_scores`) stays on the calling thread so the local
    Chroma store is never touched concurrently. ``None`` for an empty *cands*.
    """
    if not cands:
        return None
    embeds = list(ex.map(lambda bs: _appearance_embedding(ctx, img, bs[0]), cands))
    return _score_attire_candidates(ctx, cands, embeds, person_id, reacquiring=reacquiring)


def _attire_pass_parallel(
    ctx: TaskContext,
    img: Image.Image,
    cands: list[tuple[BBox, float | None]],
    person_id: str,
    hint_box: BBox | None,
    *,
    reacquiring: bool = False,
) -> BBox | None:
    """Parallel sibling of :func:`_attire_pass` (HRI_FOLLOW_PARALLEL=1).

    Gates the per-candidate OSNet embeds to *hint_box*, fans them across a thread
    pool — the costly part in a crowd is one OSNet call per candidate box — then
    widens to the full set on a miss, exactly as the serial pass does. Only the
    attire embeds parallelise: pose + face now arrive together from the single
    :func:`_detect_pose_and_face` request, so there is no longer a second
    detection round-trip to overlap. Identity scoring (:meth:`fused_scores`,
    which reads the local Chroma store) stays on the calling thread, so no Chroma
    access is concurrent. Returns exactly the box :func:`_attire_pass` would.
    """
    workers = max(2, int(os.getenv("HRI_FOLLOW_PARALLEL_WORKERS", "8")))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        gated = _gate_candidates(cands, hint_box, radius_scale=_gate_radius_scale())
        box = _embed_and_score_attire_parallel(
            ex, ctx, img, gated, person_id, reacquiring=reacquiring
        )
        if box is None and len(gated) != len(cands):
            box = _embed_and_score_attire_parallel(
                ex, ctx, img, cands, person_id, reacquiring=reacquiring
            )
        return box


def _box_shows_nonhost_face(
    ctx: TaskContext,
    faces: list[FaceEmbedding],
    box: BBox,
    person_id: str,
    *,
    min_det: float,
    face_sim_floor: float,
    nonhost_margin: float,
) -> bool:
    """True when *box* contains a confidently-detected face that is clearly NOT
    the host's.

    While the host is being followed their back is to the camera (no face in their
    box), but bystanders usually face the robot — so a person box that shows a
    good frontal face whose host similarity is well below the face floor is
    provably someone else and must not be locked onto via attire alone. Only
    excludes on a *confident negative*: the contained face must clear *min_det*
    AND the hard-decision size gate (a tiny blurry face gives unreliable sims and
    must never veto a candidate), and its host similarity must fall below
    ``face_sim_floor - nonhost_margin``. Same-person face sims in bad conditions
    land around 0.4–0.6 while different-person sims sit around 0.1–0.3, so the
    default margin (0.25 → exclude below ~0.35) rules out real bystanders without
    ruling out a host whose one-frame enrollment scores modestly. Never excludes
    when the host was enrolled without a face (then a face gives no signal about
    them).
    """
    min_area = float(os.getenv("HRI_RECOG_MIN_FACE_AREA_PX", "2500"))
    for face in faces:
        if face.det_score < min_det or face.area() < min_area:
            continue
        fx = (face.bbox_xyxy[0] + face.bbox_xyxy[2]) / 2
        fy = (face.bbox_xyxy[1] + face.bbox_xyxy[3]) / 2
        if not (box[0] <= fx <= box[2] and box[1] <= fy <= box[3]):
            continue
        host_sim = ctx.people.fused_scores(face.embedding).get(person_id)
        if host_sim is None:  # host has no stored face -> face can't judge
            continue
        if host_sim < face_sim_floor - nonhost_margin:
            return True
    return False


def _drop_nonhost_face_candidates(
    ctx: TaskContext,
    faces: list[FaceEmbedding],
    cands: list[tuple[BBox, float | None]],
    person_id: str,
) -> list[tuple[BBox, float | None]]:
    """Drop attire candidates whose box shows a clearly non-host face.

    The key lever for "the host just left frame": a similarly-dressed bystander
    who took the host's spot but is FACING the robot is excluded by their own
    (non-host) face, even though their clothing matches. The back-turned host's
    box has no face inside, so it is never dropped. No-op when the feature is
    disabled (``HRI_FOLLOW_EXCLUDE_NONHOST_FACE=0``) or no faces were detected.
    """
    if not faces or os.getenv("HRI_FOLLOW_EXCLUDE_NONHOST_FACE", "1").lower() not in (
        "1", "true", "yes",
    ):
        return cands
    min_det = float(os.getenv("HRI_RECOG_MIN_DET_SCORE", "0.5"))
    nonhost_margin = float(os.getenv("HRI_FOLLOW_NONHOST_FACE_MARGIN", "0.25"))
    face_sim_floor = 1.0 - ctx.people.face_match_max_distance()
    kept = [
        (box, scale)
        for (box, scale) in cands
        if not _box_shows_nonhost_face(
            ctx, faces, box, person_id,
            min_det=min_det, face_sim_floor=face_sim_floor, nonhost_margin=nonhost_margin,
        )
    ]
    if len(kept) != len(cands):
        print(
            f"[identity] follow: excluded {len(cands) - len(kept)} candidate(s) "
            "showing a non-host face"
        )
    return kept


def _detect_pose_and_face(
    ctx: TaskContext, img: Image.Image, *, run_face: bool
) -> tuple[list[PersonPose], list[FaceEmbedding]]:
    """Pose (always) + face (only when *run_face*) for *img* in ONE round-trip.

    The unified ``/image/process`` endpoint uploads and decodes the frame once
    and runs both models server-side, so a follow tick pays a single JPEG encode
    + transfer instead of the two it cost when pose and face were separate
    sub-client calls (:meth:`estimate_poses` then :meth:`faces`). That collapses
    one whole-frame round-trip per tick — a guaranteed win even when the server
    serializes inference on a single GPU, since the saved cost is the duplicate
    encode/transfer/decode, not the model time. Throttled ticks (``run_face`` is
    False) request pose only, so the face model is never run server-side then.
    Best-effort: a failed request degrades to ``([], [])`` and the tick coasts on
    the follow loop's motion prediction.
    """
    try:
        res = ctx.walkieAI.image.process(img, pose=True, face=run_face)
    except Exception as exc:
        print(f"[identity] follow pose/face detect failed ({exc})")
        return [], []
    return (res.pose or []), (res.face or [])


def select_person_to_follow(
    ctx: TaskContext,
    snap,
    person_id: str,
    *,
    hint_box: BBox | None = None,
    run_face: bool = True,
    reacquiring: bool = False,
) -> BBox | None:
    """:func:`skills.follow_person` selector: the person box matching *person_id*,
    FACE first then ATTIRE, biased toward the nearest qualifying candidate.

    Per tick: ONE :func:`_detect_pose_and_face` request returns the poses (each
    box paired with a pose-keypoint closeness proxy, :func:`_person_scale_px`)
    and — when *run_face* — the faces, from a single uploaded frame. A FACE pass
    runs first (:func:`_match_by_face`, only when *run_face*) — faces are far more
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
    (usually fruitless, while following) face inference: a throttled tick asks
    the unified call for pose only. *reacquiring* (set by :class:`FollowSelector`
    when it holds no confirmed lock) tightens the gates for establishing a *new*
    lock — a higher :func:`_match_floor` and margin — so a look-alike who wanders
    into the host's last spot can't grab the lock; steady-state maintenance keeps
    the permissive floor. All three default to the pre-tracking behaviour (no hint,
    face every tick, not re-acquiring), so direct callers and the enrollment paths
    are unaffected — the stateful :class:`FollowSelector` supplies them during a
    follow.

    With ``HRI_FOLLOW_PARALLEL=1`` the attire-fallback embeds are fanned across a
    thread pool (:func:`_attire_pass_parallel`); the result is identical, only
    the latency differs. (Pose + face no longer need overlapping — the unified
    ``process`` call already returns both from one round-trip.)

    *person_id* must already be enrolled (the host is, at ``OfferSeat``); the
    face pass needs a face embedding, the attire fallback an attire one.
    """
    if snap is None or ctx.people is None or ctx.people.count() == 0:
        return None
    img = snap.img
    persons, faces = _detect_pose_and_face(ctx, img, run_face=run_face)
    cands = _persons_to_candidates(persons)
    if not cands:
        return None
    if run_face:
        box = _match_by_face(ctx, img, faces, cands, person_id, reacquiring=reacquiring)
        if box is not None:
            return box  # face match: never pay for the attire embeds
        # No positive face match this tick -> attire fallback, but first drop any
        # candidate whose box shows a clearly non-host face (a face-shown bystander
        # standing where the back-turned host was). Faces are only available when
        # they were fetched this tick (run_face).
        cands = _drop_nonhost_face_candidates(ctx, faces, cands, person_id)
        if not cands:
            return None
    if _follow_parallel_enabled():
        return _attire_pass_parallel(ctx, img, cands, person_id, hint_box, reacquiring=reacquiring)
    return _attire_pass(ctx, img, cands, person_id, hint_box, reacquiring=reacquiring)


class FollowSelector:
    """Stateful :func:`skills.follow_person` selector: tracks the host tick-to-tick.

    The follow loop calls ``select(ctx, snap)`` as a plain 2-arg callable; this
    object satisfies that while carrying the state that makes a follow both cheap
    and hard to hijack:

    * ``locked`` / ``ever_locked`` — whether a host lock is currently held, and
      whether one was ever held this follow. The INITIAL acquisition (never
      locked yet — the host is front-and-center right after being asked where the
      bag goes) uses the permissive base gates and commits on the FIRST
      qualifying tick, exactly like the pre-hysteresis behaviour: every
      unconfirmed ``None`` makes ``follow_person`` rotate-search, so being strict
      at start just spins the robot away from a host whose stored vectors (one
      seated enrollment frame) score modestly. Only RE-acquiring a LOST lock —
      the look-alike-hijack case — is strict: ``reacquiring=True`` (higher
      :func:`_match_floor` + margin, hard face gates) and
      ``HRI_FOLLOW_LOCK_CONFIRM_TICKS`` consecutive confirming ticks before the
      lock re-commits, returning ``None`` meanwhile so a one-frame fluke on a
      look-alike never commits.
    * ``last_box`` — the last matched box, used as the attire-gate hint (and set
      during confirmation so the confirming ticks tighten onto the same person).
    * once locked it throttles the FACE pass to every ``HRI_FOLLOW_FACE_EVERY_N``
      ticks (while following, the host's back is to the camera so the face pass
      otherwise burns a full-frame inference for ~no matches), gates the attire
      embed to ``last_box`` (full scan on a miss), and tolerates up to
      ``HRI_FOLLOW_LOCK_MISS_TOLERANCE`` consecutive no-match ticks (coasting via
      ``follow_person``'s predictor) before dropping the lock and re-acquiring
      under the strict gate again.

    Returning ``None`` on an unconfirmed / ambiguous / lost tick is safe:
    ``follow_person`` coasts on its motion prediction and rotate-searches rather
    than driving toward the wrong person. One instance per follow; not thread-safe
    (the loop is single-threaded).
    """

    def __init__(self, person_id: str) -> None:
        self.person_id = person_id
        self.last_box: BBox | None = None
        self.tick = 0
        self.locked = False
        self.ever_locked = False  # strict re-acquire applies only after a LOST lock
        self.confirm = 0  # consecutive confirming ticks while re-acquiring
        self.miss = 0  # consecutive no-match ticks while locked

    def _run_face_this_tick(self) -> bool:
        if not self.locked:
            return True  # acquiring / confirming: use every modality
        every_n = max(1, int(os.getenv("HRI_FOLLOW_FACE_EVERY_N", "5")))
        return self.tick % every_n == 0

    def __call__(self, ctx: TaskContext, snap) -> BBox | None:
        box = select_person_to_follow(
            ctx,
            snap,
            self.person_id,
            hint_box=self.last_box,
            run_face=self._run_face_this_tick(),
            reacquiring=self.ever_locked and not self.locked,
        )
        self.tick += 1
        return self._update(box)

    def _update(self, box: BBox | None) -> BBox | None:
        """Apply the lock hysteresis to this tick's raw match and return what the
        follow loop should drive toward (``None`` = coast)."""
        miss_tol = max(0, int(os.getenv("HRI_FOLLOW_LOCK_MISS_TOLERANCE", "2")))
        if not self.locked:
            if box is None:
                self.confirm = 0
                self.last_box = None
                return None
            # Initial acquisition commits on the first qualifying tick (base
            # gates); only a RE-acquire after a lost lock needs K confirming
            # ticks — every unconfirmed None makes follow_person rotate-search,
            # which at start would spin the robot away from the host.
            confirm_k = (
                max(1, int(os.getenv("HRI_FOLLOW_LOCK_CONFIRM_TICKS", "2")))
                if self.ever_locked
                else 1
            )
            self.confirm += 1
            self.last_box = box  # hint the next confirming tick onto this person
            if self.confirm >= confirm_k:
                self.locked = True
                self.ever_locked = True
                self.miss = 0
                return box
            return None  # still confirming -> coast
        # locked: hold through brief misses, drop only past the tolerance
        if box is not None:
            self.miss = 0
            self.last_box = box
            return box
        self.miss += 1
        if self.miss > miss_tol:
            self.locked = False
            self.confirm = 0
            self.last_box = None
        return None


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
