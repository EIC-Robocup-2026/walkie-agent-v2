"""Guest identity: enroll faces/appearance into ctx.people, find them again.

Enrollment happens at the door (the guest stands alone in front of the
camera — the best face shot of the whole run); recognition happens in the
living room at introduction time, where guests may have switched seats.

Everything here is best-effort: any AI-client failure logs and degrades to a
partial result — never raises (a missed enrollment costs points, a crashed
task costs the run).
"""

from __future__ import annotations

from PIL import Image

from client.face_recognition import FaceEmbedding
from tasks.base import TaskContext

from .skills import BBox, cxcywh_to_xyxy


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


def _detect_persons_xyxy(ctx: TaskContext, img: Image.Image) -> list[BBox]:
    try:
        return [cxcywh_to_xyxy(p.bbox) for p in ctx.walkieAI.pose_estimation.estimate(img)]
    except Exception as exc:
        print(f"[identity] pose estimate failed ({exc})")
        return []


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
    ctx: TaskContext, img: Image.Image, person_ids: list[str]
) -> dict[str, BBox]:
    """Find enrolled people in one frame: {person_id: person bbox_xyxy}.

    Every detected face is scored against the whole store with
    ``recognize_fused`` (face + attire of the same person crop), then faces
    are greedily assigned to identities by similarity — so the host (also
    enrolled) acts as a distractor a guest match must beat. Ids still missing
    get an appearance-only pass over the remaining pose-detected persons
    (covers a guest facing away). Partial results on any failure.
    """
    if ctx.people is None or ctx.people.count() == 0:
        return {}
    wanted = set(person_ids)
    faces = _detect_faces(ctx, img)
    persons_xyxy = _detect_persons_xyxy(ctx, img)

    min_score = ctx.people.fused_min_score()
    # Score every face against the store, remembering its person box.
    candidates: list[tuple[float, str, BBox]] = []  # (similarity, id, person box)
    used_boxes: list[BBox] = []
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
            candidates.append((rec.similarity, rec.id, person_box))

    located: dict[str, BBox] = {}
    for sim, rid, box in sorted(candidates, key=lambda c: c[0], reverse=True):
        if rid in located or any(b == box for b in used_boxes):
            continue
        located[rid] = box
        used_boxes.append(box)
        print(f"[identity] located {rid} (similarity {sim:.2f})")

    # Appearance-only pass for ids still missing (face turned away).
    missing = wanted - set(located)
    if missing:
        free_boxes = [b for b in persons_xyxy if not any(b == u for u in used_boxes)]
        for box in free_boxes:
            app_emb = _appearance_embedding(ctx, img, box)
            if app_emb is None:
                continue
            rec = ctx.people.recognize_fused(None, app_emb, min_score=min_score)
            if rec is not None and rec.id in missing:
                located[rec.id] = box
                used_boxes.append(box)
                missing.discard(rec.id)
                print(f"[identity] located {rec.id} by appearance only")
            if not missing:
                break

    return {rid: box for rid, box in located.items() if rid in wanted}
