"""Targeted object search for the Vision agent, mirroring the arm/grasp detector.

The robot used to "not see" an object the arm could grasp: the vision tool called
``walkieAI.image.detect(img)`` with no prompts and no confidence filter, so the
open-vocab detector (YOLOE) ran on its default vocabulary and reported a brand-named
item like *coke* as "bottle"/"cup" — or nothing — while the grasp system found it
fine. The grasp pipeline (``tasks/skills/grasp.py``) wins because it (a) expands the
target into GENERIC visual descriptors the detector can box ("coke" -> "can", "red
can"), (b) CLIP-reranks the resulting boxes against the SPECIFIC target, and (c)
drops low-confidence boxes.

This module reproduces that 2D detect-and-rerank so "what the arm can grasp" and
"what the robot says it sees" agree. It is deliberately a SELF-CONTAINED copy (it
does NOT import ``tasks.skills.grasp``, which pulls heavy Open3D / GraspNet deps into
the agent process): the descriptor map below mirrors ``grasp._GRASP_DESCRIPTORS`` and
may drift from it — re-sync the two by hand if a hard item starts disagreeing.
"""

from __future__ import annotations

import os

import numpy as np


# Mirror of tasks.skills.grasp._GRASP_DESCRIPTORS. YOLOE can't box a brand name, but
# under a generic visual descriptor it returns boxes we then CLIP-rerank against the
# specific target. Keys are lowercased item names.
_DESCRIPTORS: dict[str, list[str]] = {
    "cola": ["can", "red can", "soda can"],
    "coke": ["can", "red can", "soda can"],
    "red bull": ["can", "blue can", "slim can", "energy drink can"],
    "ice tea": ["bottle", "carton", "drink bottle"],
    "orange juice": ["carton", "bottle", "juice carton"],
    "milk": ["carton", "bottle", "milk carton"],
    "water bottle": ["bottle", "clear bottle", "plastic bottle"],
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

# Success-only cache of CLIP text embeddings, keyed by the formatted query (not
# functools.lru_cache: that would pin a None failure forever and never retry).
_TEXT_EMB_CACHE: dict[str, list[float]] = {}


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


def _prompts_for(query: str) -> list[str]:
    """The specific target first, then its generic visual descriptors (deduped)."""
    out = [query.strip()]
    for d in _DESCRIPTORS.get(query.strip().lower(), []):
        if d not in out:
            out.append(d)
    return out


def _target_text_embedding(walkieAI, query: str) -> list[float] | None:
    """CLIP text embedding for the SPECIFIC target, cached; None on any failure."""
    tmpl = os.getenv("VISION_CLIP_QUERY_TMPL", "a photo of {t}")
    try:
        key = tmpl.format(t=query.strip())
    except (KeyError, IndexError, ValueError):
        key = query.strip()
    cached = _TEXT_EMB_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        emb = walkieAI.image.embed_text(key)
    except Exception as exc:  # noqa: BLE001 — network/empty-text; degrade to conf order
        print(f"[vision.detect] embed_text failed for {key!r} ({exc}); CLIP rerank off")
        return None
    if emb:
        _TEXT_EMB_CACHE[key] = list(emb)
        return _TEXT_EMB_CACHE[key]
    return None


def find_object_in_image(walkieAI, img, query: str, *, min_conf: float | None = None,
                         sim_floor: float | None = None):
    """Detect ``query`` the way the arm does: descriptor prompts -> CLIP rerank.

    Returns ``[(class_name, confidence, sim, bbox), ...]`` for matches, best first;
    an empty list if nothing clears the thresholds. ``sim`` is ``None`` when CLIP
    rerank is unavailable (the result then falls back to confidence-ranked boxes).
    Never raises — degrades gracefully so the caller always gets an answer.
    """
    if not query or not query.strip():
        return []
    if min_conf is None:
        min_conf = float(os.getenv("VISION_DETECT_MIN_CONFIDENCE", "0.2"))
    if sim_floor is None:
        sim_floor = float(os.getenv("VISION_CLIP_SIM_FLOOR", "0.0"))

    prompts = _prompts_for(query)
    try:
        res = walkieAI.image.process(
            img,
            detection={"prompts": prompts, "return_mask": False},
            per_detection={"embed": True},
        )
        dets = res.detection or []
    except Exception as exc:  # noqa: BLE001 — fall back to a plain prompted detect
        print(f"[vision.detect] process() failed ({exc}); plain detect")
        try:
            dets = walkieAI.image.detect(img, prompts=prompts)
        except Exception as exc2:  # noqa: BLE001
            print(f"[vision.detect] detect() failed ({exc2})")
            return []

    dets = [d for d in dets if d.confidence is None or d.confidence >= min_conf]
    if not dets:
        return []

    # CLIP-rerank against the SPECIFIC target (the descriptors are generic on
    # purpose, so detector class/confidence alone can't disambiguate the right box).
    text_emb = _target_text_embedding(walkieAI, query)
    ranked: list[tuple] = []
    if text_emb:
        for d in dets:
            emb = getattr(d, "embedding", None)
            if not emb:
                continue
            sim = _cosine(emb, text_emb)
            if sim >= sim_floor:
                ranked.append((d, sim))
        ranked.sort(key=lambda ds: ds[1], reverse=True)
    if not ranked:
        # No embeddings / rerank unavailable -> fall back to confidence order.
        ranked = [(d, None) for d in sorted(dets, key=lambda d: (d.confidence or 0.0), reverse=True)]

    return [(d.class_name, d.confidence, sim, tuple(d.bbox)) for d, sim in ranked]
