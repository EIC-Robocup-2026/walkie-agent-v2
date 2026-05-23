"""Seed throwaway Chroma stores with mock data for the viewer.

Creates two persist directories that mirror the real schemas the viewer
renders, plus a folder of generated JPEG thumbnails so frame images, the
gallery, and the position map all light up:

  * ``chroma_db_mock``        — ``objects`` collection (legacy WalkieVectorDB
                                 shape: class_name/x/y/z/confidence/sightings/
                                 caption/frame_ref/last_seen_ts).
  * ``chroma_db_scene_mock``  — ``scene_entries`` collection (CLIP SceneStore
                                 shape with 512-dim embeddings + first/last
                                 seen, position_conf, embedding_model, etc.).

Embeddings are written explicitly (random unit vectors) so Chroma never has
to download a default embedding model — the seeder runs fully offline.

Run::

    uv run python -m tools.seed_mock_chroma
    uv run python -m tools.chroma_viewer --dirs chroma_db_mock,chroma_db_scene_mock
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from PIL import Image, ImageDraw

OBJ_DIR = "chroma_db_mock"
SCENE_DIR = "chroma_db_scene_mock"
FRAMES_DIR = Path("frames_mock")

# class -> (a few caption variants)
CLASSES = {
    "chair":    ["a wooden office chair", "a black swivel chair", "a folding chair"],
    "table":    ["a round wooden table", "a long meeting table", "a small side table"],
    "person":   ["a person standing", "someone seated at a desk", "a person walking by"],
    "laptop":   ["a silver laptop, lid open", "a closed laptop on a desk"],
    "cup":      ["a white coffee mug", "a glass of water", "a paper cup"],
    "bottle":   ["a plastic water bottle", "a glass bottle, half full"],
    "book":     ["a stack of books", "an open notebook", "a thick textbook"],
    "plant":    ["a potted green plant", "a tall leafy plant in the corner"],
    "monitor":  ["a wide computer monitor", "a dark monitor, powered off"],
    "keyboard": ["a mechanical keyboard", "a slim wireless keyboard"],
    "backpack": ["a blue backpack on the floor", "a black backpack on a chair"],
    "door":     ["a wooden door, closed", "an open doorway"],
}
CLASS_ID = {name: i for i, name in enumerate(sorted(CLASSES))}

rng = np.random.default_rng(7)


def _hue(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 360


def _hsl_to_rgb(h: float, s: float, light: float) -> tuple[int, int, int]:
    c = (1 - abs(2 * light - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = light - c / 2
    r, g, b = {
        0: (c, x, 0), 1: (x, c, 0), 2: (0, c, x),
        3: (0, x, c), 4: (x, 0, c), 5: (c, 0, x),
    }[int(h // 60) % 6]
    return tuple(int((v + m) * 255) for v in (r, g, b))


def _make_thumb(class_name: str, rid: str) -> str:
    """Render a small labelled JPEG keyed on class colour; return its path."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    hue = _hue(class_name)
    bg = _hsl_to_rgb(hue, 0.55, 0.45)
    img = Image.new("RGB", (320, 240), bg)
    d = ImageDraw.Draw(img)
    # a couple of accent rectangles so frames don't look flat
    d.rectangle([24, 24, 296, 216], outline=_hsl_to_rgb(hue, 0.6, 0.78), width=4)
    d.rectangle([60, 150, 260, 200], fill=_hsl_to_rgb(hue, 0.5, 0.30))
    d.text((34, 34), class_name, fill="white")
    d.text((34, 54), rid[:18], fill=_hsl_to_rgb(hue, 0.4, 0.85))
    path = FRAMES_DIR / f"{class_name}_{rid[:8]}.jpg"
    img.save(path, format="JPEG", quality=85)
    return str(path)


def _unit_vec(dim: int) -> list[float]:
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) or 1.0
    return v.tolist()


def _fresh_client(directory: str) -> chromadb.api.ClientAPI:
    shutil.rmtree(directory, ignore_errors=True)
    return chromadb.PersistentClient(
        path=str(Path(directory).resolve()),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def seed_objects(n: int = 36) -> None:
    client = _fresh_client(OBJ_DIR)
    coll = client.get_or_create_collection(
        name="objects", metadata={"hnsw:space": "cosine"}
    )
    now = time.time()
    names = list(CLASSES)
    ids, docs, metas, embs = [], [], [], []
    for _ in range(n):
        cls = names[int(rng.integers(len(names)))]
        caption = CLASSES[cls][int(rng.integers(len(CLASSES[cls])))]
        rid = str(uuid.uuid4())
        x = float(rng.uniform(-4, 4))
        y = float(rng.uniform(-3, 3))
        z = float(rng.uniform(0, 1.2))
        last_seen = now - float(rng.uniform(0, 6 * 3600))  # within last 6h
        ids.append(rid)
        docs.append(f"{cls}: {caption}")
        metas.append({
            "class_name": cls,
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
            "confidence": round(float(rng.uniform(0.45, 0.98)), 3),
            "sightings": int(rng.integers(1, 13)),
            "caption": caption,
            "frame_ref": _make_thumb(cls, rid),
            "last_seen_ts": last_seen,
        })
        embs.append(_unit_vec(384))
    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"[seed] {OBJ_DIR}/objects: {coll.count()} records")


def seed_scene_entries(n: int = 30) -> None:
    client = _fresh_client(SCENE_DIR)
    coll = client.get_or_create_collection(
        name="scene_entries", metadata={"hnsw:space": "cosine"}
    )
    now = time.time()
    names = list(CLASSES)
    ids, docs, metas, embs = [], [], [], []
    for _ in range(n):
        cls = names[int(rng.integers(len(names)))]
        caption = CLASSES[cls][int(rng.integers(len(CLASSES[cls])))]
        bx, by, bz = (round(float(rng.uniform(-4, 4)), 2),
                      round(float(rng.uniform(-3, 3)), 2),
                      round(float(rng.uniform(0, 1.5)), 2))
        rid = f"{cls}:{int(bx)}:{int(by)}:{int(bz)}:{uuid.uuid4().hex[:8]}"
        first_seen = now - float(rng.uniform(6 * 3600, 48 * 3600))  # 6-48h ago
        last_seen = first_seen + float(rng.uniform(0, 6 * 3600))
        x1 = int(rng.integers(0, 200)); y1 = int(rng.integers(0, 150))
        bbox = [x1, y1, x1 + int(rng.integers(40, 200)), y1 + int(rng.integers(40, 150))]
        ids.append(rid)
        docs.append(f"{cls}. {caption}")
        metas.append({
            "class_name": cls,
            "class_id": CLASS_ID[cls],
            "first_seen_ts": first_seen,
            "last_seen_ts": last_seen,
            "sightings": int(rng.integers(1, 20)),
            "x": bx, "y": by, "z": bz,
            "position_frame": "map",
            "position_conf": round(float(rng.uniform(0.4, 0.97)), 3),
            "caption": caption,
            "bbox_last": json.dumps(bbox),
            "frame_ref": _make_thumb(cls, rid.split(":")[-1]),
            "embedding_model": "clip-ViT-B-32",
            "embedding_dim": 512,
        })
        embs.append(_unit_vec(512))
    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"[seed] {SCENE_DIR}/scene_entries: {coll.count()} records")


def main() -> None:
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    seed_objects()
    seed_scene_entries()
    print(f"[seed] frames in {FRAMES_DIR}/")
    print("[seed] run the viewer:")
    print(f"  uv run python -m tools.chroma_viewer "
          f"--dirs {OBJ_DIR},{SCENE_DIR}")


if __name__ == "__main__":
    main()
