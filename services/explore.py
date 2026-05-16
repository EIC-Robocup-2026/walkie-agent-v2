from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

from db.walkie_db import WalkieVectorDB
from interfaces.walkie_interface import WalkieInterface


Position = tuple[float, float, float]


def _xyxy_to_cxcywh(bbox) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [float((x1 + x2) / 2), float((y1 + y2) / 2), float(x2 - x1), float(y2 - y1)]


def _l2(a: Position, b: Position) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


@dataclass
class _Track:
    class_name: str
    mean_position: Position
    mean_confidence: float
    count: int = 1
    promoted_id: str | None = None
    last_caption: str = ""
    last_seen_ts: float = field(default_factory=time.time)

    def update(self, position: Position, confidence: float) -> None:
        n = self.count
        # Running mean
        mp = self.mean_position
        new_mean = (
            (mp[0] * n + position[0]) / (n + 1),
            (mp[1] * n + position[1]) / (n + 1),
            (mp[2] * n + position[2]) / (n + 1),
        )
        new_conf = (self.mean_confidence * n + confidence) / (n + 1)
        self.mean_position = new_mean
        self.mean_confidence = new_conf
        self.count += 1
        self.last_seen_ts = time.time()


class ExploreService(threading.Thread):
    """Background loop: detect objects, lift to 3D, track, promote confident ones to DB.

    Algorithm:
        For each tick:
          - capture image
          - detect objects (class, bbox, conf)
          - bboxes_to_positions → 3D map-frame position per detection
          - update or create a track keyed by (class_name, spatial bucket)
          - if track satisfies min_sightings AND mean_conf >= min_conf:
                promote: insert into DB (or update existing nearby record)
    """

    def __init__(
        self,
        walkieAI,
        walkie: WalkieInterface,
        db: WalkieVectorDB,
        *,
        interval: float = 1.0,
        min_sightings: int = 5,
        dedup_radius: float = 1.0,
        min_conf: float = 0.6,
        position_timeout: float = 2.0,
        verbose: bool = True,
    ) -> None:
        super().__init__(daemon=True, name="ExploreService")
        self.walkieAI = walkieAI
        self.walkie = walkie
        self.db = db
        self.interval = interval
        self.min_sightings = min_sightings
        self.dedup_radius = dedup_radius
        self.min_conf = min_conf
        self.position_timeout = position_timeout
        self.verbose = verbose
        self._stop = threading.Event()
        self._tracks: dict[tuple[str, tuple[int, int, int]], _Track] = {}

    def stop_and_join(self, timeout: float | None = None) -> None:
        self._stop.set()
        self.join(timeout)

    def _bucket(self, pos: Position) -> tuple[int, int, int]:
        r = max(self.dedup_radius, 1e-6)
        return (round(pos[0] / r), round(pos[1] / r), round(pos[2] / r))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[explore] {msg}")

    def run(self) -> None:
        self._log(
            f"started (interval={self.interval}s, min_sightings={self.min_sightings}, "
            f"dedup_radius={self.dedup_radius}m, min_conf={self.min_conf})"
        )
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001
                self._log(f"tick error: {e}")
            self._stop.wait(self.interval)
        self._log("stopped.")

    def _tick(self) -> None:
        img = self.walkie.camera.capture_pil()
        objects = self.walkieAI.object_detection.detect(img)
        if not objects:
            return
        coords = [_xyxy_to_cxcywh(o.bbox) for o in objects]
        positions = self.walkie.tools.bboxes_to_positions(
            coords, timeout=self.position_timeout
        )
        if not positions:
            self._log("bboxes_to_positions returned None (timeout or no transport).")
            return
        # `positions` is aligned with `coords` order.
        for obj, pos in zip(objects, positions):
            if pos is None or len(pos) < 3:
                continue
            self._handle_detection(
                obj.class_name or "unknown",
                tuple(pos[:3]),
                float(obj.confidence or 0.0),
                img,
            )

    def _handle_detection(
        self,
        class_name: str,
        position: Position,
        confidence: float,
        img,
    ) -> None:
        key = (class_name, self._bucket(position))
        track = self._tracks.get(key)
        if track is None:
            track = _Track(
                class_name=class_name,
                mean_position=position,
                mean_confidence=confidence,
                count=1,
            )
            self._tracks[key] = track
        else:
            track.update(position, confidence)

        if (
            track.count >= self.min_sightings
            and track.mean_confidence >= self.min_conf
        ):
            self._promote(track, img)

    def _promote(self, track: _Track, img) -> None:
        existing = self.db.find_nearby(
            track.class_name, track.mean_position, self.dedup_radius
        )
        if existing:
            best = existing[0]
            self.db.update_object(
                best["id"],
                position=track.mean_position,
                confidence=max(track.mean_confidence, float(best.get("confidence", 0.0))),
                sightings=int(best.get("sightings", 1)) + 1,
            )
            track.promoted_id = best["id"]
            self._log(
                f"updated existing {track.class_name} @ "
                f"{tuple(round(c, 2) for c in track.mean_position)}"
            )
            return
        if track.promoted_id:
            # Already promoted by us — refresh.
            self.db.update_object(
                track.promoted_id,
                position=track.mean_position,
                confidence=track.mean_confidence,
                sightings=track.count,
            )
            return
        # New entry — caption once on promotion.
        try:
            caption = self.walkieAI.image_caption.caption(
                img, prompt=f"Briefly describe the {track.class_name} in this image."
            )
        except Exception as e:  # noqa: BLE001
            caption = ""
            self._log(f"caption failed for {track.class_name}: {e}")
        track.last_caption = caption
        obj_id = self.db.add_object(
            class_name=track.class_name,
            position=track.mean_position,
            confidence=track.mean_confidence,
            caption=caption,
            sightings=track.count,
        )
        track.promoted_id = obj_id
        self._log(
            f"promoted {track.class_name} @ "
            f"{tuple(round(c, 2) for c in track.mean_position)} "
            f"({track.count} sightings, conf={track.mean_confidence:.2f})"
        )
