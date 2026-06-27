"""Lean immutable scene store — the query backend for walkie_graphs v2.

Replaces the ChromaDB-backed :class:`GraphMemory` *for queries*. A
:class:`SceneStore` holds a single immutable :class:`BuiltScene` snapshot
(nodes + an L2-normalized embedding matrix + an id index + relations) behind one
:class:`threading.RLock`. Queries read the snapshot pointer after a brief lock;
a rebuild assembles a brand-new :class:`BuiltScene` and swaps the pointer in one
assignment, so an in-flight query keeps reading its own consistent snapshot and
never blocks the rebuild.

The query contract is a byte-for-byte reproduction of the v1 ``GraphMemory``
queries (consumers: ``agents/database_agent/tools.py``, ``tasks/GPSR/skills.py``):
CLIP text search via one normalized matmul with a four-trigger keyword fallback,
a confirmation gate applied on every list query, spatial filtering, and the exact
``to_text_description`` format.

The ingest side is :meth:`SceneStore.merge` — a class+distance merge-into-persisted
that NEVER shrinks the node set (the caller derives relations and calls
:meth:`install`). Detection/caption/embedding happen upstream; this module is pure
numpy/scipy and knows nothing about cameras, ChromaDB, or the walkie SDK.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:  # scipy is a hard dep; guard mirrors fusion/dbscan so partial installs degrade
    from scipy.spatial import cKDTree as _cKDTree  # noqa: F401  (kept for parity / future use)
except Exception:  # pragma: no cover
    _cKDTree = None

from interfaces.perception.geometry import voxel_downsample

DEFAULT_EMB_DIM = 512  # CLIP ViT-B/16
_CAPTIONS_MAX = 20  # cap the per-node caption union (keyword recall vs. unbounded growth)
_MERGE_VOXEL_M = 0.02  # voxel for the union(concat) merge of two clouds


# ---------------------------------------------------------------------------
# Small numeric helpers (pure; importable by other modules)
# ---------------------------------------------------------------------------
def cosine(a, b) -> float:
    """Cosine similarity of two vectors; 0.0 if either is empty or zero-norm."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if va.size == 0 or vb.size == 0:
        return 0.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def l2(a, b) -> float:
    """Euclidean distance over the shorter common length (2D center vs 3D centroid XY)."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(va), len(vb))
    return float(np.linalg.norm(va[:n] - vb[:n]))


def aabb_of(points: np.ndarray):
    """Return (centroid, aabb_min, aabb_max, extent) as 3-tuples from an (N,3) cloud."""
    pts = np.asarray(points, dtype=float)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    c = pts.mean(axis=0)
    ext = mx - mn
    return (
        tuple(float(x) for x in c),
        tuple(float(x) for x in mn),
        tuple(float(x) for x in mx),
        tuple(float(x) for x in ext),
    )


def _normalize(vec) -> list[float]:
    """L2-normalize a vector to a python list ([] for an empty input)."""
    v = np.asarray(vec, dtype=float)
    if v.size == 0:
        return []
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()


_voxel = voxel_downsample


# ---------------------------------------------------------------------------
# Canonical shared types
# ---------------------------------------------------------------------------
@dataclass
class ObjectNode:
    """One fused 3D object (a scene-graph node)."""

    id: str
    class_name: str
    class_id: Optional[int]
    centroid: tuple[float, float, float]
    extent: tuple[float, float, float]
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
    clip_emb: list[float]  # CLIP image embedding (L2-normalized; [] if unknown)
    captions: list[str]  # UNION of all member captions (keyword recall depends on this)
    best_caption: str
    n_obs: int
    conf: float
    first_seen_ts: float
    last_seen_ts: float
    points: Optional[np.ndarray] = None  # RAM-only fused cloud (NOT persisted to nodes.json)


@dataclass(frozen=True)
class Relation:
    """A directed edge between two nodes (near is stored once, treated symmetric)."""

    src_id: str
    dst_id: str
    predicate: str  # one of: near | on | above | inside
    weight: float = 1.0


@dataclass(frozen=True)
class BuiltScene:
    """Immutable snapshot a query reads — assembled once per rebuild, never mutated."""

    nodes: list[ObjectNode]
    emb: np.ndarray  # (N, D) float32, L2-normalized rows; zero-row for no-embedding nodes
    id_index: dict[str, int]
    relations: list[Relation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SceneStore
# ---------------------------------------------------------------------------
class SceneStore:
    """Immutable-snapshot scene store: lock-free queries, pointer-swap rebuilds."""

    def __init__(
        self,
        *,
        store_dir: str | Path | None = None,
        embed_text: Optional[Callable[[str], list[float]]] = None,
        embed_dim: int = DEFAULT_EMB_DIM,
        min_obs_confirm: int = 3,
        require_confirmation: bool = True,
        prune_max_records: int = 500,
        merge_dist: float = 0.5,
    ) -> None:
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self.embed_text = embed_text
        self.embed_dim = int(embed_dim)
        self.min_obs_confirm = int(min_obs_confirm)
        self.require_confirmation = bool(require_confirmation)
        self.prune_max_records = int(prune_max_records)
        self.merge_dist = float(merge_dist)

        self._lock = threading.RLock()
        self._scene = BuiltScene(nodes=[], emb=self._empty_emb(), id_index={})
        if self.store_dir is not None and self._nodes_path().exists():
            self.load()

    # ------------------------------------------------------------------
    # Snapshot assembly
    # ------------------------------------------------------------------
    def _empty_emb(self) -> np.ndarray:
        return np.zeros((0, self.embed_dim), dtype=np.float32)

    def _build_scene(self, nodes: list[ObjectNode], relations: list[Relation]) -> BuiltScene:
        """Assemble the (N,D) L2-normalized emb matrix + id_index into a new snapshot."""
        id_index = {n.id: i for i, n in enumerate(nodes)}
        dim = self.embed_dim
        for n in nodes:  # adopt the stored embedding width if it differs from the default
            if n.clip_emb:
                dim = len(n.clip_emb)
                break
        emb = np.zeros((len(nodes), dim), dtype=np.float32)
        for i, n in enumerate(nodes):
            if not n.clip_emb:
                continue
            v = np.asarray(n.clip_emb, dtype=np.float32)
            if v.shape[0] != dim:
                continue  # width mismatch -> leave a zero row (no match) rather than crash
            nrm = float(np.linalg.norm(v))
            if nrm > 0:
                emb[i] = v / nrm
        return BuiltScene(nodes=list(nodes), emb=emb, id_index=id_index, relations=list(relations))

    def install(self, nodes: list[ObjectNode], relations: list[Relation]) -> None:
        """Build a new snapshot from ``nodes``/``relations``, swap it in, and persist.

        ONE pointer assignment under the lock: any in-flight query keeps reading the
        previous (consistent) snapshot until it finishes.
        """
        scene = self._build_scene(nodes, relations)
        with self._lock:
            self._scene = scene
        self._persist()

    def _snapshot(self) -> BuiltScene:
        """Read the current snapshot pointer (brief lock, then lock-free use)."""
        with self._lock:
            return self._scene

    # ------------------------------------------------------------------
    # Ingest: merge-into-persisted, never shrink
    # ------------------------------------------------------------------
    def merge(self, observations, now: float) -> list[ObjectNode]:
        """Fold a batch of observations into the persisted nodes, never shrinking.

        For each observation, find the best existing node with the SAME ``class_name``
        and centroid within ``merge_dist``; update it in place, else insert a new node
        with a fresh uuid4 id. Applies the ``prune_max_records`` cap by ``last_seen_ts``
        (oldest dropped). Returns the merged node list — the caller derives relations
        and calls :meth:`install`.
        """
        scene = self._snapshot()
        # Work on shallow copies so an in-flight query reading the old snapshot is safe.
        nodes: list[ObjectNode] = [self._copy_node(n) for n in scene.nodes]

        for obs in observations:
            cls = obs.class_name
            oc = obs.centroid
            best_i = -1
            best_d = self.merge_dist
            for i, n in enumerate(nodes):
                if n.class_name != cls:
                    continue
                d = l2(n.centroid, oc)
                if d <= best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                self._update_node(nodes[best_i], obs)
            else:
                nodes.append(self._node_from_obs(obs, now))

        # Prune by recency to the cap (oldest dropped); a no-op below the cap.
        cap = self.prune_max_records
        if cap > 0 and len(nodes) > cap:
            nodes.sort(key=lambda n: n.last_seen_ts, reverse=True)
            nodes = nodes[:cap]
        return nodes

    @staticmethod
    def _copy_node(n: ObjectNode) -> ObjectNode:
        return ObjectNode(
            id=n.id,
            class_name=n.class_name,
            class_id=n.class_id,
            centroid=n.centroid,
            extent=n.extent,
            aabb_min=n.aabb_min,
            aabb_max=n.aabb_max,
            clip_emb=list(n.clip_emb),
            captions=list(n.captions),
            best_caption=n.best_caption,
            n_obs=n.n_obs,
            conf=n.conf,
            first_seen_ts=n.first_seen_ts,
            last_seen_ts=n.last_seen_ts,
            points=n.points,
        )

    def _node_from_obs(self, obs, now: float) -> ObjectNode:
        caps = self._dedup_captions(list(getattr(obs, "captions", []) or []))
        pts = getattr(obs, "points", None)
        if pts is not None and len(pts):
            pts = np.asarray(pts, dtype=np.float32)
            centroid, mn, mx, ext = aabb_of(pts)
        else:
            pts = None
            centroid = tuple(float(x) for x in obs.centroid)
            ext = tuple(float(x) for x in obs.extent)
            mn = tuple(float(x) for x in obs.aabb_min)
            mx = tuple(float(x) for x in obs.aabb_max)
        return ObjectNode(
            id=f"{obs.class_name}-{uuid.uuid4().hex[:8]}",
            class_name=obs.class_name,
            class_id=obs.class_id,
            centroid=centroid,
            extent=ext,
            aabb_min=mn,
            aabb_max=mx,
            clip_emb=_normalize(getattr(obs, "clip_emb", []) or []),
            captions=caps,
            best_caption=self._longest(caps),
            n_obs=int(getattr(obs, "n_obs", 1) or 1),
            conf=float(getattr(obs, "conf", 0.0) or 0.0),
            first_seen_ts=float(getattr(obs, "ts_first", now)),
            last_seen_ts=float(getattr(obs, "ts_last", now)),
            points=pts,
        )

    def _update_node(self, node: ObjectNode, obs) -> None:
        n0 = node.n_obs
        n1 = int(getattr(obs, "n_obs", 1) or 1)
        node.n_obs = n0 + n1
        node.last_seen_ts = max(node.last_seen_ts, float(getattr(obs, "ts_last", node.last_seen_ts)))
        node.first_seen_ts = min(node.first_seen_ts, float(getattr(obs, "ts_first", node.first_seen_ts)))
        node.conf = max(node.conf, float(getattr(obs, "conf", 0.0) or 0.0))

        # Captions: union (dedup, capped); best = longest non-empty.
        caps = list(node.captions)
        for c in getattr(obs, "captions", []) or []:
            c = (c or "").strip()
            if c and c not in caps:
                caps.append(c)
        node.captions = self._dedup_captions(caps)
        best = self._longest(node.captions)
        if best:
            node.best_caption = best

        # Embedding: L2-normalized weighted mean by n_obs.
        obs_emb = getattr(obs, "clip_emb", []) or []
        if obs_emb:
            if node.clip_emb:
                blended = np.asarray(node.clip_emb, float) * n0 + np.asarray(obs_emb, float) * n1
                node.clip_emb = _normalize(blended / (n0 + n1))
            else:
                node.clip_emb = _normalize(obs_emb)

        # Points: union(voxel(concat)) if both present else whichever exists; recompute geom.
        obs_pts = getattr(obs, "points", None)
        obs_pts = np.asarray(obs_pts, dtype=np.float32) if obs_pts is not None and len(obs_pts) else None
        if node.points is not None and obs_pts is not None:
            node.points = _voxel(np.vstack([node.points, obs_pts]), _MERGE_VOXEL_M)
        elif obs_pts is not None:
            node.points = obs_pts
        if node.points is not None and len(node.points):
            centroid, mn, mx, ext = aabb_of(node.points)
            node.centroid, node.aabb_min, node.aabb_max, node.extent = centroid, mn, mx, ext

    @staticmethod
    def _dedup_captions(caps: list[str]) -> list[str]:
        out: list[str] = []
        for c in caps:
            c = (c or "").strip()
            if c and c not in out:
                out.append(c)
            if len(out) >= _CAPTIONS_MAX:
                break
        return out

    @staticmethod
    def _longest(caps: list[str]) -> str:
        best = ""
        for c in caps:
            if c and len(c) > len(best):
                best = c
        return best

    # ------------------------------------------------------------------
    # Queries (v1 GraphMemory contract)
    # ------------------------------------------------------------------
    def query_text(
        self, query: str, k: int = 5, *, near=None, radius: Optional[float] = None
    ) -> list[ObjectNode]:
        """Find objects by text. CLIP cross-modal search; falls back to keyword.

        Four fallback triggers, each routing to :meth:`_keyword`: ``embed_text`` is
        None, it raises, it returns a falsy/empty vector, or the vector search itself
        fails for any reason.
        """
        scene = self._snapshot()
        vec = None
        if self.embed_text is not None:
            try:
                vec = self.embed_text(query)
            except Exception:
                vec = None
        if not vec:  # embed None / empty / failed
            return self._keyword(scene, query, k, near=near, radius=radius)
        try:
            qv = np.asarray(vec, dtype=np.float32)
            nrm = float(np.linalg.norm(qv))
            if nrm == 0 or scene.emb.shape[0] == 0 or qv.shape[0] != scene.emb.shape[1]:
                raise ValueError("query vector unusable for this scene")
            qv = qv / nrm
            scores = scene.emb @ qv  # (N,) cosine over L2-normalized rows
            top_n = max(k * 3, k)
            order = np.argsort(scores)[::-1][:top_n]
            hits = [scene.nodes[int(i)] for i in order]
        except Exception:
            return self._keyword(scene, query, k, near=near, radius=radius)
        hits = self._confirmed(self._spatial_filter(hits, near, radius))
        return hits[:k]

    def _keyword(self, scene: BuiltScene, query: str, k: int, *, near=None, radius=None):
        terms = {w for w in query.lower().split() if w}
        scored = []
        for n in scene.nodes:
            text = f"{n.class_name} {n.best_caption} {' '.join(n.captions)}".lower()
            score = sum(1 for t in terms if t in text)
            if score:
                scored.append((score, n))
        scored.sort(key=lambda s: s[0], reverse=True)
        hits = [n for _, n in scored]
        hits = self._confirmed(self._spatial_filter(hits, near, radius))
        return hits[:k]

    def _confirmed(self, nodes: list[ObjectNode]) -> list[ObjectNode]:
        """Drop provisional nodes (``n_obs < min_obs_confirm``) when confirmation is on."""
        if not self.require_confirmation:
            return nodes
        return [n for n in nodes if n.n_obs >= self.min_obs_confirm]

    @staticmethod
    def _spatial_filter(nodes, near, radius):
        if near is None or radius is None:
            return nodes
        return [n for n in nodes if l2(n.centroid, near) <= radius]

    def query_near(self, center, radius: float) -> list[ObjectNode]:
        """Confirmed objects within ``radius`` of ``center`` (2D center -> horizontal dist)."""
        scene = self._snapshot()
        out = [(l2(n.centroid, center), n) for n in self._confirmed(list(scene.nodes))]
        out = [(d, n) for d, n in out if d <= radius]
        out.sort(key=lambda s: s[0])
        return [n for _, n in out]

    def recently_seen(self, limit: int = 5) -> list[ObjectNode]:
        scene = self._snapshot()
        nodes = sorted(
            self._confirmed(list(scene.nodes)), key=lambda n: n.last_seen_ts, reverse=True
        )
        return nodes[:limit]

    def all_objects(self) -> list[ObjectNode]:
        scene = self._snapshot()
        return self._confirmed(list(scene.nodes))

    def get(self, node_id: str) -> Optional[ObjectNode]:
        """Fetch a node by id (NOT gated by confirmation)."""
        scene = self._snapshot()
        i = scene.id_index.get(node_id)
        return scene.nodes[i] if i is not None else None

    def relations_of(self, node_id: str) -> list[Relation]:
        scene = self._snapshot()
        if node_id not in scene.id_index:
            return []
        return [r for r in scene.relations if r.src_id == node_id or r.dst_id == node_id]

    def all_relations(self) -> list[Relation]:
        return list(self._snapshot().relations)

    def count(self) -> int:
        return len(self._snapshot().nodes)

    def to_text_description(self) -> str:
        """Plain-text dump of confirmed objects + their relations, for LLM planning."""
        scene = self._snapshot()
        nodes = sorted(
            self._confirmed(list(scene.nodes)), key=lambda n: n.last_seen_ts, reverse=True
        )
        shown = {n.id for n in nodes}
        by_id = {n.id: n for n in scene.nodes}
        lines = [f"Objects ({len(nodes)}):"]
        for n in nodes:
            cap = f' "{n.best_caption}"' if n.best_caption else ""
            x, y, z = n.centroid
            lines.append(
                f" [{n.id}] {n.class_name}{cap} at ({x:.2f}, {y:.2f}, {z:.2f}) "
                f"seen {n.n_obs}x"
            )
        rels = [r for r in scene.relations if r.src_id in shown and r.dst_id in shown]
        if rels:
            lines.append("Relations:")
            for r in rels:
                cu = by_id[r.src_id].class_name if r.src_id in by_id else r.src_id
                cv = by_id[r.dst_id].class_name if r.dst_id in by_id else r.dst_id
                lines.append(f" {cu} [{r.src_id}] {r.predicate} {cv} [{r.dst_id}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence: nodes.json + embeddings.npy + edges.json (atomic)
    # ------------------------------------------------------------------
    def _nodes_path(self) -> Path:
        return self.store_dir / "nodes.json"

    def _emb_path(self) -> Path:
        return self.store_dir / "embeddings.npy"

    def _edges_path(self) -> Path:
        return self.store_dir / "edges.json"

    @staticmethod
    def _node_to_dict(n: ObjectNode) -> dict:
        """Serialize a node WITHOUT its RAM-only ``points`` cloud."""
        return {
            "id": n.id,
            "class_name": n.class_name,
            "class_id": n.class_id,
            "centroid": list(n.centroid),
            "extent": list(n.extent),
            "aabb_min": list(n.aabb_min),
            "aabb_max": list(n.aabb_max),
            "clip_emb": list(n.clip_emb),
            "captions": list(n.captions),
            "best_caption": n.best_caption,
            "n_obs": int(n.n_obs),
            "conf": float(n.conf),
            "first_seen_ts": float(n.first_seen_ts),
            "last_seen_ts": float(n.last_seen_ts),
        }

    @staticmethod
    def _node_from_dict(d: dict, emb_row: Optional[np.ndarray]) -> ObjectNode:
        clip = list(d.get("clip_emb") or [])
        if not clip and emb_row is not None and float(np.linalg.norm(emb_row)) > 0:
            clip = [float(x) for x in emb_row]
        cid = d.get("class_id")
        return ObjectNode(
            id=d["id"],
            class_name=d.get("class_name", ""),
            class_id=None if cid is None else int(cid),
            centroid=tuple(float(x) for x in d["centroid"]),
            extent=tuple(float(x) for x in d["extent"]),
            aabb_min=tuple(float(x) for x in d["aabb_min"]),
            aabb_max=tuple(float(x) for x in d["aabb_max"]),
            clip_emb=clip,
            captions=list(d.get("captions") or []),
            best_caption=d.get("best_caption", ""),
            n_obs=int(d.get("n_obs", 1)),
            conf=float(d.get("conf", 0.0)),
            first_seen_ts=float(d.get("first_seen_ts", 0.0)),
            last_seen_ts=float(d.get("last_seen_ts", 0.0)),
            points=None,  # clouds are RAM-only, never persisted to nodes.json
        )

    def _persist(self) -> None:
        if self.store_dir is None:
            return
        scene = self._snapshot()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._write_atomic(self._nodes_path(), [self._node_to_dict(n) for n in scene.nodes])
        self._write_npy_atomic(self._emb_path(), scene.emb)
        self._write_atomic(
            self._edges_path(),
            [
                {"src": r.src_id, "dst": r.dst_id, "predicate": r.predicate, "weight": r.weight}
                for r in scene.relations
            ],
        )

    @staticmethod
    def _write_atomic(path: Path, obj) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)

    @staticmethod
    def _write_npy_atomic(path: Path, arr: np.ndarray) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            np.save(f, np.asarray(arr, dtype=np.float32))
        os.replace(tmp, path)

    def load(self) -> None:
        """Rebuild the snapshot from disk (points=None); the result answers queries."""
        if self.store_dir is None or not self._nodes_path().exists():
            return
        try:
            node_dicts = json.loads(self._nodes_path().read_text())
        except Exception:
            return
        emb = None
        if self._emb_path().exists():
            try:
                emb = np.load(self._emb_path())
            except Exception:
                emb = None
        nodes: list[ObjectNode] = []
        for i, d in enumerate(node_dicts):
            row = emb[i] if emb is not None and i < len(emb) else None
            nodes.append(self._node_from_dict(d, row))
        relations: list[Relation] = []
        if self._edges_path().exists():
            try:
                edge_dicts = json.loads(self._edges_path().read_text())
                ids = {d["id"] for d in node_dicts}
                relations = [
                    Relation(e["src"], e["dst"], e["predicate"], float(e.get("weight", 1.0)))
                    for e in edge_dicts
                    if e.get("src") in ids and e.get("dst") in ids
                ]
            except Exception:
                relations = []
        scene = self._build_scene(nodes, relations)
        with self._lock:
            self._scene = scene
