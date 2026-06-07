"""GraphMemory — the persistent 3D scene-graph store for walkie_graphs.

One mapped object = one :class:`ObjectNode`: a fused 3D point cloud + CLIP image
embedding + captions + class, with an axis-aligned bounding box. Nodes are:

- **stored** in a ChromaDB collection (CLIP image embedding + scalar metadata) for
  durable persistence and cross-modal text search,
- their **point clouds** saved as ``.npz`` sidecars (one per node),
- their **relations** held in a NetworkX ``MultiDiGraph`` and mirrored to JSON.

Incremental fusion (``upsert``) decides insert-vs-merge from CLIP cosine + 3D
distance; ``derive_relations`` adds distance-based edges (near / on / above /
inside). All mutating ops and queries take a lock — the background service writes
while the database agent reads, in the same process.

Detection/caption/embedding are NOT done here; the caller (service) provides a
:class:`Detection3D` with the embedding/caption already computed via the AI client.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:  # networkx is a core dep; guard only so a partial install fails loudly later
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

import chromadb

DEFAULT_EMB_DIM = 512  # CLIP ViT-B/16


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection3D:
    """One masked detection lifted to world-frame 3D points (input to upsert)."""

    class_name: str
    class_id: Optional[int]
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    points_world: np.ndarray  # (N, 3) float32, world frame
    clip_emb: list[float]  # CLIP image embedding (may be empty if embed failed)
    caption: str
    ts: float
    crop: object = None  # PIL.Image or None, for the archived thumbnail

    @property
    def centroid(self) -> tuple[float, float, float]:
        c = self.points_world.mean(axis=0)
        return float(c[0]), float(c[1]), float(c[2])


@dataclass
class ObjectNode:
    """One fused 3D object (a graph node)."""

    id: str
    class_name: str
    class_id: Optional[int]
    centroid: tuple[float, float, float]
    extent: tuple[float, float, float]
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
    clip_emb: list[float]
    captions: list[str]
    best_caption: str
    n_obs: int
    conf: float
    first_seen_ts: float
    last_seen_ts: float
    pcd_ref: Optional[str] = None
    frame_ref: Optional[str] = None


@dataclass(frozen=True)
class Relation:
    """A directed edge between two nodes (near is stored once, treated symmetric)."""

    src_id: str
    dst_id: str
    predicate: str  # near | on | above | inside
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def cosine(a, b) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if va.size == 0 or vb.size == 0:
        return 0.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def l2(a, b) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(va), len(vb))
    return float(np.linalg.norm(va[:n] - vb[:n]))


def aabb_of(points: np.ndarray):
    """Return (centroid, aabb_min, aabb_max, extent) as tuples from an (N,3) cloud."""
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    c = points.mean(axis=0)
    ext = mx - mn
    return (
        tuple(float(x) for x in c),
        tuple(float(x) for x in mn),
        tuple(float(x) for x in mx),
        tuple(float(x) for x in ext),
    )


def _xy_overlap(a: ObjectNode, b: ObjectNode) -> float:
    """Footprint overlap ratio = intersection / smaller AABB area (XY plane).

    Ratio (not IoU) so a small object fully over a large surface scores ~1.0 — IoU
    would be tiny there and miss every "mug on table".
    """
    ax0, ay0 = a.aabb_min[0], a.aabb_min[1]
    ax1, ay1 = a.aabb_max[0], a.aabb_max[1]
    bx0, by0 = b.aabb_min[0], b.aabb_min[1]
    bx1, by1 = b.aabb_max[0], b.aabb_max[1]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _volume(n: ObjectNode) -> float:
    return n.extent[0] * n.extent[1] * n.extent[2]


# ---------------------------------------------------------------------------
# GraphMemory
# ---------------------------------------------------------------------------
class GraphMemory:
    """Persistent node/edge store. Construct via :meth:`from_env` in production."""

    def __init__(
        self,
        *,
        chroma_dir: Optional[str] = None,
        pcds_dir: str = "graph_pcds",
        thumbs_dir: str = "graph_thumbs",
        edges_path: Optional[str] = "graph_edges.json",
        embed_text: Optional[Callable[[str], list[float]]] = None,
        dedup_radius_m: float = 0.4,
        dedup_tight_m: float = 0.2,
        sim_high: float = 0.85,
        sim_low: float = 0.65,
        dedup_visual_k: int = 5,
        voxel_m: float = 0.02,
        max_points_per_obj: int = 2000,
        relation_max_dist: float = 1.0,
        near_m: float = 0.6,
        xy_overlap_min: float = 0.15,
        z_tol: float = 0.05,
        on_gap: float = 0.08,
        contain_tol: float = 0.02,
        prune_max_records: int = 500,
        emb_dim: int = DEFAULT_EMB_DIM,
    ) -> None:
        self.pcds_dir = Path(pcds_dir)
        self.thumbs_dir = Path(thumbs_dir)
        self.pcds_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.edges_path = Path(edges_path) if edges_path else None
        self.embed_text = embed_text

        self.dedup_radius_m = dedup_radius_m
        self.dedup_tight_m = dedup_tight_m
        self.sim_high = sim_high
        self.sim_low = sim_low
        self.dedup_visual_k = dedup_visual_k
        self.voxel_m = voxel_m
        self.max_points_per_obj = max_points_per_obj
        self.relation_max_dist = relation_max_dist
        self.near_m = near_m
        self.xy_overlap_min = xy_overlap_min
        self.z_tol = z_tol
        self.on_gap = on_gap
        self.contain_tol = contain_tol
        self.prune_max_records = prune_max_records
        self.emb_dim = emb_dim

        self._lock = threading.RLock()
        if chroma_dir:
            Path(chroma_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(chroma_dir))
            col_name = "objects"
        else:
            # EphemeralClient shares one in-memory DB process-wide, so give each
            # in-memory store a unique collection to stay isolated (tests).
            self._client = chromadb.EphemeralClient()
            col_name = f"objects_{uuid.uuid4().hex[:8]}"
        self._col = self._client.get_or_create_collection(
            name=col_name, metadata={"hnsw:space": "cosine"}
        )

        self._nodes: dict[str, ObjectNode] = {}
        self._graph = nx.MultiDiGraph() if nx is not None else None
        self._load_nodes()
        self._load_edges()

    @classmethod
    def from_env(cls, *, embed_text=None) -> "GraphMemory":
        """Build from the WALKIE_GRAPHS_* environment (config.toml defaults)."""

        def _f(name, default):
            return float(os.getenv(name, default))

        def _i(name, default):
            return int(os.getenv(name, default))

        return cls(
            chroma_dir=os.getenv("WALKIE_GRAPHS_CHROMA_DIR", "chroma_db_graph"),
            pcds_dir=os.getenv("WALKIE_GRAPHS_PCDS_DIR", "graph_pcds"),
            thumbs_dir=os.getenv("WALKIE_GRAPHS_THUMBS_DIR", "graph_thumbs"),
            edges_path=os.getenv("WALKIE_GRAPHS_EDGES_PATH", "graph_edges.json"),
            embed_text=embed_text,
            dedup_radius_m=_f("WALKIE_GRAPHS_DEDUP_RADIUS_M", "0.4"),
            dedup_tight_m=_f("WALKIE_GRAPHS_DEDUP_TIGHT_M", "0.2"),
            sim_high=_f("WALKIE_GRAPHS_SIM_HIGH", "0.85"),
            sim_low=_f("WALKIE_GRAPHS_SIM_LOW", "0.65"),
            dedup_visual_k=_i("WALKIE_GRAPHS_DEDUP_VISUAL_K", "5"),
            voxel_m=_f("WALKIE_GRAPHS_VOXEL_M", "0.02"),
            max_points_per_obj=_i("WALKIE_GRAPHS_MAX_POINTS_PER_OBJ", "2000"),
            relation_max_dist=_f("WALKIE_GRAPHS_RELATION_MAX_DIST", "1.0"),
            near_m=_f("WALKIE_GRAPHS_NEAR_M", "0.6"),
            xy_overlap_min=_f("WALKIE_GRAPHS_XY_OVERLAP_MIN", "0.15"),
            z_tol=_f("WALKIE_GRAPHS_Z_TOL", "0.05"),
            on_gap=_f("WALKIE_GRAPHS_ON_GAP", "0.08"),
            contain_tol=_f("WALKIE_GRAPHS_CONTAIN_TOL", "0.02"),
            prune_max_records=_i("WALKIE_GRAPHS_PRUNE_MAX_RECORDS", "500"),
        )

    # ------------------------------------------------------------------
    # Persistence (de)serialization
    # ------------------------------------------------------------------
    def _metadata(self, n: ObjectNode) -> dict:
        return {
            "class_name": n.class_name,
            "class_id": -1 if n.class_id is None else int(n.class_id),
            "x": n.centroid[0], "y": n.centroid[1], "z": n.centroid[2],
            "ext_x": n.extent[0], "ext_y": n.extent[1], "ext_z": n.extent[2],
            "aabb_min": json.dumps(n.aabb_min),
            "aabb_max": json.dumps(n.aabb_max),
            "captions_json": json.dumps(n.captions),
            "best_caption": n.best_caption,
            "n_obs": int(n.n_obs),
            "conf": float(n.conf),
            "first_seen_ts": float(n.first_seen_ts),
            "last_seen_ts": float(n.last_seen_ts),
            "pcd_ref": n.pcd_ref or "",
            "frame_ref": n.frame_ref or "",
        }

    @staticmethod
    def _node_from_chroma(node_id: str, emb, md: dict) -> ObjectNode:
        cid = md.get("class_id", -1)
        return ObjectNode(
            id=node_id,
            class_name=md.get("class_name", ""),
            class_id=None if cid == -1 else int(cid),
            centroid=(md["x"], md["y"], md["z"]),
            extent=(md["ext_x"], md["ext_y"], md["ext_z"]),
            aabb_min=tuple(json.loads(md["aabb_min"])),
            aabb_max=tuple(json.loads(md["aabb_max"])),
            clip_emb=list(emb) if emb is not None else [],
            captions=json.loads(md.get("captions_json", "[]")),
            best_caption=md.get("best_caption", ""),
            n_obs=int(md.get("n_obs", 1)),
            conf=float(md.get("conf", 0.0)),
            first_seen_ts=float(md.get("first_seen_ts", 0.0)),
            last_seen_ts=float(md.get("last_seen_ts", 0.0)),
            pcd_ref=md.get("pcd_ref") or None,
            frame_ref=md.get("frame_ref") or None,
        )

    def _load_nodes(self) -> None:
        got = self._col.get(include=["embeddings", "metadatas"])
        ids = got.get("ids") or []
        embs = got.get("embeddings")
        mds = got.get("metadatas") or []
        for i, node_id in enumerate(ids):
            emb = embs[i] if embs is not None and i < len(embs) else None
            self._nodes[node_id] = self._node_from_chroma(node_id, emb, mds[i])

    def _write_node(self, n: ObjectNode) -> None:
        emb = n.clip_emb if n.clip_emb else [0.0] * self.emb_dim
        self._col.upsert(
            ids=[n.id],
            embeddings=[list(emb)],
            metadatas=[self._metadata(n)],
            documents=[n.best_caption or n.class_name],
        )
        self._nodes[n.id] = n

    def _pcd_path(self, node_id: str) -> Path:
        return self.pcds_dir / f"{node_id}.npz"

    def _save_pcd(self, node_id: str, points: np.ndarray) -> str:
        path = self._pcd_path(node_id)
        np.savez_compressed(path, points=points.astype(np.float32))
        return str(path)

    def load_pcd(self, node_id: str) -> np.ndarray:
        path = self._pcd_path(node_id)
        if not path.exists():
            return np.zeros((0, 3), dtype=np.float32)
        return np.load(path)["points"]

    def _save_thumb(self, node_id: str, crop) -> Optional[str]:
        if crop is None:
            return None
        path = self.thumbs_dir / f"{node_id}.jpg"
        try:
            crop.convert("RGB").save(path, "JPEG", quality=85)
        except Exception:
            return None
        return str(path)

    # ------------------------------------------------------------------
    # Ingestion (fusion / dedup)
    # ------------------------------------------------------------------
    def upsert(self, det: Detection3D) -> ObjectNode:
        """Insert a detection as a new node, or merge it into an existing one."""
        with self._lock:
            candidates = self._candidates(det)
            target = self._classify(det, candidates)
            if target is None:
                return self._insert(det)
            return self._merge(target, det)

    def _candidates(self, det: Detection3D) -> list[ObjectNode]:
        same = [n for n in self._nodes.values() if n.class_name == det.class_name]
        if not same:
            return []
        c = det.centroid
        spatial = sorted(
            (n for n in same if l2(c, n.centroid) <= self.dedup_radius_m),
            key=lambda n: l2(c, n.centroid),
        )
        visual: list[ObjectNode] = []
        if self.dedup_visual_k > 0 and det.clip_emb:
            scored = sorted(
                same, key=lambda n: cosine(det.clip_emb, n.clip_emb), reverse=True
            )
            visual = scored[: self.dedup_visual_k]
        # Spatial first (closest wins), then visual extras not already included.
        out = list(spatial)
        seen = {n.id for n in out}
        out.extend(n for n in visual if n.id not in seen)
        return out

    def _classify(self, det: Detection3D, candidates: list[ObjectNode]):
        for n in candidates:
            cos = cosine(det.clip_emb, n.clip_emb)
            dist = l2(det.centroid, n.centroid)
            if cos >= self.sim_high:
                return n
            if cos >= self.sim_low and dist <= self.dedup_tight_m:
                return n
        return None

    def _insert(self, det: Detection3D) -> ObjectNode:
        node_id = f"{det.class_name}-{uuid.uuid4().hex[:8]}"
        pts = det.points_world.astype(np.float32)
        centroid, mn, mx, ext = aabb_of(pts)
        emb = _normalize(det.clip_emb)
        caption = det.caption.strip()
        node = ObjectNode(
            id=node_id,
            class_name=det.class_name,
            class_id=det.class_id,
            centroid=centroid, extent=ext, aabb_min=mn, aabb_max=mx,
            clip_emb=emb,
            captions=[caption] if caption else [],
            best_caption=caption,
            n_obs=1,
            conf=float(det.confidence),
            first_seen_ts=det.ts,
            last_seen_ts=det.ts,
        )
        node.pcd_ref = self._save_pcd(node_id, pts)
        node.frame_ref = self._save_thumb(node_id, det.crop)
        self._write_node(node)
        return node

    def _merge(self, node: ObjectNode, det: Detection3D) -> ObjectNode:
        n = node.n_obs
        far = l2(det.centroid, node.centroid) > self.dedup_radius_m
        det_pts = det.points_world.astype(np.float32)

        if far:
            # Visual-only re-sighting whose 3D estimate drifted: don't average two
            # far positions into empty space — keep the higher-confidence geometry.
            if det.confidence > node.conf:
                centroid, mn, mx, ext = aabb_of(det_pts)
                node.centroid, node.aabb_min, node.aabb_max, node.extent = (
                    centroid, mn, mx, ext
                )
                node.pcd_ref = self._save_pcd(node.id, det_pts)
            node.conf = max(node.conf, float(det.confidence))
        else:
            merged = np.vstack([self.load_pcd(node.id), det_pts])
            merged = _voxel(merged, self.voxel_m)
            if len(merged) > self.max_points_per_obj:
                idx = np.linspace(0, len(merged) - 1, self.max_points_per_obj).astype(int)
                merged = merged[idx]
            centroid, mn, mx, ext = aabb_of(merged)
            node.centroid, node.aabb_min, node.aabb_max, node.extent = (
                centroid, mn, mx, ext
            )
            node.pcd_ref = self._save_pcd(node.id, merged)
            node.conf = (node.conf * n + float(det.confidence)) / (n + 1)

        if det.clip_emb:
            blended = np.asarray(node.clip_emb, float) * n + np.asarray(det.clip_emb, float)
            node.clip_emb = _normalize(blended / (n + 1))

        cap = det.caption.strip()
        if cap and cap not in node.captions:
            node.captions.append(cap)
        if cap and len(cap) > len(node.best_caption):
            node.best_caption = cap

        node.n_obs = n + 1
        node.last_seen_ts = det.ts
        if det.crop is not None:
            node.frame_ref = self._save_thumb(node.id, det.crop)
        self._write_node(node)
        return node

    # ------------------------------------------------------------------
    # Relations (geometric / distance-based)
    # ------------------------------------------------------------------
    def derive_relations(self) -> list[Relation]:
        """Recompute all geometric edges from the current node geometry."""
        with self._lock:
            rels: list[Relation] = []
            nodes = list(self._nodes.values())
            for i, a in enumerate(nodes):
                for b in nodes[i + 1:]:
                    d = l2(a.centroid, b.centroid)
                    if d > self.relation_max_dist:
                        continue
                    if d <= self.near_m:
                        w = 1.0 - d / self.near_m if self.near_m > 0 else 1.0
                        rels.append(Relation(a.id, b.id, "near", round(w, 3)))
                    for x, y in ((a, b), (b, a)):
                        pred = self._vertical(x, y)
                        if pred:
                            rels.append(Relation(x.id, y.id, pred, 1.0))
                        if self._inside(x, y):
                            rels.append(Relation(x.id, y.id, "inside", 1.0))
            self._set_relations(rels)
            return rels

    def _vertical(self, x: ObjectNode, y: ObjectNode) -> Optional[str]:
        """'x on/above y': x sits over y with overlapping footprint."""
        if _xy_overlap(x, y) < self.xy_overlap_min:
            return None
        gap = x.aabb_min[2] - y.aabb_max[2]
        if gap < -self.z_tol:  # x's base is well below y's top → not on top
            return None
        return "on" if gap <= self.on_gap else "above"

    def _inside(self, x: ObjectNode, y: ObjectNode) -> bool:
        """'x inside y': x's AABB is contained in y's (with slack) and smaller."""
        t = self.contain_tol
        contained = all(
            y.aabb_min[i] - t <= x.aabb_min[i] and x.aabb_max[i] <= y.aabb_max[i] + t
            for i in range(3)
        )
        return contained and _volume(x) < _volume(y)

    def _set_relations(self, rels: list[Relation]) -> None:
        if self._graph is not None:
            self._graph.clear()
            self._graph.add_nodes_from(self._nodes.keys())
            for r in rels:
                self._graph.add_edge(r.src_id, r.dst_id, key=r.predicate, weight=r.weight)
        self._persist_edges(rels)

    def _persist_edges(self, rels: list[Relation]) -> None:
        if self.edges_path is None:
            return
        data = [
            {"src": r.src_id, "dst": r.dst_id, "predicate": r.predicate, "weight": r.weight}
            for r in rels
        ]
        tmp = self.edges_path.with_suffix(self.edges_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w") as f:
            json.dump(data, f)
        os.replace(tmp, self.edges_path)

    def _load_edges(self) -> None:
        if self.edges_path is None or not self.edges_path.exists():
            return
        try:
            data = json.loads(self.edges_path.read_text())
        except Exception:
            return
        if self._graph is None:
            return
        self._graph.add_nodes_from(self._nodes.keys())
        for e in data:
            if e["src"] in self._nodes and e["dst"] in self._nodes:
                self._graph.add_edge(
                    e["src"], e["dst"], key=e["predicate"], weight=e.get("weight", 1.0)
                )

    def relations_of(self, node_id: str) -> list[Relation]:
        with self._lock:
            if self._graph is None or node_id not in self._graph:
                return []
            out = []
            for u, v, k, data in self._graph.out_edges(node_id, keys=True, data=True):
                out.append(Relation(u, v, k, data.get("weight", 1.0)))
            for u, v, k, data in self._graph.in_edges(node_id, keys=True, data=True):
                out.append(Relation(u, v, k, data.get("weight", 1.0)))
            return out

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def query_text(
        self, query: str, k: int = 5, *, near=None, radius: Optional[float] = None
    ) -> list[ObjectNode]:
        """Find objects by text. Uses CLIP cross-modal search; falls back to keyword."""
        with self._lock:
            vec = None
            if self.embed_text is not None:
                try:
                    vec = self.embed_text(query)
                except Exception:
                    vec = None
            if not vec:
                return self._keyword(query, k, near=near, radius=radius)
            try:
                res = self._col.query(query_embeddings=[list(vec)], n_results=max(k * 3, k))
                ids = (res.get("ids") or [[]])[0]
            except Exception:
                return self._keyword(query, k, near=near, radius=radius)
            hits = [self._nodes[i] for i in ids if i in self._nodes]
            hits = self._spatial_filter(hits, near, radius)
            return hits[:k]

    def _keyword(self, query: str, k: int, *, near=None, radius=None) -> list[ObjectNode]:
        terms = {w for w in query.lower().split() if w}
        scored = []
        for n in self._nodes.values():
            text = f"{n.class_name} {n.best_caption} {' '.join(n.captions)}".lower()
            score = sum(1 for t in terms if t in text)
            if score:
                scored.append((score, n))
        scored.sort(key=lambda s: s[0], reverse=True)
        hits = [n for _, n in scored]
        hits = self._spatial_filter(hits, near, radius)
        return hits[:k]

    def _spatial_filter(self, nodes, near, radius):
        if near is None or radius is None:
            return nodes
        return [n for n in nodes if l2(n.centroid, near) <= radius]

    def query_near(self, center, radius: float) -> list[ObjectNode]:
        """Objects within ``radius`` of ``center`` (2D center → horizontal distance)."""
        with self._lock:
            out = [(l2(n.centroid, center), n) for n in self._nodes.values()]
            out = [(d, n) for d, n in out if d <= radius]
            out.sort(key=lambda s: s[0])
            return [n for _, n in out]

    def recently_seen(self, limit: int = 5) -> list[ObjectNode]:
        with self._lock:
            nodes = sorted(
                self._nodes.values(), key=lambda n: n.last_seen_ts, reverse=True
            )
            return nodes[:limit]

    def all_objects(self) -> list[ObjectNode]:
        with self._lock:
            return list(self._nodes.values())

    def get(self, node_id: str) -> Optional[ObjectNode]:
        with self._lock:
            return self._nodes.get(node_id)

    def count(self) -> int:
        with self._lock:
            return len(self._nodes)

    def to_text_description(self) -> str:
        """Plain-text dump of objects + relations, for LLM task planning."""
        with self._lock:
            nodes = sorted(self._nodes.values(), key=lambda n: n.last_seen_ts, reverse=True)
            lines = [f"Objects ({len(nodes)}):"]
            for n in nodes:
                cap = f' "{n.best_caption}"' if n.best_caption else ""
                x, y, z = n.centroid
                lines.append(
                    f" [{n.id}] {n.class_name}{cap} at ({x:.2f}, {y:.2f}, {z:.2f}) "
                    f"seen {n.n_obs}x"
                )
            rels = self._graph.edges(keys=True) if self._graph is not None else []
            if rels:
                lines.append("Relations:")
                for u, v, kpred in rels:
                    cu = self._nodes[u].class_name if u in self._nodes else u
                    cv = self._nodes[v].class_name if v in self._nodes else v
                    lines.append(f" {cu} [{u}] {kpred} {cv} [{v}]")
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def prune(self, max_records: Optional[int] = None) -> int:
        """Evict the oldest nodes (by last_seen) beyond the capacity cap."""
        cap = self.prune_max_records if max_records is None else max_records
        with self._lock:
            if cap <= 0 or len(self._nodes) <= cap:
                return 0
            ordered = sorted(self._nodes.values(), key=lambda n: n.last_seen_ts)
            doomed = ordered[: len(self._nodes) - cap]
            for n in doomed:
                self._delete(n.id)
            return len(doomed)

    def _delete(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        try:
            self._col.delete(ids=[node_id])
        except Exception:
            pass
        self._pcd_path(node_id).unlink(missing_ok=True)
        (self.thumbs_dir / f"{node_id}.jpg").unlink(missing_ok=True)
        if self._graph is not None and node_id in self._graph:
            self._graph.remove_node(node_id)

    def clear(self) -> None:
        with self._lock:
            for node_id in list(self._nodes.keys()):
                self._delete(node_id)
            if self._graph is not None:
                self._graph.clear()
            self._persist_edges([])


# ---------------------------------------------------------------------------
def _normalize(vec) -> list[float]:
    v = np.asarray(vec, dtype=float)
    if v.size == 0:
        return []
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()


def _voxel(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel is None or voxel <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_cells = int(inverse.max()) + 1
    sums = np.zeros((n_cells, 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse, minlength=n_cells).reshape(-1, 1)
    return (sums / counts).astype(np.float32)
