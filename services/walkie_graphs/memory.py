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
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:  # networkx is a core dep; guard only so a partial install fails loudly later
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:  # scipy is a hard dep; guard mirrors fusion/dbscan so partial installs degrade
    from scipy.spatial import cKDTree as _cKDTree
except Exception:  # pragma: no cover
    _cKDTree = None

from perception.vector_db import get_collection, make_client

from .dbscan import (
    dbscan_largest_cluster,
    dbscan_remove_noise,
    statistical_outlier_removal,
)
from .fusion import (
    aabb_overlap,
    additive_similarity,
    icp_align,
    nn_ratio,
    nn_ratio_symmetric,
    pairs_within,
)
from .geometry import voxel_downsample

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
    # Tier 3: up to N highest-confidence crop paths [(conf, path), ...], for the
    # multi-view LLM caption refinement. Empty unless best_views_n > 0.
    best_views: list = field(default_factory=list)


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
        visual_merge_max_dist_m: float = 0.0,
        sim_threshold: float = 1.1,
        cross_class_sim_threshold: float = 0.0,
        w_geo: float = 1.0,
        w_sem: float = 1.0,
        nn_voxel_m: float = 0.025,
        dbscan_enabled: bool = True,
        dbscan_eps: float = 0.05,
        dbscan_min_points: int = 10,
        sor_k: int = 0,
        sor_std_ratio: float = 2.0,
        icp_max_dist_m: float = 0.0,
        icp_min_fitness: float = 0.6,
        icp_min_points: int = 150,
        icp_skip_overlap: float = 0.75,
        icp_cooldown_sec: float = 0.0,
        defer_pcd_writes: bool = False,
        denoise_keep_min_frac: float = 0.5,
        merge_overlap_thresh: float = 0.7,
        merge_visual_sim_thresh: float = 0.7,
        merge_text_sim_thresh: float = 0.7,
        merge_radius_m: float = 0.5,
        min_obs_confirm: int = 1,
        require_confirmation: bool = False,
        ghost_grace_sec: float = 0.0,
        best_views_n: int = 0,
        thumbnails: bool = True,
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
        self.visual_merge_max_dist_m = visual_merge_max_dist_m
        self.sim_threshold = sim_threshold
        self.cross_class_sim_threshold = cross_class_sim_threshold
        self.w_geo = w_geo
        self.w_sem = w_sem
        self.nn_voxel_m = nn_voxel_m
        self.dbscan_enabled = dbscan_enabled
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.sor_k = sor_k
        self.sor_std_ratio = sor_std_ratio
        self.icp_max_dist_m = icp_max_dist_m
        self.icp_min_fitness = icp_min_fitness
        self.icp_min_points = icp_min_points
        self.icp_skip_overlap = icp_skip_overlap
        self.icp_cooldown_sec = icp_cooldown_sec
        self.defer_pcd_writes = defer_pcd_writes
        self.denoise_keep_min_frac = denoise_keep_min_frac
        self.merge_overlap_thresh = merge_overlap_thresh
        self.merge_visual_sim_thresh = merge_visual_sim_thresh
        self.merge_text_sim_thresh = merge_text_sim_thresh
        self.merge_radius_m = merge_radius_m
        self.min_obs_confirm = min_obs_confirm
        self.require_confirmation = require_confirmation
        self.ghost_grace_sec = ghost_grace_sec
        self.best_views_n = best_views_n
        self.thumbnails = thumbnails
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
        self._client = make_client(chroma_dir or None)
        # EphemeralClient shares one in-memory DB process-wide, so each
        # in-memory store gets a unique collection to stay isolated (tests).
        self._col = get_collection(self._client, "objects", unique_if_ephemeral=True)

        self._nodes: dict[str, ObjectNode] = {}
        self._dirty: set[str] = set()  # node ids whose cloud changed since last denoise
        # Write-through cloud cache: association reads every candidate's cloud (under
        # the lock), so serve those from memory instead of re-reading .npz from disk.
        # Bounded by the prune cap (500 nodes x 2000 pts x 12 B ~ 12 MB).
        self._pcd_cache: dict[str, np.ndarray] = {}
        # Chroma write batching: inside a ``batch_writes()`` block, node writes only
        # update the in-memory dict (what association/maintenance read) and queue the
        # Chroma upsert, which is flushed as a single call on exit — one batched
        # sqlite/HNSW write per frame instead of one per object.
        self._batching = False
        self._chroma_pending: dict[str, ObjectNode] = {}
        # Per-stage wall-time accumulators for the perf log (read+reset by the service
        # each tick via pop_perf_stats) — pinpoints where slow upserts spend their time.
        self._perf_stats: dict[str, float] = {}
        # Node ids whose cloud changed in memory but hasn't hit its .npz yet (only
        # used when defer_pcd_writes is on; flushed by flush_pcds on a cadence).
        self._pcd_unflushed: set[str] = set()
        # Per-node cKDTree over the stored cloud, for the overlap checks that run on
        # every association/merge — built once per cloud version (invalidated on save).
        self._tree_cache: dict[str, object] = {}
        # When each node last had ICP considered (ran or was skipped-as-aligned);
        # within icp_cooldown_sec it isn't reconsidered — pose error varies slowly,
        # so realigning the same object every tick is pure cost (ephemeral, not persisted).
        self._icp_last: dict[str, float] = {}
        self._graph = nx.MultiDiGraph() if nx is not None else None
        self._load_nodes()
        self._load_edges()

    def _perf_add(self, key: str, t0: float) -> None:
        self._perf_stats[key] = self._perf_stats.get(key, 0.0) + (time.perf_counter() - t0)

    def pop_perf_stats(self) -> dict[str, float]:
        """Return and reset the per-stage upsert timing accumulators (seconds)."""
        with self._lock:
            stats, self._perf_stats = self._perf_stats, {}
            return stats

    @classmethod
    def from_env(cls, *, embed_text=None) -> "GraphMemory":
        """Build from the WALKIE_GRAPHS_* environment (config.toml defaults)."""

        def _f(name, default):
            return float(os.getenv(name, default))

        def _i(name, default):
            return int(os.getenv(name, default))

        def _b(name, default):
            return os.getenv(name, default).strip().lower() in ("1", "true", "yes")

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
            dedup_visual_k=_i("WALKIE_GRAPHS_DEDUP_VISUAL_K", "0"),
            visual_merge_max_dist_m=_f("WALKIE_GRAPHS_VISUAL_MERGE_MAX_DIST_M", "0.4"),
            sim_threshold=_f("WALKIE_GRAPHS_SIM_THRESHOLD", "1.1"),
            cross_class_sim_threshold=_f("WALKIE_GRAPHS_CROSS_CLASS_SIM_THRESHOLD", "1.5"),
            w_geo=_f("WALKIE_GRAPHS_W_GEO", "1.0"),
            w_sem=_f("WALKIE_GRAPHS_W_SEM", "1.0"),
            nn_voxel_m=_f("WALKIE_GRAPHS_NN_VOXEL_M", "0.025"),
            dbscan_enabled=_b("WALKIE_GRAPHS_DBSCAN_ENABLED", "1"),
            dbscan_eps=_f("WALKIE_GRAPHS_DBSCAN_EPS", "0.05"),
            dbscan_min_points=_i("WALKIE_GRAPHS_DBSCAN_MIN_POINTS", "10"),
            sor_k=_i("WALKIE_GRAPHS_SOR_K", "16"),
            sor_std_ratio=_f("WALKIE_GRAPHS_SOR_STD_RATIO", "2.0"),
            icp_max_dist_m=_f("WALKIE_GRAPHS_ICP_MAX_DIST_M", "0.1"),
            icp_min_fitness=_f("WALKIE_GRAPHS_ICP_MIN_FITNESS", "0.6"),
            icp_min_points=_i("WALKIE_GRAPHS_ICP_MIN_POINTS", "150"),
            icp_skip_overlap=_f("WALKIE_GRAPHS_ICP_SKIP_OVERLAP", "0.75"),
            icp_cooldown_sec=_f("WALKIE_GRAPHS_ICP_COOLDOWN_SEC", "10"),
            defer_pcd_writes=_b("WALKIE_GRAPHS_DEFER_PCD_WRITES", "1"),
            denoise_keep_min_frac=_f("WALKIE_GRAPHS_DENOISE_KEEP_MIN_FRAC", "0.5"),
            merge_overlap_thresh=_f("WALKIE_GRAPHS_MERGE_OVERLAP_THRESH", "0.7"),
            merge_visual_sim_thresh=_f("WALKIE_GRAPHS_MERGE_VISUAL_SIM_THRESH", "0.7"),
            merge_text_sim_thresh=_f("WALKIE_GRAPHS_MERGE_TEXT_SIM_THRESH", "0.7"),
            merge_radius_m=_f("WALKIE_GRAPHS_MERGE_RADIUS_M", "0.5"),
            min_obs_confirm=_i("WALKIE_GRAPHS_MIN_OBS_CONFIRM", "3"),
            require_confirmation=_b("WALKIE_GRAPHS_REQUIRE_CONFIRMATION", "1"),
            ghost_grace_sec=_f("WALKIE_GRAPHS_GHOST_GRACE_SEC", "0"),
            best_views_n=_i("WALKIE_GRAPHS_BEST_VIEWS", "0"),
            thumbnails=_b("WALKIE_GRAPHS_THUMBNAILS", "1"),
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
            "best_views_json": json.dumps(n.best_views),
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
            best_views=json.loads(md.get("best_views_json", "[]")),
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
        self._nodes[n.id] = n
        if self._batching:
            self._chroma_pending[n.id] = n  # flushed in one batched upsert on exit
        else:
            self._chroma_upsert([n])

    def _chroma_upsert(self, nodes: list[ObjectNode]) -> None:
        """Upsert a batch of nodes' embeddings + metadata into Chroma in one call."""
        if not nodes:
            return
        self._col.upsert(
            ids=[n.id for n in nodes],
            embeddings=[list(n.clip_emb if n.clip_emb else [0.0] * self.emb_dim) for n in nodes],
            metadatas=[self._metadata(n) for n in nodes],
            documents=[n.best_caption or n.class_name for n in nodes],
        )

    def _flush_chroma(self) -> None:
        with self._lock:
            pending = list(self._chroma_pending.values())
            self._chroma_pending.clear()
        if pending:
            t0 = time.perf_counter()
            self._chroma_upsert(pending)
            self._perf_add("chroma_flush", t0)

    @contextmanager
    def batch_writes(self):
        """Defer Chroma writes to a single batched upsert on exit (one per frame).

        In-memory state (``self._nodes``, point clouds, edges) stays immediately
        consistent inside the block — only the Chroma persistence/text-index write is
        batched. Safe to leave Chroma slightly behind the in-memory graph: queries are
        eventually consistent and maintenance reads the in-memory dict.
        """
        self._batching = True
        try:
            yield
        finally:
            self._batching = False
            self._flush_chroma()

    def _pcd_path(self, node_id: str) -> Path:
        return self.pcds_dir / f"{node_id}.npz"

    def _save_pcd(self, node_id: str, points: np.ndarray) -> str:
        pts = points.astype(np.float32)
        path = self._pcd_path(node_id)
        self._pcd_cache[node_id] = pts
        self._tree_cache.pop(node_id, None)  # cloud changed → KD-tree is stale
        if self.defer_pcd_writes:
            # On slow robot storage the per-sighting .npz rewrite dominates upsert
            # time, so just mark it pending — all reads come from the cache, and
            # flush_pcds persists the latest cloud once per flush cadence instead of
            # once per sighting. Crash window: clouds newer than the last flush.
            self._pcd_unflushed.add(node_id)
        else:
            # Uncompressed: clouds are small and rewritten often, so the zlib cost
            # per save outweighs the disk saved (the cache serves reads anyway).
            np.savez(path, points=pts)
        return str(path)

    def flush_pcds(self) -> int:
        """Write every pending (deferred) point cloud to its ``.npz``; returns count.

        Only the *latest* cloud per node is written no matter how many sightings
        accumulated since the last flush — that's the entire point of deferring.
        """
        with self._lock:
            pending = [
                (nid, self._pcd_cache[nid])
                for nid in self._pcd_unflushed
                if nid in self._nodes and nid in self._pcd_cache
            ]
            self._pcd_unflushed.clear()
        for nid, pts in pending:
            np.savez(self._pcd_path(nid), points=pts)
        return len(pending)

    def _stored_tree(self, node_id: str):
        """Cached ``cKDTree`` over the node's stored cloud (None when scipy/cloud absent)."""
        tree = self._tree_cache.get(node_id)
        if tree is None and _cKDTree is not None:
            pts = self.load_pcd(node_id)
            if len(pts):
                tree = _cKDTree(pts.astype(np.float64))
                self._tree_cache[node_id] = tree
        return tree

    def load_pcd(self, node_id: str) -> np.ndarray:
        """The node's world cloud — from the write-through cache, disk on a cold start."""
        cached = self._pcd_cache.get(node_id)
        if cached is not None:
            return cached
        path = self._pcd_path(node_id)
        if not path.exists():
            return np.zeros((0, 3), dtype=np.float32)
        pts = np.load(path)["points"]
        self._pcd_cache[node_id] = pts
        return pts

    def _save_thumb(self, node_id: str, crop) -> Optional[str]:
        if crop is None or not self.thumbnails:
            return None
        path = self.thumbs_dir / f"{node_id}.jpg"
        try:
            crop.convert("RGB").save(path, "JPEG", quality=85)
        except Exception:
            return None
        return str(path)

    @staticmethod
    def _free_slot(node_id: str, views: list) -> int:
        """Lowest view-slot index in 0..N not currently used by ``views``."""
        used = set()
        for _conf, p in views:
            try:
                used.add(int(Path(p).stem.rsplit("-v", 1)[1]))
            except Exception:
                pass
        k = 0
        while k in used:
            k += 1
        return k

    def _record_view(self, node: ObjectNode, conf: float, crop) -> None:
        """Keep the top ``best_views_n`` crops of a node by confidence (Tier 3 infra).

        These feed the optional multi-view LLM caption refinement. No-op when
        ``best_views_n <= 0`` (Tier 3 off) or no crop is supplied — so it costs nothing
        in the default configuration and never touches the unit-test path (crop=None).
        """
        if crop is None or self.best_views_n <= 0:
            return
        views = list(node.best_views)
        if len(views) >= self.best_views_n:
            lowest = min(views, key=lambda v: v[0])
            if conf <= lowest[0]:
                return
            views.remove(lowest)
            Path(lowest[1]).unlink(missing_ok=True)
        path = self.thumbs_dir / f"{node.id}-v{self._free_slot(node.id, views)}.jpg"
        try:
            crop.convert("RGB").save(path, "JPEG", quality=85)
        except Exception:
            return
        views.append((float(conf), str(path)))
        views.sort(key=lambda v: v[0], reverse=True)
        node.best_views = views

    # ------------------------------------------------------------------
    # Ingestion (fusion / dedup)
    # ------------------------------------------------------------------
    def upsert(self, det: Detection3D) -> ObjectNode:
        """Insert a detection as a new node, or merge it into an existing one.

        The detection's cloud is DBSCAN-denoised **once** here (largest cluster), so
        both association and the eventual insert/merge run on the clean cloud — the
        ConceptGraphs order (denoise the detection, then associate with
        ``run_dbscan=False`` on merge). Association is additive-greedy on point-cloud
        overlap + CLIP (:meth:`_associate`); when that finds no geometric match it
        falls back to the visual-K distance cascade (:meth:`_classify`), which keeps
        walkie's recovery of drifted-position re-sightings that pure overlap misses.
        """
        with self._lock:
            t0 = time.perf_counter()
            pts = self._denoise(det.points_world.astype(np.float32))
            if len(pts) == 0:
                pts = det.points_world.astype(np.float32)
            det = replace(det, points_world=pts)
            self._perf_add("denoise", t0)

            t0 = time.perf_counter()
            target, overlap = self._associate(det)
            self._perf_add("assoc", t0)
            if target is None:
                t0 = time.perf_counter()
                target = self._classify(det, self._candidates(det))
                overlap = 0.0  # matched on appearance only; alignment state unknown
                self._perf_add("classify", t0)
            if target is None:
                return self._insert(det)
            return self._merge(target, det, assoc_overlap=overlap)

    def _denoise(self, points: np.ndarray) -> np.ndarray:
        """DBSCAN-denoise a detection cloud to its largest cluster (no-op if disabled).

        Per-frame SOR already ran at deprojection (``deproject_mask(sor_k=...)``), so
        this only adds the single-view cluster cut; the periodic :meth:`denoise_nodes`
        SOR handles what *accumulates* across sightings.
        """
        if not self.dbscan_enabled or len(points) == 0:
            return points
        # subsample bounds DBSCAN's superlinear cost on dense (12k-pt) clouds — the
        # cluster verdict is computed on ≤4000 points and mapped back to full res.
        return dbscan_largest_cluster(
            points, self.dbscan_eps, self.dbscan_min_points, subsample=4000
        )

    def _associate(self, det: Detection3D) -> tuple[Optional[ObjectNode], float]:
        """ConceptGraphs additive-greedy match → ``(node, overlap)`` or ``(None, 0.0)``.

        Over nodes that pass a cheap prefilter (AABB intersection padded by the NN
        radius, OR within ``dedup_radius_m``), score
        ``w_geo·nn_ratio + w_sem·(0.5·cos + 0.5)`` and return the best candidate that
        clears its threshold. Same-class candidates need ``sim_threshold``; candidates
        of a **different** class need the stricter ``cross_class_sim_threshold`` —
        CG's association is fully class-agnostic, and the detector here flip-flops
        labels for one object ("cup" vs "mug"), which a same-class-only matcher turns
        into duplicate nodes. ``cross_class_sim_threshold = 0`` disables cross-class.

        With the default thresholds this path can only fire on real geometric overlap
        (pure visual tops out at ``w_sem`` = 1.0 < 1.1), so a re-sighting whose 3D
        estimate drifted produces no match here and flows to ``_classify`` unchanged.

        The returned ``overlap`` is the winner's ``nn_ratio`` — the fraction of the
        detection already coinciding with the stored cloud — which the merge uses to
        skip ICP when the clouds are pre-aligned.
        """
        if len(det.points_world) == 0:
            return None, 0.0
        det_c, det_mn, det_mx, _ = aabb_of(det.points_world.astype(np.float32))
        best: Optional[ObjectNode] = None
        best_sim = -1.0
        best_overlap = 0.0
        for n in self._nodes.values():
            same_class = n.class_name == det.class_name
            if not same_class and self.cross_class_sim_threshold <= 0:
                continue
            if not (
                aabb_overlap(det_mn, det_mx, n.aabb_min, n.aabb_max, pad=self.nn_voxel_m)
                or (same_class and l2(det_c, n.centroid) <= self.dedup_radius_m)
            ):
                continue
            ratio = nn_ratio(
                self.load_pcd(n.id),
                det.points_world,
                self.nn_voxel_m,
                obj_tree=self._stored_tree(n.id),  # cached per cloud version
                max_query=1024,  # a fraction is stable on ~1k samples
            )
            cos = cosine(det.clip_emb, n.clip_emb)
            sim = additive_similarity(ratio, cos, w_geo=self.w_geo, w_sem=self.w_sem)
            gate = self.sim_threshold if same_class else self.cross_class_sim_threshold
            if sim >= gate and sim > best_sim:
                best, best_sim, best_overlap = n, sim, ratio
        return best, (best_overlap if best is not None else 0.0)

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
        """Visual-similarity merge cascade (the fallback when clouds don't overlap).

        The pure-visual rule is distance-capped by ``visual_merge_max_dist_m``:
        identical twin objects (two matching chairs side by side) exceed any workable
        CLIP threshold, so high cosine alone must never bridge more distance than
        plausible pose drift — beyond the cap they are distinct objects. 0 = uncapped
        (legacy behavior, from when 3D estimates could drift by metres).
        """
        for n in candidates:
            cos = cosine(det.clip_emb, n.clip_emb)
            dist = l2(det.centroid, n.centroid)
            if cos >= self.sim_high and (
                self.visual_merge_max_dist_m <= 0 or dist <= self.visual_merge_max_dist_m
            ):
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
        self._record_view(node, det.confidence, det.crop)
        self._write_node(node)
        self._dirty.add(node_id)
        return node

    def _merge(
        self, node: ObjectNode, det: Detection3D, *, assoc_overlap: float = 0.0
    ) -> ObjectNode:
        """Fold a matched detection into ``node``, growing its cloud whenever possible.

        **Union** (vstack old + new, voxel, cap) whenever the detection's points
        geometrically overlap the stored cloud OR its centroid is near — so a *partial*
        view of a large object (one end of a bed) adds to the accumulated cloud instead
        of replacing it, and the object fills in across sightings. Only a matched
        detection that is far AND non-overlapping (a pure-visual re-sighting whose 3D
        estimate drifted into empty space, or an object that physically moved) takes the
        **replace** branch — unioning there would smear one object across two places.

        ``assoc_overlap`` is the association step's nn_ratio for this match: when most
        of the detection already coincides with the stored cloud (≥ ``icp_skip_overlap``)
        the clouds are pre-aligned and the ICP step is skipped — with a good pose that
        is the common case, so ICP only pays for itself on actually-misaligned frames.
        """
        n = node.n_obs
        det_pts = det.points_world.astype(np.float32)
        _, det_mn, det_mx, _ = aabb_of(det_pts)
        union = (
            l2(det.centroid, node.centroid) <= self.dedup_radius_m
            or aabb_overlap(det_mn, det_mx, node.aabb_min, node.aabb_max, pad=self.nn_voxel_m)
        )

        if union:
            stored = self.load_pcd(node.id)
            # Residual camera-pose error lands each sighting a few cm off; ICP-align
            # the new cloud to the stored one so the union sharpens the shape instead
            # of double-exposing it. ICP is the most expensive step in the whole
            # upsert, so it's doubly gated: a per-node COOLDOWN (pose error varies
            # slowly — realigning the same object every tick is pure cost), and an
            # overlap check (reuse the association's ratio, or measure it with the
            # cached tree) so pre-aligned clouds never pay at all.
            on_cooldown = (
                self.icp_cooldown_sec > 0
                and (det.ts - self._icp_last.get(node.id, -1e18)) < self.icp_cooldown_sec
            )
            if self.icp_max_dist_m > 0 and not on_cooldown:
                t0 = time.perf_counter()
                ratio = assoc_overlap
                if ratio <= 0.0:
                    ratio = nn_ratio(
                        stored,
                        det_pts,
                        self.nn_voxel_m,
                        obj_tree=self._stored_tree(node.id),
                        max_query=1024,
                    )
                if ratio < self.icp_skip_overlap:
                    det_pts, _fit = icp_align(
                        det_pts,
                        stored,
                        self.icp_max_dist_m,
                        min_fitness=self.icp_min_fitness,
                        min_points=self.icp_min_points,
                    )
                self._icp_last[node.id] = det.ts
                self._perf_add("icp", t0)
            t0 = time.perf_counter()
            merged = np.vstack([stored, det_pts])
            merged = _voxel(merged, self.voxel_m)
            if len(merged) > self.max_points_per_obj:
                idx = np.linspace(0, len(merged) - 1, self.max_points_per_obj).astype(int)
                merged = merged[idx]
            centroid, mn, mx, ext = aabb_of(merged)
            node.centroid, node.aabb_min, node.aabb_max, node.extent = (
                centroid, mn, mx, ext
            )
            self._perf_add("fuse", t0)
            t0 = time.perf_counter()
            node.pcd_ref = self._save_pcd(node.id, merged)
            self._perf_add("save_pcd", t0)
            node.conf = (node.conf * n + float(det.confidence)) / (n + 1)
        else:
            # Drifted/moved re-sighting: don't average two far positions into empty
            # space — keep the higher-confidence geometry.
            if det.confidence > node.conf:
                centroid, mn, mx, ext = aabb_of(det_pts)
                node.centroid, node.aabb_min, node.aabb_max, node.extent = (
                    centroid, mn, mx, ext
                )
                node.pcd_ref = self._save_pcd(node.id, det_pts)
            node.conf = max(node.conf, float(det.confidence))

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
            t0 = time.perf_counter()
            node.frame_ref = self._save_thumb(node.id, det.crop)
            self._record_view(node, det.confidence, det.crop)
            self._perf_add("thumb", t0)
        self._write_node(node)
        if union:
            self._dirty.add(node.id)  # cloud grew; the replace branch swaps, never accretes
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
            # Preserve LLM-inferred edges (predicate "llm:*") across a geometric
            # rebuild — they're computed on a separate cadence and must not be wiped.
            llm = [
                (u, v, k, data.get("weight", 1.0))
                for u, v, k, data in self._graph.edges(keys=True, data=True)
                if str(k).startswith("llm:")
            ]
            self._graph.clear()
            self._graph.add_nodes_from(self._nodes.keys())
            for r in rels:
                self._graph.add_edge(r.src_id, r.dst_id, key=r.predicate, weight=r.weight)
            for u, v, k, w in llm:
                if u in self._nodes and v in self._nodes:
                    self._graph.add_edge(u, v, key=k, weight=w)
        self._persist_edges()

    def _persist_edges(self) -> None:
        """Write the full current edge set (geometric + LLM) to JSON atomically."""
        if self.edges_path is None or self._graph is None:
            return
        data = [
            {"src": u, "dst": v, "predicate": k, "weight": d.get("weight", 1.0)}
            for u, v, k, d in self._graph.edges(keys=True, data=True)
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

    def all_relations(self) -> list[Relation]:
        with self._lock:
            if self._graph is None:
                return []
            return [
                Relation(u, v, k, data.get("weight", 1.0))
                for u, v, k, data in self._graph.edges(keys=True, data=True)
            ]

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
            hits = self._confirmed(self._spatial_filter(hits, near, radius))
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
        hits = self._confirmed(self._spatial_filter(hits, near, radius))
        return hits[:k]

    def _confirmed(self, nodes: list[ObjectNode]) -> list[ObjectNode]:
        """Drop provisional nodes (``n_obs < min_obs_confirm``) when confirmation is on.

        ConceptGraphs filters objects seen fewer than ``obj_min_detections`` times to
        keep one-frame false positives out of the map; we hide rather than delete, so a
        re-sighting can still promote a provisional node to confirmed. ``count()`` and
        the visualizer still see everything.
        """
        if not self.require_confirmation:
            return nodes
        return [n for n in nodes if n.n_obs >= self.min_obs_confirm]

    def _spatial_filter(self, nodes, near, radius):
        if near is None or radius is None:
            return nodes
        return [n for n in nodes if l2(n.centroid, near) <= radius]

    def query_near(self, center, radius: float) -> list[ObjectNode]:
        """Objects within ``radius`` of ``center`` (2D center → horizontal distance)."""
        with self._lock:
            out = [(l2(n.centroid, center), n) for n in self._confirmed(list(self._nodes.values()))]
            out = [(d, n) for d, n in out if d <= radius]
            out.sort(key=lambda s: s[0])
            return [n for _, n in out]

    def recently_seen(self, limit: int = 5) -> list[ObjectNode]:
        with self._lock:
            nodes = sorted(
                self._confirmed(list(self._nodes.values())),
                key=lambda n: n.last_seen_ts,
                reverse=True,
            )
            return nodes[:limit]

    def all_objects(self) -> list[ObjectNode]:
        with self._lock:
            return self._confirmed(list(self._nodes.values()))

    def get(self, node_id: str) -> Optional[ObjectNode]:
        with self._lock:
            return self._nodes.get(node_id)

    def count(self) -> int:
        with self._lock:
            return len(self._nodes)

    def to_text_description(self) -> str:
        """Plain-text dump of objects + relations, for LLM task planning."""
        with self._lock:
            nodes = sorted(
                self._confirmed(list(self._nodes.values())),
                key=lambda n: n.last_seen_ts,
                reverse=True,
            )
            shown = {n.id for n in nodes}
            lines = [f"Objects ({len(nodes)}):"]
            for n in nodes:
                cap = f' "{n.best_caption}"' if n.best_caption else ""
                x, y, z = n.centroid
                lines.append(
                    f" [{n.id}] {n.class_name}{cap} at ({x:.2f}, {y:.2f}, {z:.2f}) "
                    f"seen {n.n_obs}x"
                )
            rels = self._graph.edges(keys=True) if self._graph is not None else []
            # Only show edges between objects that made the (confirmed) object list.
            rels = [(u, v, k) for u, v, k in rels if u in shown and v in shown]
            if rels:
                lines.append("Relations:")
                for u, v, kpred in rels:
                    cu = self._nodes[u].class_name if u in self._nodes else u
                    cv = self._nodes[v].class_name if v in self._nodes else v
                    lines.append(f" {cu} [{u}] {kpred} {cv} [{v}]")
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Maintenance (ConceptGraphs periodic post-processing analogs)
    # ------------------------------------------------------------------
    def denoise_nodes(self) -> int:
        """Clean the accumulated clouds of nodes that changed since the last pass.

        ConceptGraphs' ``denoise_objects`` analog, adapted for accumulated multi-view
        clouds. Two stages, both cluster-preserving (a cloud built from disjoint
        partial views — the two ends of a bed — is legitimately multi-cluster, and the
        largest-cluster keep CG uses would truncate it):

        1. **Statistical outlier removal** — erases the fuzzy halo accumulation
           collects (each frame's few surviving edge artifacts); local-density based,
           so dense view clusters survive.
        2. **Noise-only DBSCAN** — drops isolated scatter that forms no cluster.

        Only *dirty* nodes (changed since the last call) are touched, and a node is
        skipped when denoising would drop more than ``denoise_keep_min_frac`` of its
        points (a big drop means real structure, not noise). Returns the number changed.
        """
        if not (self.dbscan_enabled or self.sor_k > 0):
            return 0
        with self._lock:
            dirty = [nid for nid in self._dirty if nid in self._nodes]
            self._dirty.clear()
        changed = 0
        for nid in dirty:
            with self._lock:
                if nid not in self._nodes:
                    continue
                pts = self.load_pcd(nid)
            if len(pts) < self.dbscan_min_points:
                continue
            kept = pts
            if self.sor_k > 0:
                kept = statistical_outlier_removal(kept, self.sor_k, self.sor_std_ratio)
            if self.dbscan_enabled:
                kept = dbscan_remove_noise(kept, self.dbscan_eps, self.dbscan_min_points)
            if len(kept) >= len(pts):  # nothing removed
                continue
            if len(kept) < self.denoise_keep_min_frac * len(pts):
                continue  # would cut too deep → treat as a spread object, leave it
            with self._lock:
                node = self._nodes.get(nid)
                if node is None:
                    continue
                centroid, mn, mx, ext = aabb_of(kept)
                node.centroid, node.aabb_min, node.aabb_max, node.extent = (
                    centroid, mn, mx, ext
                )
                node.pcd_ref = self._save_pcd(nid, kept)
                self._write_node(node)
                changed += 1
        return changed

    def merge_overlapping_nodes(self) -> int:
        """Fuse nodes that became duplicates over time (ConceptGraphs ``merge_overlap_objects``).

        Two same-class nodes within ``merge_radius_m`` whose clouds overlap by more than
        ``merge_overlap_thresh`` (symmetric ``nn_ratio``) **and** whose CLIP embeddings
        agree above ``merge_visual_sim_thresh`` are merged (highest ``n_obs`` kept). This
        recovers the case the incremental matcher can't: one object first mapped from two
        sides as two nodes, later seen to physically coincide. The expensive overlap math
        runs on a cloud snapshot taken outside the short mutation critical section.

        When two clouds clear the identity guard (class + CLIP) but their raw overlap is
        low purely because a few-cm pose offset has shifted one off the other, the pair
        is ICP-aligned and the overlap re-tested on the aligned clouds — so duplicates
        that drifted slightly apart still fuse instead of accumulating as ghosts.
        """
        # Lightweight snapshot under the lock (no disk I/O): the candidate prefilter
        # needs only centroids/AABBs/embeddings, so the lock is held for microseconds.
        with self._lock:
            snap = [
                (n.id, n.class_name, n.centroid, n.aabb_min, n.aabb_max, list(n.clip_emb))
                for n in self._nodes.values()
            ]
        if len(snap) < 2:
            return 0
        # Candidate pairs: centroids close (KD-tree pair query), AABBs touch, and the
        # classes either match or cross-class merging is enabled (the detector's labels
        # flip-flop, so one object can be stored under two class names) — all cheap
        # prefilters before any nn_ratio (the O(points) part), mirroring CG's
        # bbox-IoU>0 gate. CG itself merges with no class constraint at all.
        cand: list[tuple[tuple, tuple]] = []
        for i, j in pairs_within([s[2] for s in snap], self.merge_radius_m):
            a, b = snap[i], snap[j]
            if a[1] != b[1] and self.cross_class_sim_threshold <= 0:
                continue
            if aabb_overlap(a[3], a[4], b[3], b[4], pad=self.nn_voxel_m):
                cand.append((a, b))
        if not cand:
            return 0
        # Load only the clouds that survived the prefilter (outside the lock).
        need = {nid for a, b in cand for nid in (a[0], b[0])}
        clouds = {nid: self.load_pcd(nid) for nid in need}
        pairs: list[tuple[float, str, str]] = []
        for a, b in cand:
            # Identity guard FIRST (cheap, and it decides whether the ICP rescue below
            # is even allowed): same-class pairs may merge on geometry alone, but any
            # pair with embeddings — and every cross-class pair — must clear the CLIP
            # gate, since the only thing overriding two different class labels is strong
            # visual agreement.
            if a[5] and b[5]:
                if cosine(a[5], b[5]) < self.merge_visual_sim_thresh:
                    continue
            elif a[1] != b[1]:
                continue
            ca, cb = clouds[a[0]], clouds[b[0]]
            overlap = nn_ratio_symmetric(ca, cb, self.nn_voxel_m)
            if overlap <= self.merge_overlap_thresh and self.icp_max_dist_m > 0:
                # Raw overlap is measured pre-alignment at NN_VOXEL_M (2.5 cm), so a
                # mere few-cm pose offset between two sightings of ONE object reads as
                # near-zero overlap and the pair never reaches _merge_nodes (the only
                # place ICP runs) — the classic "shifted-but-not-fused" duplicate.
                # Identity is already vouched for by class+CLIP above, so align the pair
                # and re-test overlap on the aligned clouds; if they truly are one
                # surface the offset collapses and the test now passes. Distinct objects
                # don't snap together (ICP fitness stays low / surfaces stay apart), so
                # this rescues drift without loosening the threshold. _merge_nodes
                # re-aligns when it actually fuses, so the stored cloud is corrected too.
                aligned, fit = icp_align(
                    cb, ca, self.icp_max_dist_m,
                    min_fitness=self.icp_min_fitness, min_points=self.icp_min_points,
                )
                if fit >= self.icp_min_fitness:
                    overlap = nn_ratio_symmetric(ca, aligned, self.nn_voxel_m)
            if overlap <= self.merge_overlap_thresh:
                continue
            pairs.append((overlap, a[0], b[0]))
        pairs.sort(reverse=True)  # strongest overlap first
        merged = 0
        gone: set[str] = set()
        with self._lock:
            for overlap, aid, bid in pairs:
                if aid in gone or bid in gone:
                    continue
                a, b = self._nodes.get(aid), self._nodes.get(bid)
                if a is None or b is None:
                    continue
                keep, drop = (a, b) if a.n_obs >= b.n_obs else (b, a)
                self._merge_nodes(keep, drop, overlap=overlap)
                gone.add(drop.id)
                merged += 1
        return merged

    def _merge_nodes(self, keep: ObjectNode, drop: ObjectNode, *, overlap: float = 0.0) -> None:
        """Fold node ``drop`` into ``keep`` in place (clouds, CLIP, captions), delete ``drop``.

        ``overlap`` is the caller's symmetric nn_ratio for the pair; when the clouds
        already coincide (≥ ``icp_skip_overlap``) the ICP step is skipped.
        """
        kp, dp = self.load_pcd(keep.id), self.load_pcd(drop.id)
        if self.icp_max_dist_m > 0 and overlap < self.icp_skip_overlap and len(kp) and len(dp):
            # Two nodes of one object usually split *because* of pose error; align the
            # dropped cloud onto the kept one so the fusion doesn't double-expose.
            dp, _fit = icp_align(
                dp,
                kp,
                self.icp_max_dist_m,
                min_fitness=self.icp_min_fitness,
                min_points=self.icp_min_points,
            )
        clouds = [c for c in (kp, dp) if len(c)]
        merged = np.vstack(clouds) if clouds else np.zeros((0, 3), dtype=np.float32)
        merged = _voxel(merged, self.voxel_m)
        if len(merged) > self.max_points_per_obj:
            idx = np.linspace(0, len(merged) - 1, self.max_points_per_obj).astype(int)
            merged = merged[idx]
        if len(merged):
            centroid, mn, mx, ext = aabb_of(merged)
            keep.centroid, keep.aabb_min, keep.aabb_max, keep.extent = centroid, mn, mx, ext
        nk, nd = keep.n_obs, drop.n_obs
        if keep.clip_emb and drop.clip_emb:
            blended = np.asarray(keep.clip_emb, float) * nk + np.asarray(drop.clip_emb, float) * nd
            keep.clip_emb = _normalize(blended / (nk + nd))
        elif drop.clip_emb and not keep.clip_emb:
            keep.clip_emb = drop.clip_emb
        for cap in drop.captions:
            if cap and cap not in keep.captions:
                keep.captions.append(cap)
        if len(drop.best_caption) > len(keep.best_caption):
            keep.best_caption = drop.best_caption
        keep.conf = max(keep.conf, drop.conf)
        keep.n_obs = nk + nd
        keep.first_seen_ts = min(keep.first_seen_ts, drop.first_seen_ts)
        keep.last_seen_ts = max(keep.last_seen_ts, drop.last_seen_ts)
        keep.pcd_ref = self._save_pcd(keep.id, merged)
        self._write_node(keep)
        self._dirty.add(keep.id)
        self._delete(drop.id)

    def evict_stale_provisional(self, now_ts: float) -> int:
        """Delete provisional nodes (``n_obs < min_obs_confirm``) unseen past the grace window.

        Off unless ``ghost_grace_sec > 0`` and confirmation is in force. This is the
        destructive complement to query-time hiding: a transient false positive that is
        never re-confirmed is eventually removed instead of lingering forever.
        """
        if self.ghost_grace_sec <= 0 or self.min_obs_confirm <= 1:
            return 0
        with self._lock:
            doomed = [
                n.id
                for n in self._nodes.values()
                if n.n_obs < self.min_obs_confirm
                and (now_ts - n.last_seen_ts) > self.ghost_grace_sec
            ]
            for nid in doomed:
                self._delete(nid)
            return len(doomed)

    # ------------------------------------------------------------------
    # Semantic refinement (Tier 3 — optional LLM; off unless a model is passed)
    # ------------------------------------------------------------------
    def refine_captions(self, model, *, limit: Optional[int] = None, use_images: bool = False) -> int:
        """Condense each object's accumulated view captions into one coherent caption.

        ConceptGraphs' node-captioning stage: an LLM summarizes the per-view rough
        captions into a single object tag. Best-effort and **outside the lock** for the
        model call — snapshot what each node needs, invoke the (network-bound) model
        unlocked, then write the result back under the lock with an existence check.
        Returns the number of captions updated.
        """
        if model is None:
            return 0
        with self._lock:
            cands = [n for n in self._nodes.values() if n.captions]
            cands.sort(key=lambda n: n.last_seen_ts, reverse=True)
            if limit:
                cands = cands[:limit]
            snap = [
                (n.id, n.class_name, list(n.captions), list(n.best_views) if use_images else [])
                for n in cands
            ]
        refined = 0
        for nid, cls, captions, views in snap:
            try:
                text = self._summarize_caption(model, cls, captions, views, use_images)
            except Exception:
                continue
            if not text:
                continue
            with self._lock:
                node = self._nodes.get(nid)
                if node is None:
                    continue
                node.best_caption = text
                if text not in node.captions:
                    node.captions.append(text)
                self._write_node(node)
            refined += 1
        return refined

    @staticmethod
    def _summarize_caption(model, cls, captions, views, use_images) -> str:
        prompt = (
            "You label one physical object for a robot's spatial memory. The detector "
            f"called it '{cls}'. Independent view captions:\n"
            + "\n".join(f"- {c}" for c in captions)
            + "\nReply with ONE concise noun phrase (max 8 words) naming the object."
        )
        if use_images and views:
            from base64 import b64encode

            from langchain_core.messages import HumanMessage

            content = [{"type": "text", "text": prompt}]
            for _conf, path in views[:4]:
                try:
                    raw = Path(path).read_bytes()
                except Exception:  # noqa: BLE001
                    continue
                b64 = b64encode(raw).decode()
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )
            resp = model.invoke([HumanMessage(content=content)])
        else:
            resp = model.invoke(prompt)
        return (getattr(resp, "content", "") or "").strip()

    def infer_edges_llm(self, model, *, max_pairs: Optional[int] = None) -> int:
        """Label spatial relations between nearby objects with an LLM (CG scene-graph step).

        Builds a minimum spanning tree over centroid-proximal nodes (so only the most
        relevant adjacencies are queried — the paper's MST pruning), asks the model for
        each MST edge's relation, and stores accepted ones as ``llm:<label>`` edges —
        kept separate from the geometric near/on/above/inside edges, which stay primary
        and survive every relation rebuild. Off unless a model is supplied. Returns the
        number of LLM edges written.
        """
        if model is None or nx is None:
            return 0
        with self._lock:
            nodes = list(self._nodes.values())
            by_id = {n.id: n for n in nodes}
        if len(nodes) < 2:
            return 0
        g = nx.Graph()
        g.add_nodes_from(by_id.keys())
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                d = l2(a.centroid, b.centroid)
                if d <= self.relation_max_dist:
                    g.add_edge(a.id, b.id, weight=d)
        if g.number_of_edges() == 0:
            return 0
        edges = list(nx.minimum_spanning_tree(g).edges())
        if max_pairs:
            edges = edges[:max_pairs]
        new_edges = []
        for u, v in edges:
            a, b = by_id.get(u), by_id.get(v)
            if a is None or b is None:
                continue
            try:
                label = self._infer_relation(model, a, b)
            except Exception:  # noqa: BLE001
                continue
            label = (label or "").strip().lower()
            if label and label not in ("none", "none of these", "unrelated"):
                new_edges.append((u, v, f"llm:{label}"))
        with self._lock:
            if self._graph is None:
                return 0
            stale = [
                (u, v, k)
                for u, v, k in self._graph.edges(keys=True)
                if str(k).startswith("llm:")
            ]
            for u, v, k in stale:
                self._graph.remove_edge(u, v, key=k)
            written = 0
            for u, v, k in new_edges:
                if u in self._nodes and v in self._nodes:
                    self._graph.add_edge(u, v, key=k, weight=1.0)
                    written += 1
            self._persist_edges()
        return written

    @staticmethod
    def _infer_relation(model, a: ObjectNode, b: ObjectNode) -> str:
        def fmt(n: ObjectNode) -> str:
            c = tuple(round(x, 2) for x in n.centroid)
            e = tuple(round(x, 2) for x in n.extent)
            return f"{n.best_caption or n.class_name} (center {c}, size {e})"

        prompt = (
            "Two objects in a room (metres, z is up). State the spatial relationship of "
            "A to B using ONE of: on, under, in, next to, none.\n"
            f"A = {fmt(a)}\nB = {fmt(b)}\nReply with only the relationship."
        )
        resp = model.invoke(prompt)
        return getattr(resp, "content", "") or ""

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
        node = self._nodes.pop(node_id, None)
        try:
            self._col.delete(ids=[node_id])
        except Exception:
            pass
        self._pcd_path(node_id).unlink(missing_ok=True)
        (self.thumbs_dir / f"{node_id}.jpg").unlink(missing_ok=True)
        for _conf, p in (node.best_views if node else []):
            Path(p).unlink(missing_ok=True)
        self._pcd_cache.pop(node_id, None)
        self._chroma_pending.pop(node_id, None)
        self._pcd_unflushed.discard(node_id)
        self._tree_cache.pop(node_id, None)
        self._icp_last.pop(node_id, None)
        self._dirty.discard(node_id)
        if self._graph is not None and node_id in self._graph:
            self._graph.remove_node(node_id)

    def clear(self) -> None:
        with self._lock:
            for node_id in list(self._nodes.keys()):
                self._delete(node_id)
            self._dirty.clear()
            self._pcd_cache.clear()
            self._chroma_pending.clear()
            self._pcd_unflushed.clear()
            self._tree_cache.clear()
            self._icp_last.clear()
            if self._graph is not None:
                self._graph.clear()
            self._persist_edges()


# ---------------------------------------------------------------------------
def _normalize(vec) -> list[float]:
    v = np.asarray(vec, dtype=float)
    if v.size == 0:
        return []
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()


# The point-cloud voxel grid is shared with the deprojection path — one fast
# (bincount-based) implementation, used here for the merge/fuse downsampling too.
_voxel = voxel_downsample
