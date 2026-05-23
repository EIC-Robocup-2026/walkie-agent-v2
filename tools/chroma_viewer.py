"""Read-only web viewer for this project's ChromaDB stores.

Walkie writes to more than one Chroma directory:

  * ``chroma_db``        — legacy ``WalkieVectorDB``: ``objects`` (and older
                           ``people`` / ``scenes``) collections.
  * ``chroma_db_scene``  — the CLIP ``SceneStore``: a ``scene_entries``
                           collection with image embeddings + archived
                           JPEG frames under ``frames/``.

Rather than hardcode those schemas, this viewer is *generic*: it opens each
directory, enumerates every collection, and renders rows from whatever
metadata they carry. Frame thumbnails appear automatically when a record has
a ``frame_ref`` pointing at a readable image, and a top-down map appears when
records carry ``x``/``y``/``z`` positions.

It is strictly **read-only** — no add/update/delete routes exist — so it is
safe to run against a directory the robot is actively writing to (SQLite
handles the concurrent reads).

Run it::

    uv run python -m tools.chroma_viewer
    uv run python -m tools.chroma_viewer --dirs chroma_db,chroma_db_scene --port 8500

then open http://localhost:8500.

Environment overrides (CLI flags win):
    CHROMA_VIEWER_DIRS        comma-separated dirs (default: chroma_db,chroma_db_scene)
    CHROMA_VIEWER_PORT        port                 (default: 8500)
    CHROMA_VIEWER_REFRESH_SEC initial auto-refresh (default: 5; 0 = off)
    SCENE_FRAMES_DIR          extra image root     (default: frames)
    WALKIE_AI_BASE_URL        for CLIP semantic search on scene_entries
"""

from __future__ import annotations

import argparse
import hashlib
import html
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from flask import Flask, Response, abort, render_template_string, request, send_file

# Pick up .env (CHROMA_VIEWER_*, SCENE_FRAMES_DIR, WALKIE_AI_BASE_URL) at import
# time, before the module-level getenv reads below, mirroring main.py.
load_dotenv()

PAGE_SIZE = 50
SEARCH_SCAN_CAP = 5000   # max rows scanned for a substring search
BROWSE_FETCH_CAP = 3000  # max rows pulled for sort/filter/map before paginating
DEFAULT_REFRESH_SEC = os.getenv("CHROMA_VIEWER_REFRESH_SEC", "5")

# Metadata keys promoted to table columns, in display order, when present.
# ``id`` / ``frame`` / ``document`` / ``position`` / ``distance`` are handled
# separately. Everything else still shows on the per-record detail page.
# ``embedding_model`` is intentionally omitted — it's always the same model,
# so it only clutters the table. It still shows on the per-record detail page.
TABLE_KEYS = [
    "class_name",
    "confidence",
    "position_conf",
    "sightings",
    "last_seen_ts",
]
SORTABLE = {
    "id", "class_name", "confidence", "position_conf",
    "sightings", "last_seen_ts", "first_seen_ts", "distance", "position",
}

app = Flask(__name__)


# --------------------------------------------------------------------------- #
# Chroma access (read-only, cached per directory)
# --------------------------------------------------------------------------- #


class Store:
    """One Chroma persistent directory and the collections inside it."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.path = Path(directory).resolve()
        self.error: Optional[str] = None
        self._client: Optional[chromadb.api.ClientAPI] = None
        if not self.path.exists():
            self.error = "directory does not exist"
            return
        try:
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as e:  # noqa: BLE001 — surface, don't crash the page
            self.error = repr(e)

    def collections(self) -> list[tuple[str, int]]:
        if self._client is None:
            return []
        out = []
        for c in self._client.list_collections():
            try:
                out.append((c.name, c.count()))
            except Exception:  # noqa: BLE001
                out.append((c.name, -1))
        return sorted(out)

    def collection(self, name: str):
        if self._client is None:
            abort(404)
        try:
            return self._client.get_collection(name)
        except Exception:  # noqa: BLE001
            abort(404)


# Built once at startup; index in URLs keeps directory paths out of the path.
STORES: list[Store] = []
FRAME_ROOTS: list[Path] = []


def _build_stores(dirs: list[str]) -> None:
    STORES.clear()
    STORES.extend(Store(d) for d in dirs)


def _build_frame_roots(dirs: list[str], frames_dir: str) -> None:
    """Directories under which a ``frame_ref`` is allowed to resolve.

    Guards the ``/frame`` route against path traversal: a ref is served only
    if, once resolved, it lives under one of these roots.
    """
    roots = {Path.cwd().resolve()}
    if frames_dir:
        roots.add(Path(frames_dir).resolve())
    for d in dirs:
        roots.add(Path(d).resolve().parent)
    FRAME_ROOTS.clear()
    FRAME_ROOTS.extend(roots)


# --------------------------------------------------------------------------- #
# Row / embedding helpers
# --------------------------------------------------------------------------- #


def _emb_list(result: dict, n: int) -> list:
    """``embeddings`` may be a numpy array, a list, or absent. Normalize."""
    embs = result.get("embeddings")
    if embs is None:
        return [None] * n
    try:
        if len(embs) == 0:
            return [None] * n
    except TypeError:
        return [None] * n
    return list(embs)


def _rows(result: dict, with_emb: bool = False) -> list[dict[str, Any]]:
    ids = result.get("ids") or []
    metas = result.get("metadatas") or [{}] * len(ids)
    docs = result.get("documents") or [""] * len(ids)
    embs = _emb_list(result, len(ids)) if with_emb else [None] * len(ids)
    rows = []
    for rid, meta, doc, emb in zip(ids, metas, docs, embs):
        rows.append({"id": rid, "meta": dict(meta or {}), "doc": doc or "", "emb": emb})
    return rows


def _position(meta: dict) -> Optional[tuple[float, float, float]]:
    if all(k in meta for k in ("x", "y", "z")):
        try:
            return (float(meta["x"]), float(meta["y"]), float(meta["z"]))
        except (TypeError, ValueError):
            return None
    return None


def _frame_url(meta: dict) -> Optional[str]:
    ref = meta.get("frame_ref")
    if not ref:
        return None
    return "/frame?ref=" + quote(str(ref))


def _emb_stats(vec) -> Optional[dict[str, Any]]:
    if vec is None:
        return None
    try:
        vals = [float(v) for v in vec]
    except (TypeError, ValueError):
        return None
    if not vals:
        return None
    norm = math.sqrt(sum(v * v for v in vals))
    return {"dim": len(vals), "norm": norm, "min": min(vals),
            "max": max(vals), "head": vals[:8], "vals": vals}


# --------------------------------------------------------------------------- #
# Formatting helpers (all HTML-escape their inputs)
# --------------------------------------------------------------------------- #


def e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _is_ts_key(k: str) -> bool:
    return k == "ts" or k.endswith("_ts")


def _rel_time(ts: Any) -> Optional[tuple[str, str]]:
    """Return (relative, absolute) for an epoch-seconds value, or None."""
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    d = max(0.0, time.time() - ts)
    if d < 60:
        rel = f"{int(d)}s ago"
    elif d < 3600:
        rel = f"{int(d // 60)}m ago"
    elif d < 86400:
        rel = f"{int(d // 3600)}h ago"
    else:
        rel = f"{int(d // 86400)}d ago"
    return rel, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _fmt_num(v: Any) -> Optional[str]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:.4g}"
    return None


def _hue(name: Any) -> int:
    return int(hashlib.md5(str(name).encode()).hexdigest(), 16) % 360


def _class_badge(name: Any) -> str:
    return f"<span class='badge' style='--h:{_hue(name)}'>{e(name)}</span>"


def _bar(frac: float, text: str, hue: int = 211) -> str:
    frac = max(0.0, min(1.0, frac))
    return (
        f"<span class='bar' title='{e(text)}'>"
        f"<span class='bar-fill' style='width:{frac * 100:.0f}%;--h:{hue}'></span>"
        f"<span class='bar-txt'>{e(text)}</span></span>"
    )


def _ts_html(val: Any) -> str:
    rt = _rel_time(val)
    if not rt:
        return e(val)
    return f"<span title='{e(rt[1])}'>{e(rt[0])}</span>"


# --------------------------------------------------------------------------- #
# URL building
# --------------------------------------------------------------------------- #


def _url(path: str, carry: bool = False, **overrides: Any) -> str:
    params: dict[str, str] = {}
    if carry:
        params = {k: v for k, v in request.args.items() if k != "_partial"}
    for k, v in overrides.items():
        if v is None:
            params.pop(k, None)
        else:
            params[k] = str(v)
    qs = urlencode(params)
    return f"{path}?{qs}" if qs else path


def _browse_path(di: int, coll: str) -> str:
    return f"/c/{di}/{quote(coll)}"


def _search_path(di: int, coll: str) -> str:
    return f"/c/{di}/{quote(coll)}/search"


def _record_url(di: int, coll: str, rid: str) -> str:
    return f"/c/{di}/{quote(coll)}/r/{quote(str(rid))}"


# --------------------------------------------------------------------------- #
# Sorting
# --------------------------------------------------------------------------- #


def _sort_key(key: str):
    def k(r: dict):
        if key == "id":
            v = r["id"]
        elif key == "distance":
            v = r.get("distance")
        elif key == "position":
            pos = _position(r["meta"])
            v = pos[0] if pos else None
        else:
            v = r["meta"].get(key)
        if v is None or v == "":
            return (1, 0.0, "")
        try:
            return (0, float(v), "")
        except (TypeError, ValueError):
            return (0, 0.0, str(v).lower())

    return k


def _sort_rows(rows: list[dict], key: str, direction: str) -> None:
    rows.sort(key=_sort_key(key), reverse=(direction == "desc"))


def _default_sort(rows: list[dict]) -> tuple[str, str]:
    keys: set[str] = set()
    for r in rows[:64]:
        keys |= set(r["meta"].keys())
    if "last_seen_ts" in keys:
        return ("last_seen_ts", "desc")
    if "first_seen_ts" in keys:
        return ("first_seen_ts", "desc")
    return ("id", "asc")


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def _substring_search(coll, q: str) -> list[dict[str, Any]]:
    ql = q.lower()
    res = coll.get(include=["metadatas", "documents"], limit=SEARCH_SCAN_CAP)
    hits = []
    for r in _rows(res):
        hay = (r["doc"] + " " + " ".join(str(v) for v in r["meta"].values())).lower()
        if ql in hay:
            hits.append(r)
    return hits


def _semantic_search(coll, q: str, rows_sample: list[dict]) -> list[dict[str, Any]]:
    """Best-effort vector search. Raises on any failure so the caller can
    fall back to substring with a visible warning."""
    is_clip = any(
        "clip" in str(r["meta"].get("embedding_model", "")).lower()
        or int(r["meta"].get("embedding_dim", 0) or 0) == 512
        for r in rows_sample
    )
    if is_clip:
        from client.image_embed import ImageEmbedClient
        from perception import RemoteCLIPEmbedder

        base = os.getenv("WALKIE_AI_BASE_URL", "http://localhost:5000")
        embedder = RemoteCLIPEmbedder(ImageEmbedClient(base_url=base))
        vec = embedder.embed_text(q)
        res = coll.query(
            query_embeddings=[vec],
            n_results=min(PAGE_SIZE, max(1, coll.count())),
            include=["metadatas", "documents", "distances"],
        )
    else:
        res = coll.query(
            query_texts=[q],
            n_results=min(PAGE_SIZE, max(1, coll.count())),
            include=["metadatas", "documents", "distances"],
        )
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    docs = res["documents"][0]
    dists = (res.get("distances") or [[None] * len(ids)])[0]
    out = []
    for rid, meta, doc, dist in zip(ids, metas, docs, dists):
        out.append({"id": rid, "meta": dict(meta or {}), "doc": doc or "",
                    "emb": None, "distance": dist})
    return out


# --------------------------------------------------------------------------- #
# Rendering: table, map, toolbar
# --------------------------------------------------------------------------- #


def _cell(di: int, coll: str, key: str, r: dict) -> str:
    meta = r["meta"]
    if key == "id":
        rid = str(r["id"])
        return (
            f"<td class='c-id'><a href='{_record_url(di, coll, rid)}'>{e(rid)}</a>"
            f"<button class='copy' data-copy='{e(rid)}' title='Copy id'>⧉</button></td>"
        )
    if key == "frame":
        furl = _frame_url(meta)
        return f"<td><img class='thumb' data-zoom src='{furl}' loading='lazy'></td>" if furl else "<td></td>"
    if key == "document":
        return f"<td class='c-doc' title='{e(r['doc'])}'>{e(r['doc'])}</td>"
    if key == "class_name":
        return f"<td>{_class_badge(meta.get('class_name', ''))}</td>" if meta.get("class_name") else "<td></td>"
    if key == "position":
        pos = _position(meta)
        if not pos:
            return "<td></td>"
        return ("<td class='c-pos'>"
                f"<code>{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}</code></td>")
    if key in ("confidence", "position_conf"):
        try:
            f = float(meta[key])
            return f"<td>{_bar(f, f'{f:.3f}')}</td>"
        except (KeyError, TypeError, ValueError):
            return "<td></td>"
    if key == "distance":
        d = r.get("distance")
        if d is None:
            return "<td></td>"
        # cosine distance ~[0,2]; nearer = fuller bar
        return f"<td>{_bar(1 - min(float(d), 2.0) / 2.0, f'{float(d):.4f}', hue=152)}</td>"
    if _is_ts_key(key):
        return f"<td>{_ts_html(meta.get(key))}</td>"
    if key == "embedding_model":
        return f"<td class='muted small'>{e(meta.get(key, ''))}</td>"
    n = _fmt_num(meta.get(key))
    return f"<td>{e(n) if n is not None else e(meta.get(key, ''))}</td>"


def _columns(rows: list[dict]) -> list[str]:
    present: set[str] = set()
    has_pos = has_frame = has_dist = False
    for r in rows:
        present |= set(r["meta"].keys())
        has_pos = has_pos or _position(r["meta"]) is not None
        has_frame = has_frame or _frame_url(r["meta"]) is not None
        has_dist = has_dist or ("distance" in r and r["distance"] is not None)
    cols = ["id"]
    if has_frame:
        cols.append("frame")
    if "class_name" in present:
        cols.append("class_name")
    if has_pos:
        cols.append("position")
    cols.append("document")
    for k in TABLE_KEYS:
        if k != "class_name" and k in present:
            cols.append(k)
    if has_dist:
        cols.append("distance")
    return cols


def _header(base: str, key: str, cur_sort: str, cur_dir: str) -> str:
    label = {"id": "id", "frame": "frame", "document": "document",
             "class_name": "class", "position": "position (x, y, z)",
             "confidence": "confidence", "position_conf": "pos conf",
             "sightings": "sightings", "last_seen_ts": "last seen",
             "first_seen_ts": "first seen", "embedding_model": "model",
             "distance": "distance"}.get(key, key)
    if key not in SORTABLE:
        return f"<th>{e(label)}</th>"
    nxt = "desc" if (cur_sort == key and cur_dir == "asc") else "asc"
    arrow = ""
    cls = "sortable"
    if cur_sort == key:
        arrow = " ▴" if cur_dir == "asc" else " ▾"
        cls += " on"
    url = _url(base, carry=True, sort=key, dir=nxt, page=None)
    return f"<th class='{cls}'><a href='{url}'>{e(label)}{arrow}</a></th>"


def _render_table(di: int, coll: str, rows: list[dict], base: str,
                  cur_sort: str, cur_dir: str, linked: bool = False) -> str:
    if not rows:
        return ("<div class='empty'><div class='empty-ico'>∅</div>"
                "<p>No records to show.</p>"
                "<p class='muted'>This collection is empty, or nothing matched "
                "your filter. New rows appear here as the robot writes them.</p></div>")
    cols = _columns(rows)
    tcls = " class='linked'" if linked else ""
    out = [f"<div class='card tablewrap'><table{tcls}><thead><tr>"]
    out += [_header(base, c, cur_sort, cur_dir) for c in cols]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append(f"<tr data-rid='{e(str(r['id']))}'>")
        out += [_cell(di, coll, c, r) for c in cols]
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _render_map(di: int, coll: str, rows: list[dict], base: str = "",
                colorby: str = "class", aspect: str = "fill") -> str:
    pts = []
    for r in rows:
        pos = _position(r["meta"])
        if not pos:
            continue
        m = r["meta"]
        try:
            conf = float(m.get("position_conf", m.get("confidence")))
        except (TypeError, ValueError):
            conf = None
        try:
            ts = float(m.get("last_seen_ts"))
        except (TypeError, ValueError):
            ts = None
        pts.append({"x": pos[0], "y": pos[1], "cls": str(m.get("class_name", "?")),
                    "rid": str(r["id"]), "conf": conf, "ts": ts})
    if not pts:
        return ""
    xs = [p["x"] for p in pts] + [0.0]
    ys = [p["y"] for p in pts] + [0.0]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    if maxx - minx < 1e-6:
        minx, maxx = minx - 1, maxx + 1
    if maxy - miny < 1e-6:
        miny, maxy = miny - 1, maxy + 1
    W, H, pad = 1040.0, 460.0, 48.0  # wide landscape rectangle
    if aspect == "equal":  # true-to-scale: 1 m is the same in x and y (square cells)
        scale = min((W - 2 * pad) / (maxx - minx), (H - 2 * pad) / (maxy - miny))
        uw, uh = (maxx - minx) * scale, (maxy - miny) * scale
        plot_l, plot_t = (W - uw) / 2, (H - uh) / 2
        plot_r, plot_b = plot_l + uw, plot_t + uh
        px = lambda x: plot_l + (x - minx) * scale
        py = lambda y: plot_b - (y - miny) * scale
    else:  # fill: x and y scaled independently to fill the rectangle
        plot_l, plot_r, plot_t, plot_b = pad, W - pad, pad, H - pad
        px = lambda x: plot_l + (x - minx) / (maxx - minx) * (plot_r - plot_l)
        py = lambda y: plot_b - (y - miny) / (maxy - miny) * (plot_b - plot_t)

    svg = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' class='map' "
           f"preserveAspectRatio='xMidYMid meet'>"]
    svg.append(f"<rect x='0' y='0' width='{W:.0f}' height='{H:.0f}' class='map-bg'/>")

    # 0.5 m grid (heavier line + tick label at each whole metre). Skipped if the
    # span is so large the lines would just be noise.
    step = 0.5
    if (maxx - minx) <= 40 and (maxy - miny) <= 40:
        for i in range(math.ceil(minx / step - 1e-9), math.floor(maxx / step + 1e-9) + 1):
            gx = i * step
            if abs(gx) < 1e-9:
                continue  # origin axis drawn below
            X = px(gx)
            major = i % 2 == 0
            svg.append(f"<line x1='{X:.1f}' y1='{plot_t:.1f}' x2='{X:.1f}' "
                       f"y2='{plot_b:.1f}' class='map-grid{' major' if major else ''}'/>")
            if major:
                svg.append(f"<text x='{X:.1f}' y='{plot_b + 14:.1f}' text-anchor='middle' "
                           f"class='map-tick'>{gx:.0f}</text>")
        for i in range(math.ceil(miny / step - 1e-9), math.floor(maxy / step + 1e-9) + 1):
            gy = i * step
            if abs(gy) < 1e-9:
                continue
            Y = py(gy)
            major = i % 2 == 0
            svg.append(f"<line x1='{plot_l:.1f}' y1='{Y:.1f}' x2='{plot_r:.1f}' "
                       f"y2='{Y:.1f}' class='map-grid{' major' if major else ''}'/>")
            if major:
                svg.append(f"<text x='{plot_l - 7:.1f}' y='{Y + 3:.1f}' text-anchor='end' "
                           f"class='map-tick'>{gy:.0f}</text>")

    # axes through origin
    if minx <= 0 <= maxx:
        svg.append(f"<line x1='{px(0):.1f}' y1='{plot_t:.1f}' x2='{px(0):.1f}' "
                   f"y2='{plot_b:.1f}' class='map-axis'/>")
    if miny <= 0 <= maxy:
        svg.append(f"<line x1='{plot_l:.1f}' y1='{py(0):.1f}' x2='{plot_r:.1f}' "
                   f"y2='{py(0):.1f}' class='map-axis'/>")
    if minx <= 0 <= maxx and miny <= 0 <= maxy:
        svg.append(f"<circle cx='{px(0):.1f}' cy='{py(0):.1f}' r='5' class='map-origin'/>"
                   f"<text x='{px(0) + 8:.1f}' y='{py(0) - 8:.1f}' class='map-olabel'>origin</text>")

    tss = [p["ts"] for p in pts if p["ts"] is not None]
    mints, maxts = (min(tss), max(tss)) if tss else (None, None)

    def fill(p: dict) -> str:
        if colorby == "confidence" and p["conf"] is not None:
            return f"hsl({max(0.0, min(1.0, p['conf'])) * 120:.0f} 70% 50%)"
        if colorby == "recency" and p["ts"] is not None and maxts and maxts > mints:
            frac = (p["ts"] - mints) / (maxts - mints)
            return f"hsl({frac * 120:.0f} 70% 50%)"
        return f"hsl({_hue(p['cls'])} 70% 55%)"

    for p in pts[:1000]:
        x, y, cls, rid = p["x"], p["y"], p["cls"], p["rid"]
        extra = ""
        if p["conf"] is not None:
            extra += f" conf={p['conf']:.2f}"
        svg.append(
            f"<a href='{_record_url(di, coll, rid)}'>"
            f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' r='6' data-pt='{e(rid)}' "
            f"data-x='{x:.3f}' data-y='{y:.3f}' "
            f"style='fill:{fill(p)}' class='map-pt'>"
            f"<title>{e(cls)} — ({x:.2f}, {y:.2f}){e(extra)}\n{e(rid)}</title></circle></a>"
        )
    svg.append("</svg>")

    # legend adapts to the colour mode
    if colorby == "class":
        legend = "".join(
            f"<span class='leg'><i style='background:hsl({_hue(c)} 70% 55%)'></i>{e(c)}</span>"
            for c in sorted({p["cls"] for p in pts})
        )
    elif colorby == "confidence":
        legend = ("<span class='leg'>low 0.0</span><span class='gradient'></span>"
                  "<span class='leg'>1.0 high</span>")
    else:  # recency
        lo = _rel_time(mints)[0] if mints else "older"
        hi = _rel_time(maxts)[0] if maxts else "newer"
        legend = (f"<span class='leg'>{e(lo)}</span><span class='gradient'></span>"
                  f"<span class='leg'>{e(hi)}</span>")

    controls = ""
    if base:
        def ctl(param: str, val: str, label: str, cur: str) -> str:
            on = " on" if cur == val else ""
            return f"<a class='mctl{on}' href='{_url(base, carry=True, **{param: val})}'>{label}</a>"
        controls = (
            "<div class='map-ctl'><span class='muted'>colour:</span>"
            + ctl("colorby", "class", "class", colorby)
            + ctl("colorby", "confidence", "conf", colorby)
            + ctl("colorby", "recency", "recency", colorby)
            + "<span class='muted'>· aspect:</span>"
            + ctl("aspect", "fill", "fill", aspect)
            + ctl("aspect", "equal", "equal", aspect)
            + "<button type='button' class='mctl' id='measure-btn'>📏 measure</button>"
            + "<span id='measure-out' class='muted'></span></div>"
        )

    cells = "square (true scale)" if aspect == "equal" else "rectangular (filled)"
    note = (f"grid 0.5 m, cells {cells} · x: {minx:.2f}…{maxx:.2f} · "
            f"y: {miny:.2f}…{maxy:.2f} (map frame) · click a row to highlight")
    return (
        "<details class='card map-card' open><summary>🗺 Position map "
        f"<span class='muted'>· {len(pts)} located</span></summary>"
        f"<div class='map-body'>{controls}{''.join(svg)}"
        f"<div class='legend'>{legend}</div>"
        f"<p class='muted small'>{e(note)}</p></div></details>"
    )


def _class_chips(base: str, classes: list[tuple[str, int]], active: Optional[str]) -> str:
    if not classes:
        return ""
    chips = [
        f"<a class='chip{' on' if not active else ''}' "
        f"href='{_url(base, carry=True, page=None, **{'class': None})}'>all</a>"
    ]
    for name, cnt in classes:
        on = " on" if active == name else ""
        url = _url(base, carry=True, page=None, **{"class": name})
        chips.append(
            f"<a class='chip{on}' href='{url}' style='--h:{_hue(name)}'>"
            f"<i class='dot'></i>{e(name)} <b>{cnt}</b></a>"
        )
    return f"<div class='chips'>{''.join(chips)}</div>"


def _toolbar(di: int, coll: str, *, q: str, mode: str) -> str:
    sub = "selected" if mode != "semantic" else ""
    sem = "selected" if mode == "semantic" else ""
    clear = (f"<a class='btn ghost' href='{_browse_path(di, coll)}'>clear</a>"
             if q else "")
    return (
        f"<form class='toolbar' action='{_search_path(di, coll)}' method='get'>"
        f"<input id='searchq' type='text' name='q' value='{e(q)}' "
        f"placeholder='Search this collection…  (press / to focus)' autocomplete='off'>"
        f"<select name='mode'><option value='substring' {sub}>substring</option>"
        f"<option value='semantic' {sem}>semantic</option></select>"
        f"<button class='btn' type='submit'>Search</button>{clear}</form>"
    )


def _pager(di: int, coll: str, page_num: int, pages: int) -> str:
    if pages <= 1:
        return ""
    base = _browse_path(di, coll)
    prev = (f"<a class='btn ghost' href='{_url(base, carry=True, page=page_num - 1)}'>← prev</a>"
            if page_num > 1 else "<span class='btn ghost disabled'>← prev</span>")
    nxt = (f"<a class='btn ghost' href='{_url(base, carry=True, page=page_num + 1)}'>next →</a>"
           if page_num < pages else "<span class='btn ghost disabled'>next →</span>")
    return (f"<div class='pager'>{prev}"
            f"<span class='muted'>page {page_num} / {pages}</span>{nxt}</div>")


# --------------------------------------------------------------------------- #
# Shared view helpers (nav, fetch, panels, charts)
# --------------------------------------------------------------------------- #


def _coll_nav(di: int, coll: str, active: str) -> str:
    cq = quote(coll)
    tabs = [
        ("browse", f"/c/{di}/{cq}", "Browse"),
        ("gallery", f"/c/{di}/{cq}/gallery", "Gallery"),
        ("stats", f"/c/{di}/{cq}/stats", "Stats"),
        ("changes", f"/c/{di}/{cq}/changes", "Changes"),
    ]
    out = ["<div class='tabs'>"]
    for key, url, label in tabs:
        out.append(f"<a class='tab{' on' if active == key else ''}' href='{url}'>{label}</a>")
    out.append("</div>")
    return "".join(out)


def _load_rows(collection, class_filter: Optional[str], *, with_emb: bool = False):
    where = {"class_name": class_filter} if class_filter else None
    inc = ["metadatas", "documents"] + (["embeddings"] if with_emb else [])
    return _rows(collection.get(where=where, include=inc, limit=BROWSE_FETCH_CAP),
                 with_emb=with_emb)


def _class_counts(collection) -> list[tuple[str, int]]:
    res = collection.get(include=["metadatas"], limit=BROWSE_FETCH_CAP)
    return Counter(
        str(m.get("class_name")) for m in (res.get("metadatas") or [])
        if m.get("class_name")
    ).most_common()


def _crumb(store: Store, di: int, coll: str, leaf: str) -> str:
    return (f"<div class='crumb'><a href='/'>overview</a> / "
            f"<code>{e(store.directory)}</code> / "
            f"<a href='{_browse_path(di, coll)}'>{e(coll)}</a> / <b>{e(leaf)}</b></div>")


def _agent_panel(q: str, rows: list[dict], n: int = 5) -> str:
    """Mirror what the agent's ``find_object_from_memory`` tool would return."""
    if not q:
        return ""
    lines = [f"find_object_from_memory({q!r}) →"]
    if not rows:
        lines.append(f"No record of '{q}' in memory.")
    else:
        lines.append(f"Top matches for '{q}':")
        for r in rows[:n]:
            m = r["meta"]
            pos = _position(m) or (0.0, 0.0, 0.0)
            try:
                conf = float(m.get("position_conf", m.get("confidence", 0.0)))
            except (TypeError, ValueError):
                conf = 0.0
            lines.append(
                f"- {m.get('class_name', '?')} @ "
                f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) "
                f"conf={conf:.2f} sightings={m.get('sightings', 1)} "
                f"caption={str(m.get('caption', ''))!r}"
            )
    body = e("\n".join(lines))
    return ("<div class='section-h'>🤖 agent's-eye view</div>"
            "<div class='card agentp'><div class='muted small' style='margin-bottom:6px'>"
            "What the Walkie agent receives from this lookup:</div>"
            f"<pre>{body}</pre></div>")


def _similar(collection, rid: str, emb, n: int = 6) -> list[dict]:
    """Nearest neighbours of ``emb`` in the same collection, excluding self."""
    try:
        vec = [float(v) for v in emb]
    except (TypeError, ValueError):
        return []
    res = collection.query(
        query_embeddings=[vec], n_results=min(n + 1, max(1, collection.count())),
        include=["metadatas", "documents", "distances"],
    )
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    docs = res["documents"][0]
    dists = (res.get("distances") or [[None] * len(ids)])[0]
    out = []
    for cid, meta, doc, dist in zip(ids, metas, docs, dists):
        if str(cid) == str(rid):
            continue
        out.append({"id": cid, "meta": dict(meta or {}), "doc": doc or "",
                    "emb": None, "distance": dist})
    return out[:n]


def _render_similar(di: int, coll: str, neighbors: list[dict]) -> str:
    if not neighbors:
        return ""
    rows = []
    for r in neighbors:
        m = r["meta"]
        furl = _frame_url(m)
        thumb = (f"<img class='thumb' data-zoom src='{furl}'>" if furl
                 else "<span class='nothumb'></span>")
        badge = _class_badge(m["class_name"]) if m.get("class_name") else ""
        pos = _position(m)
        posh = (f"<code class='muted'>{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}</code>"
                if pos else "")
        d = r.get("distance")
        dist = _bar(1 - min(float(d), 2.0) / 2.0, f"{float(d):.4f}", hue=152) if d is not None else ""
        rows.append(
            f"<a class='simrow' href='{_record_url(di, coll, r['id'])}'>{thumb}"
            f"<span class='simmeta'>{badge} {posh}</span>{dist}</a>"
        )
    return ("<div class='section-h'>similar (by embedding)</div>"
            f"<div class='card simwrap'>{''.join(rows)}</div>")


def _bars(items: list[tuple[str, int]], *, hue_by_label: bool = False,
          maxn: int = 12) -> str:
    if not items:
        return "<p class='muted'>no data</p>"
    top = items[:maxn]
    mx = max(c for _, c in top) or 1
    out = ["<div class='bars'>"]
    for label, cnt in top:
        w = cnt / mx * 100
        h = _hue(label) if hue_by_label else 211
        out.append(
            f"<div class='barrow'><span class='barlabel' title='{e(label)}'>{e(label)}</span>"
            f"<span class='bartrack'><span class='barv' style='width:{w:.1f}%;--h:{h}'></span></span>"
            f"<span class='barn'>{cnt:,}</span></div>"
        )
    if len(items) > maxn:
        out.append(f"<div class='muted small'>+{len(items) - maxn} more…</div>")
    out.append("</div>")
    return "".join(out)


def _histogram(values: list[float], *, buckets: int = 10,
               lo: Optional[float] = None, hi: Optional[float] = None) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "<p class='muted'>no data</p>"
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    if hi - lo < 1e-9:
        hi = lo + 1
    counts = [0] * buckets
    for v in vals:
        b = int((v - lo) / (hi - lo) * buckets)
        b = max(0, min(buckets - 1, b))
        counts[b] += 1
    mx = max(counts) or 1
    out = ["<div class='hist'>"]
    for i, c in enumerate(counts):
        edge = lo + (i + 0.5) * (hi - lo) / buckets
        out.append(
            f"<span class='hbar' title='~{edge:.2f}: {c}' "
            f"style='height:{c / mx * 100:.0f}%'></span>"
        )
    out.append("</div>")
    out.append(f"<div class='histx'><span>{lo:.2f}</span><span>{hi:.2f}</span></div>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Page shell
# --------------------------------------------------------------------------- #

BASE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} · Chroma Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>
(function(){var t=localStorage.getItem('chromaViewerTheme')||'auto';
var d=window.matchMedia&&window.matchMedia('(prefers-color-scheme:dark)').matches;
document.documentElement.setAttribute('data-theme',t==='auto'?(d?'dark':'light'):t);})();
</script>
<style>
  :root{
    --bg:#f6f7f9; --fg:#1a1d23; --muted:#6b7280; --card:#ffffff; --border:#e4e7ec;
    --topbar:#111827; --topbar-fg:#e5e7eb; --side:#fbfbfc; --accent:#2563eb;
    --hover:#f1f3f7; --chip:#eef1f5; --chip-on:#dbe4ff; --shadow:0 1px 2px #0000000d;
    color-scheme:light;
  }
  [data-theme=dark]{
    --bg:#0e1116; --fg:#e6e9ee; --muted:#8b94a3; --card:#161b22; --border:#272d36;
    --topbar:#0a0d11; --topbar-fg:#e6e9ee; --side:#11151b; --accent:#60a5fa;
    --hover:#1b212b; --chip:#1b212b; --chip-on:#1e2a44; --shadow:none;
    color-scheme:dark;
  }
  *{box-sizing:border-box}
  body{font:13.5px/1.5 'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;
       background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
  a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
  code{font-family:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}
  .muted{color:var(--muted)} .small{font-size:12px}

  .topbar{display:flex;align-items:center;gap:14px;background:var(--topbar);
          color:var(--topbar-fg);padding:0 16px;height:50px;flex:0 0 auto}
  .topbar .brand{font-weight:700;color:#fff;display:flex;align-items:center;gap:8px}
  .topbar .brand .logo{width:18px;height:18px;border-radius:5px;
       background:linear-gradient(135deg,#60a5fa,#a78bfa)}
  .topbar a{color:#cbd5e1} .spacer{margin-left:auto}
  .icon-btn{background:#ffffff1a;color:var(--topbar-fg);border:1px solid #ffffff22;
       border-radius:8px;height:30px;min-width:32px;cursor:pointer;font-size:14px}
  .icon-btn:hover{background:#ffffff2e}
  .rt{display:flex;align-items:center;gap:7px;font-size:12.5px;color:#cbd5e1}
  .rt #rt-dot{width:8px;height:8px;border-radius:50%;background:#475569;transition:.2s}
  .rt.live #rt-dot{background:#22c55e;box-shadow:0 0 7px #22c55e}
  .rt.flash #rt-dot{background:#fbbf24;box-shadow:0 0 8px #fbbf24}
  .rt select{background:#ffffff14;color:var(--topbar-fg);border:1px solid #ffffff22;
       border-radius:7px;padding:3px 6px}
  #rt-count{min-width:54px;text-align:right;font-variant-numeric:tabular-nums}

  .layout{display:flex;flex:1;min-height:0}
  .sidebar{width:248px;flex:0 0 auto;background:var(--side);border-right:1px solid var(--border);
           overflow:auto;padding:12px}
  .sidebar .store{margin-bottom:16px}
  .sidebar .store-name{font-size:11px;letter-spacing:.04em;text-transform:uppercase;
       color:var(--muted);padding:4px 8px;font-weight:600}
  .sidebar .coll{display:flex;justify-content:space-between;align-items:center;gap:8px;
       padding:6px 9px;border-radius:8px;color:var(--fg)}
  .sidebar .coll:hover{background:var(--hover);text-decoration:none}
  .sidebar .coll.active{background:var(--chip-on);color:var(--fg);font-weight:600}
  .sidebar .cbadge{background:var(--chip);border-radius:20px;padding:0 8px;font-size:11.5px;
       color:var(--muted);font-variant-numeric:tabular-nums}
  .sidebar .side-err{color:#ef4444;font-size:12px;padding:2px 8px}

  main{flex:1;overflow:auto;padding:20px 24px}
  .crumb{font-size:13px;color:var(--muted);margin-bottom:12px}
  .crumb a{color:var(--muted)} .crumb b{color:var(--fg)}
  h1{font-size:20px;margin:0 0 2px} h1 .count{font-size:13px;color:var(--muted);font-weight:400}
  .page-head{margin-bottom:14px}

  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;
        box-shadow:var(--shadow);margin-bottom:16px}
  .tablewrap{overflow:auto}
  table{border-collapse:collapse;width:100%}
  thead th{position:sticky;top:0;background:var(--card);text-align:left;font-size:12px;
     color:var(--muted);font-weight:600;padding:9px 12px;border-bottom:1px solid var(--border);
     white-space:nowrap;z-index:1}
  th.sortable a{color:var(--muted)} th.sortable.on a{color:var(--accent)}
  tbody td{padding:8px 12px;border-bottom:1px solid var(--border);vertical-align:middle;
     font-size:13px}
  tbody tr:last-child td{border-bottom:none}
  tbody tr:hover{background:var(--hover)}
  .c-id a{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:12px}
  .c-doc{max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .c-pos code{color:var(--muted)}
  .copy{margin-left:6px;border:none;background:transparent;color:var(--muted);cursor:pointer;
     opacity:0;font-size:12px} tr:hover .copy{opacity:1} .copy:hover{color:var(--accent)}

  .badge{display:inline-block;padding:1px 9px;border-radius:20px;font-size:12px;font-weight:600;
     background:hsl(var(--h) 70% 55% / .16);color:hsl(var(--h) 65% 38%);
     border:1px solid hsl(var(--h) 70% 55% / .32)}
  [data-theme=dark] .badge{color:hsl(var(--h) 80% 72%)}
  .bar{position:relative;display:inline-block;width:96px;height:18px;border-radius:5px;
     background:var(--chip);overflow:hidden;vertical-align:middle}
  .bar-fill{position:absolute;inset:0 auto 0 0;background:hsl(var(--h) 75% 55% / .85)}
  .bar-txt{position:relative;display:block;text-align:center;font-size:11px;line-height:18px;
     font-variant-numeric:tabular-nums;color:var(--fg);mix-blend-mode:normal}

  .toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
  .toolbar input[type=text]{flex:1;min-width:240px;padding:8px 12px;border-radius:9px;
     border:1px solid var(--border);background:var(--card);color:var(--fg)}
  .toolbar select{padding:8px 10px;border-radius:9px;border:1px solid var(--border);
     background:var(--card);color:var(--fg)}
  .btn{padding:8px 14px;border-radius:9px;border:1px solid var(--accent);background:var(--accent);
     color:#fff;cursor:pointer;font-size:13px}
  .btn:hover{filter:brightness(1.06);text-decoration:none}
  .btn.ghost{background:transparent;color:var(--fg);border-color:var(--border)}
  .btn.ghost:hover{background:var(--hover)} .btn.disabled{opacity:.4;pointer-events:none}

  .chips{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:14px}
  .chip{display:inline-flex;align-items:center;gap:6px;padding:3px 11px;border-radius:20px;
     background:var(--chip);color:var(--fg);font-size:12.5px;border:1px solid transparent}
  .chip:hover{text-decoration:none;border-color:var(--border)}
  .chip.on{background:var(--chip-on);font-weight:600}
  .chip .dot{width:8px;height:8px;border-radius:50%;background:hsl(var(--h) 70% 55%)}
  .chip b{color:var(--muted);font-weight:600}

  details.map-card{padding:0} details.map-card summary{cursor:pointer;padding:11px 14px;
     font-weight:600;user-select:none} .map-body{padding:0 14px 14px;text-align:center}
  svg.map{width:100%;max-width:1040px;height:auto;border-radius:9px;border:1px solid var(--border);
     display:block;margin:0 auto}
  .map-bg{fill:var(--bg)}
  .map-grid{stroke:var(--border);stroke-width:1;opacity:.4}
  .map-grid.major{opacity:.8}
  .map-tick{fill:var(--muted);font-size:9px;opacity:.85}
  .map-axis{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:5 4;opacity:.7}
  .map-pt{cursor:pointer;stroke:var(--card);stroke-width:1.5;transition:r .1s}
  .map-pt:hover{r:9} .map-origin{fill:none;stroke:var(--accent);stroke-width:2}
  .map-olabel{fill:var(--muted);font-size:11px}
  @keyframes ptpulse{0%,100%{stroke-width:1.5}50%{stroke-width:5}}
  .map-pt.hl{stroke:#fff;filter:drop-shadow(0 0 5px #fff);animation:ptpulse 1.1s ease-in-out infinite}
  .map.map-dim .map-pt:not(.hl){opacity:.28}
  .map.map-dim .map-grid,.map.map-dim .map-axis,.map.map-dim .map-tick{opacity:.2}
  tr.row-hl{background:var(--chip-on)!important}
  table.linked tbody tr{cursor:pointer}
  .legend{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 2px;justify-content:center}
  .leg{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
  .leg i{width:10px;height:10px;border-radius:50%}

  dl.meta{display:grid;grid-template-columns:max-content 1fr;gap:7px 20px;margin:0;padding:14px 16px}
  dl.meta dt{color:var(--muted);font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:12.5px}
  dl.meta dd{margin:0;word-break:break-word}
  .detail-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px}
  .detail-head .idtxt{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:13px;
     background:var(--chip);padding:3px 9px;border-radius:7px}
  img.full{max-width:520px;width:100%;border-radius:10px;cursor:zoom-in;display:block}
  .spark{width:100%;height:60px;display:block}
  .spark path{fill:none;stroke:var(--accent);stroke-width:1.5}
  .spark .mid{stroke:var(--border);stroke-width:1}
  .section-h{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
     margin:18px 0 8px;font-weight:600}

  .empty{text-align:center;padding:54px 20px;color:var(--muted)}
  .empty-ico{font-size:40px;opacity:.5;margin-bottom:8px}
  .pager{display:flex;align-items:center;gap:12px;margin-top:6px}
  .warn{background:#f59e0b1f;border:1px solid #f59e0b66;color:var(--fg);padding:9px 13px;
     border-radius:9px;margin-bottom:12px;font-size:13px}
  .ov-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
  .ov-card{padding:16px} .ov-card h3{margin:0 0 4px;font-size:15px}
  .ov-card .big{font-size:28px;font-weight:700;font-variant-numeric:tabular-nums}
  .ov-card .topc{margin-top:10px;display:flex;gap:6px;flex-wrap:wrap}

  /* collection tab bar */
  .tabs{display:flex;gap:4px;margin-bottom:14px;border-bottom:1px solid var(--border)}
  .tab{padding:7px 14px;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-1px}
  .tab:hover{text-decoration:none;color:var(--fg)}
  .tab.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}

  /* map controls + gradient legend + measure overlay */
  .map-ctl{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:10px;font-size:12.5px}
  .mctl{padding:3px 9px;border-radius:7px;background:var(--chip);color:var(--fg);
        border:1px solid transparent;cursor:pointer;font-size:12.5px}
  .mctl:hover{text-decoration:none;border-color:var(--border)} .mctl.on{background:var(--chip-on);font-weight:600}
  .gradient{display:inline-block;width:120px;height:12px;border-radius:6px;
        background:linear-gradient(90deg,hsl(0 70% 50%),hsl(60 70% 50%),hsl(120 70% 50%))}
  .measure-line{stroke:var(--accent);stroke-width:2;stroke-dasharray:5 4}
  .measure-pt{fill:none;stroke:var(--accent);stroke-width:2}
  .measure-label{fill:var(--fg);font-size:12px;font-weight:600;paint-order:stroke;
        stroke:var(--card);stroke-width:3px}

  /* similar (record page) */
  .simwrap{padding:8px;display:flex;flex-direction:column;gap:2px}
  .simrow{display:flex;align-items:center;gap:12px;padding:6px 8px;border-radius:8px;color:var(--fg)}
  .simrow:hover{background:var(--hover);text-decoration:none}
  .simrow .thumb{height:40px} .simmeta{flex:1;display:flex;align-items:center;gap:10px}
  .nothumb{width:40px;height:40px;border-radius:4px;background:var(--chip);display:inline-block}

  /* agent panel */
  .agentp{padding:14px} .agentp pre{margin:0;white-space:pre-wrap;font-size:12.5px;
        background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px}

  /* stats */
  .statgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
  .statcard{padding:16px} .stat-h{margin:0 0 12px;font-size:14px}
  .bars{display:flex;flex-direction:column;gap:7px}
  .barrow{display:flex;align-items:center;gap:10px;font-size:12.5px}
  .barlabel{width:84px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}
  .bartrack{flex:1;height:14px;background:var(--chip);border-radius:5px;overflow:hidden}
  .barv{display:block;height:100%;background:hsl(var(--h) 70% 55% / .85)}
  .barn{width:48px;text-align:right;font-variant-numeric:tabular-nums}
  .hist{display:flex;align-items:flex-end;gap:3px;height:120px;padding-top:6px}
  .hist .hbar{flex:1;background:var(--accent);border-radius:3px 3px 0 0;min-height:2px;opacity:.8}
  .histx{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:4px}
  ul.checks{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
  ul.checks li{font-size:12.5px} ul.checks li.ok{color:#16a34a} ul.checks li.bad{color:#d97706}

  /* gallery */
  .gal-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
  .gcard{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden}
  .gimg{width:100%;height:120px;object-fit:cover;display:block;cursor:zoom-in;background:var(--chip)}
  .gimg.noimg{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;cursor:default}
  .gmeta{display:flex;align-items:center;justify-content:space-between;gap:6px;padding:7px 9px 2px}
  .gid{display:block;padding:0 9px 9px;font-size:11px;font-family:'IBM Plex Mono',monospace;
        overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

  /* changes feed */
  .changes-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
  .feedrow{display:flex;align-items:center;gap:9px;padding:6px 8px;border-radius:8px;color:var(--fg)}
  .feedrow:hover{background:var(--hover);text-decoration:none} .feedrow .thumb{height:34px}

  .lightbox{position:fixed;inset:0;background:#000c;display:flex;align-items:center;
     justify-content:center;z-index:50;cursor:zoom-out;padding:24px}
  .lightbox img{max-width:96vw;max-height:92vh;border-radius:10px;box-shadow:0 10px 40px #000a}
  [hidden]{display:none!important}
</style>
</head>
<body>
<header class="topbar">
  <a class="brand" href="/"><span class="logo"></span>Chroma Viewer</a>
  <span class="spacer"></span>
  <span class="rt" id="rt"><span id="rt-dot"></span>
    <select id="rt-sel" title="Auto-refresh interval">
      <option value="0">off</option><option value="2">2s</option>
      <option value="5">5s</option><option value="10">10s</option>
      <option value="30">30s</option>
    </select>
    <span id="rt-count"></span>
  </span>
  <button id="theme-btn" class="icon-btn" title="Theme">◐</button>
</header>
<div class="layout">
  <aside class="sidebar">{{ sidebar|safe }}</aside>
  <main>{{ body|safe }}</main>
</div>
<div id="lightbox" class="lightbox" hidden><img id="lightbox-img" alt=""></div>
<script>
(function(){
  // ---- theme ----------------------------------------------------------------
  var TKEY='chromaViewerTheme';
  function applyTheme(){
    var t=localStorage.getItem(TKEY)||'auto';
    var d=window.matchMedia&&window.matchMedia('(prefers-color-scheme:dark)').matches;
    document.documentElement.setAttribute('data-theme',t==='auto'?(d?'dark':'light'):t);
    var b=document.getElementById('theme-btn');
    b.textContent=t==='auto'?'◐':(t==='dark'?'☾':'☀');
    b.title='Theme: '+t+' (click to change)';
  }
  document.getElementById('theme-btn').addEventListener('click',function(){
    var t=localStorage.getItem(TKEY)||'auto';
    t=t==='auto'?'light':(t==='light'?'dark':'auto');
    localStorage.setItem(TKEY,t);applyTheme();
  });
  applyTheme();

  // ---- delegated UI (survives partial refresh) ------------------------------
  var lb=document.getElementById('lightbox'),lbi=document.getElementById('lightbox-img');

  // Row <-> map highlighting: click a table row to light up its point.
  function cssEsc(s){return (window.CSS&&CSS.escape)?CSS.escape(s)
    :String(s).replace(/["\\]/g,'\\$&');}
  var curRid=null,curPt=null;
  function clearMapHl(){
    if(curPt){curPt.classList.remove('hl');curPt.setAttribute('r','6');curPt=null;}
    document.querySelectorAll('tr.row-hl').forEach(function(r){r.classList.remove('row-hl');});
    var m=document.querySelector('svg.map'); if(m)m.classList.remove('map-dim');
    curRid=null;
  }
  function highlightOnMap(rid,row){
    if(curRid===rid){clearMapHl();return;}   // click again to clear
    clearMapHl();
    var map=document.querySelector('svg.map'); if(!map)return;
    var pt=map.querySelector('[data-pt="'+cssEsc(rid)+'"]'); if(!pt)return;
    pt.classList.add('hl');pt.setAttribute('r','11');curPt=pt;
    map.classList.add('map-dim');
    if(row)row.classList.add('row-hl');
    var a=pt.closest('a')||pt; a.parentNode.appendChild(a);  // raise above others
    map.scrollIntoView({behavior:'smooth',block:'center'});
    curRid=rid;
  }

  // ---- distance-measure tool on the map -------------------------------------
  var SVGNS='http://www.w3.org/2000/svg',measureMode=false,measurePts=[];
  function clearMeasure(){var l=document.getElementById('measure-layer');if(l)l.remove();measurePts=[];}
  function drawMeasure(){
    var svg=document.querySelector('svg.map');if(!svg)return;
    var l=document.getElementById('measure-layer');
    if(!l){l=document.createElementNS(SVGNS,'g');l.id='measure-layer';svg.appendChild(l);}
    l.innerHTML='';
    measurePts.forEach(function(p){
      var c=document.createElementNS(SVGNS,'circle');
      c.setAttribute('cx',p.cx);c.setAttribute('cy',p.cy);c.setAttribute('r',7);
      c.setAttribute('class','measure-pt');l.appendChild(c);});
    if(measurePts.length===2){
      var a=measurePts[0],b=measurePts[1];
      var ln=document.createElementNS(SVGNS,'line');
      ln.setAttribute('x1',a.cx);ln.setAttribute('y1',a.cy);
      ln.setAttribute('x2',b.cx);ln.setAttribute('y2',b.cy);
      ln.setAttribute('class','measure-line');l.appendChild(ln);
      var d=Math.hypot(b.x-a.x,b.y-a.y);
      var t=document.createElementNS(SVGNS,'text');
      t.setAttribute('x',(a.cx+b.cx)/2);t.setAttribute('y',(a.cy+b.cy)/2-6);
      t.setAttribute('text-anchor','middle');t.setAttribute('class','measure-label');
      t.textContent=d.toFixed(2)+' m';l.appendChild(t);
      var o=document.getElementById('measure-out');if(o)o.textContent='= '+d.toFixed(2)+' m';}
  }
  function addMeasurePoint(c){
    if(measurePts.length>=2)clearMeasure();
    measurePts.push({x:parseFloat(c.getAttribute('data-x')),y:parseFloat(c.getAttribute('data-y')),
      cx:parseFloat(c.getAttribute('cx')),cy:parseFloat(c.getAttribute('cy'))});
    drawMeasure();
  }
  function setMeasure(on){
    measureMode=on;var b=document.getElementById('measure-btn');
    if(b){b.classList.toggle('on',on);b.textContent=on?'📏 click two points':'📏 measure';}
    if(!on){clearMeasure();var o=document.getElementById('measure-out');if(o)o.textContent='';}
  }

  document.addEventListener('click',function(ev){
    var mb=ev.target.closest&&ev.target.closest('#measure-btn');
    if(mb){ev.preventDefault();setMeasure(!measureMode);return;}
    if(measureMode){var mc=ev.target.closest&&ev.target.closest('.map-pt');
      if(mc){ev.preventDefault();addMeasurePoint(mc);return;}}
    var z=ev.target.closest&&ev.target.closest('[data-zoom]');
    if(z){ev.preventDefault();lbi.src=z.getAttribute('src');lb.hidden=false;return;}
    var c=ev.target.closest&&ev.target.closest('[data-copy]');
    if(c){navigator.clipboard&&navigator.clipboard.writeText(c.getAttribute('data-copy'));
      c.textContent='✓';setTimeout(function(){c.textContent='⧉';},900);return;}
    if(ev.target.closest&&ev.target.closest('.map-bg')){clearMapHl();return;}
    var row=ev.target.closest&&ev.target.closest('tr[data-rid]');
    if(row){
      if(ev.target.closest('a'))return;   // let the id link open the record
      highlightOnMap(row.getAttribute('data-rid'),row);
    }
  });
  lb.addEventListener('click',function(){lb.hidden=true;lbi.src='';});
  document.addEventListener('keydown',function(ev){
    if(ev.key==='Escape'){lb.hidden=true;lbi.src='';}
    if(ev.key==='/'){var a=document.activeElement;
      if(a&&(a.tagName==='INPUT'||a.tagName==='TEXTAREA'))return;
      var s=document.getElementById('searchq');if(s){ev.preventDefault();s.focus();}}
  });

  // ---- auto-refresh: swap <main> only, keep scroll/focus --------------------
  var RKEY='chromaViewerRefresh';
  var sel=document.getElementById('rt-sel'),box=document.getElementById('rt'),
      out=document.getElementById('rt-count');
  var cur=localStorage.getItem(RKEY); if(cur===null) cur='{{ default_refresh }}';
  sel.value=cur;
  var interval=parseInt(cur||'0',10),remaining=interval,busy=false;
  function typing(){var a=document.activeElement;
    return a&&(a.tagName==='INPUT'||a.tagName==='TEXTAREA');}
  function partialUrl(){var u=new URL(window.location.href);
    u.searchParams.set('_partial','1');return u.toString();}
  async function refreshNow(){
    if(busy)return; busy=true;
    var m=document.querySelector('main'),top=m?m.scrollTop:0;
    box.classList.add('flash');
    try{var r=await fetch(partialUrl(),{cache:'no-store'});
      if(r.ok){var t=await r.text();if(m){m.innerHTML=t;m.scrollTop=top;
        measureMode=false;measurePts=[];   // measure layer is gone with the swap
        if(curRid){var s=curRid;curRid=null;curPt=null;   // re-apply across refresh
          highlightOnMap(s,document.querySelector('tr[data-rid="'+cssEsc(s)+'"]'));}}}}
    catch(e){}
    finally{busy=false;setTimeout(function(){box.classList.remove('flash');},250);}
  }
  function render(){
    box.classList.toggle('live',interval>0);
    out.textContent=interval<=0?'':(typing()?'paused':'⟳ '+remaining+'s');
  }
  setInterval(function(){
    if(interval<=0){render();return;}
    if(typing()){render();return;}
    remaining--; if(remaining<=0){remaining=interval;render();refreshNow();} else render();
  },1000);
  sel.addEventListener('change',function(){
    localStorage.setItem(RKEY,sel.value);
    interval=parseInt(sel.value||'0',10);remaining=interval;render();
  });
  render();
})();
</script>
</body></html>
"""


def _sidebar(active: Optional[tuple[int, str]]) -> str:
    out = []
    for di, store in enumerate(STORES):
        out.append("<div class='store'>")
        out.append(f"<div class='store-name' title='{e(store.path)}'>{e(store.directory)}</div>")
        if store.error:
            out.append(f"<div class='side-err'>{e(store.error)}</div>")
        else:
            colls = store.collections()
            if not colls:
                out.append("<div class='side-err muted'>no collections</div>")
            for name, cnt in colls:
                on = " active" if active == (di, name) else ""
                badge = "?" if cnt < 0 else f"{cnt:,}"
                out.append(
                    f"<a class='coll{on}' href='{_browse_path(di, name)}'>"
                    f"<span>{e(name)}</span><span class='cbadge'>{badge}</span></a>"
                )
        out.append("</div>")
    return "".join(out)


def render_page(title: str, body: str, *, active: Optional[tuple[int, str]] = None) -> str:
    if request.args.get("_partial"):
        return body  # auto-refresh fetches just the <main> contents
    return render_template_string(
        BASE, title=title, body=body, sidebar=_sidebar(active),
        default_refresh=DEFAULT_REFRESH_SEC,
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.route("/")
def home() -> str:
    out = ["<div class='page-head'><h1>Overview</h1>"
           "<div class='muted'>ChromaDB stores discovered on disk.</div></div>"]
    for di, store in enumerate(STORES):
        out.append(f"<div class='section-h'>{e(store.directory)} "
                   f"<span class='muted' style='text-transform:none'>· {e(store.path)}</span></div>")
        if store.error:
            out.append(f"<div class='warn'>{e(store.error)}</div>")
            continue
        colls = store.collections()
        if not colls:
            out.append("<p class='muted'>no collections</p>")
            continue
        out.append("<div class='ov-grid'>")
        for name, cnt in colls:
            top = _top_classes(store, name) if cnt > 0 else []
            chips = "".join(
                f"<span class='chip' style='--h:{_hue(c)}'><i class='dot'></i>{e(c)} <b>{n}</b></span>"
                for c, n in top
            )
            out.append(
                f"<a class='card ov-card' href='{_browse_path(di, name)}'>"
                f"<h3>{e(name)}</h3><div class='big'>{'?' if cnt < 0 else f'{cnt:,}'}</div>"
                f"<div class='muted small'>records</div>"
                f"<div class='topc'>{chips}</div></a>"
            )
        out.append("</div>")
    return render_page("Overview", "".join(out))


def _top_classes(store: Store, name: str, k: int = 3) -> list[tuple[str, int]]:
    try:
        coll = store._client.get_collection(name)  # noqa: SLF001 — local read tool
        res = coll.get(include=["metadatas"], limit=2000)
    except Exception:  # noqa: BLE001
        return []
    counter = Counter(
        str(m.get("class_name")) for m in (res.get("metadatas") or []) if m.get("class_name")
    )
    return counter.most_common(k)


@app.route("/c/<int:di>/<coll>")
def browse(di: int, coll: str) -> str:
    if di < 0 or di >= len(STORES):
        abort(404)
    store = STORES[di]
    collection = store.collection(coll)
    total = collection.count()
    base = _browse_path(di, coll)

    class_filter = request.args.get("class") or None
    where = {"class_name": class_filter} if class_filter else None
    res = collection.get(
        where=where, include=["metadatas", "documents"], limit=BROWSE_FETCH_CAP
    )
    rows_all = _rows(res)

    # class breakdown from a quick scan (independent of the active filter)
    cls_res = collection.get(include=["metadatas"], limit=BROWSE_FETCH_CAP)
    classes = Counter(
        str(m.get("class_name")) for m in (cls_res.get("metadatas") or [])
        if m.get("class_name")
    ).most_common()

    sort = request.args.get("sort")
    direction = request.args.get("dir", "asc")
    if not sort:
        sort, direction = _default_sort(rows_all)
    elif direction not in ("asc", "desc"):
        direction = "asc"
    _sort_rows(rows_all, sort, direction)

    page_num = max(1, request.args.get("page", 1, type=int))
    pages = max(1, math.ceil(len(rows_all) / PAGE_SIZE))
    page_num = min(page_num, pages)
    page_rows = rows_all[(page_num - 1) * PAGE_SIZE: page_num * PAGE_SIZE]

    head = (f"<div class='page-head'><h1>{e(coll)} "
            f"<span class='count'>· {total:,} records</span></h1></div>")
    cap_note = (f"<div class='warn'>Showing the first {BROWSE_FETCH_CAP:,} of "
                f"{total:,} records (sorted/mapped within that sample).</div>"
                if total > BROWSE_FETCH_CAP else "")
    colorby = request.args.get("colorby", "class")
    aspect = request.args.get("aspect", "fill")
    map_html = _render_map(di, coll, rows_all, base, colorby, aspect)
    body = (
        head + _coll_nav(di, coll, "browse")
        + _toolbar(di, coll, q="", mode="substring")
        + _class_chips(base, classes, class_filter) + cap_note + map_html
        + _render_table(di, coll, page_rows, base, sort, direction, linked=bool(map_html))
        + _pager(di, coll, page_num, pages)
    )
    crumb = (f"<div class='crumb'><a href='/'>overview</a> / "
             f"<code>{e(store.directory)}</code> / <b>{e(coll)}</b></div>")
    return render_page(coll, crumb + body, active=(di, coll))


@app.route("/c/<int:di>/<coll>/search")
def search(di: int, coll: str) -> str:
    if di < 0 or di >= len(STORES):
        abort(404)
    store = STORES[di]
    collection = store.collection(coll)
    base = _search_path(di, coll)
    q = (request.args.get("q") or "").strip()
    mode = request.args.get("mode", "substring")
    class_filter = request.args.get("class") or None
    warn = ""
    rows: list[dict] = []
    if q:
        if mode == "semantic":
            sample = _rows(collection.get(include=["metadatas"], limit=5))
            try:
                rows = _semantic_search(collection, q, sample)
            except Exception as ex:  # noqa: BLE001
                warn = f"Semantic search unavailable ({ex}); showing substring matches."
                rows = _substring_search(collection, q)
        else:
            rows = _substring_search(collection, q)

    if class_filter:
        rows = [r for r in rows if str(r["meta"].get("class_name")) == class_filter]

    sort = request.args.get("sort")
    direction = request.args.get("dir", "asc")
    if sort:
        if direction not in ("asc", "desc"):
            direction = "asc"
        _sort_rows(rows, sort, direction)
    else:
        sort = "distance" if any("distance" in r for r in rows) else ""

    classes = Counter(
        str(r["meta"].get("class_name")) for r in rows if r["meta"].get("class_name")
    ).most_common()

    head = (f"<div class='page-head'><h1>Search · {e(coll)} "
            f"<span class='count'>· {len(rows)} result(s)</span></h1></div>")
    warn_html = f"<div class='warn'>{e(warn)}</div>" if warn else ""
    colorby = request.args.get("colorby", "class")
    aspect = request.args.get("aspect", "fill")
    map_html = _render_map(di, coll, rows, base, colorby, aspect)
    agent_html = _agent_panel(q, rows) if q else ""
    body = (
        head + _coll_nav(di, coll, "browse")
        + _toolbar(di, coll, q=q, mode=mode)
        + _class_chips(base, classes, class_filter) + warn_html
        + agent_html + map_html
        + _render_table(di, coll, rows, base, sort, direction, linked=bool(map_html))
    )
    crumb = (f"<div class='crumb'><a href='/'>overview</a> / "
             f"<code>{e(store.directory)}</code> / "
             f"<a href='{_browse_path(di, coll)}'>{e(coll)}</a> / <b>search</b></div>")
    return render_page(f"search · {coll}", crumb + body, active=(di, coll))


@app.route("/c/<int:di>/<coll>/r/<path:rid>")
def record(di: int, coll: str, rid: str) -> str:
    if di < 0 or di >= len(STORES):
        abort(404)
    store = STORES[di]
    collection = store.collection(coll)
    res = collection.get(ids=[rid], include=["metadatas", "documents", "embeddings"])
    rows = _rows(res, with_emb=True)
    if not rows:
        abort(404)
    r = rows[0]
    meta = r["meta"]

    parts = [f"<div class='page-head'><div class='detail-head'>"
             f"<span class='idtxt'>{e(r['id'])}</span>"
             f"<button class='btn ghost' data-copy='{e(r['id'])}'>⧉ copy id</button>"]
    if meta.get("class_name"):
        parts.append(_class_badge(meta["class_name"]))
    parts.append("</div>")
    if r["doc"]:
        parts.append(f"<div class='muted'>{e(r['doc'])}</div>")
    parts.append("</div>")

    frame = _frame_url(meta)
    if frame:
        parts.append(f"<div class='card' style='padding:14px'>"
                     f"<img class='full' data-zoom src='{frame}'></div>")

    parts.append("<div class='section-h'>metadata</div><div class='card'><dl class='meta'>")
    for k in sorted(meta.keys()):
        parts.append(f"<dt>{e(k)}</dt><dd>{_meta_value(k, meta[k])}</dd>")
    parts.append("</dl></div>")

    stats = _emb_stats(r["emb"])
    if stats:
        head = ", ".join(f"{v:.4f}" for v in stats["head"])
        parts.append(
            "<div class='section-h'>embedding</div><div class='card'>"
            f"{_sparkline(stats['vals'])}<dl class='meta'>"
            f"<dt>dim</dt><dd>{stats['dim']}</dd>"
            f"<dt>L2 norm</dt><dd>{stats['norm']:.4f}</dd>"
            f"<dt>min / max</dt><dd>{stats['min']:.4f} / {stats['max']:.4f}</dd>"
            f"<dt>head[0:8]</dt><dd><code>{e(head)}</code></dd></dl></div>"
        )
    else:
        parts.append("<div class='section-h'>embedding</div>"
                     "<div class='card' style='padding:14px'><span class='muted'>"
                     "no embedding stored</span></div>")

    if stats:  # nearest neighbours by this record's own embedding
        try:
            parts.append(_render_similar(di, coll, _similar(collection, r["id"], r["emb"])))
        except Exception as ex:  # noqa: BLE001 — never break the detail page
            parts.append(f"<div class='muted small'>similar lookup failed: {e(ex)}</div>")

    crumb = (f"<div class='crumb'><a href='/'>overview</a> / "
             f"<code>{e(store.directory)}</code> / "
             f"<a href='{_browse_path(di, coll)}'>{e(coll)}</a> / <b>record</b></div>")
    return render_page(str(r["id"]), crumb + "".join(parts), active=(di, coll))


@app.route("/c/<int:di>/<coll>/gallery")
def gallery(di: int, coll: str) -> str:
    if di < 0 or di >= len(STORES):
        abort(404)
    store = STORES[di]
    collection = store.collection(coll)
    total = collection.count()
    base = f"/c/{di}/{quote(coll)}/gallery"
    class_filter = request.args.get("class") or None
    rows = _load_rows(collection, class_filter)
    rows.sort(key=_sort_key("last_seen_ts"), reverse=True)
    classes = _class_counts(collection)

    cards = []
    for r in rows[:300]:
        m = r["meta"]
        furl = _frame_url(m)
        img = (f"<img class='gimg' data-zoom src='{furl}'>" if furl
               else "<div class='gimg noimg'>no image</div>")
        badge = _class_badge(m["class_name"]) if m.get("class_name") else ""
        ts = _ts_html(m.get("last_seen_ts")) if m.get("last_seen_ts") else ""
        cards.append(
            f"<div class='gcard'><a href='{_record_url(di, coll, r['id'])}'>{img}</a>"
            f"<div class='gmeta'>{badge}<span class='muted small'>{ts}</span></div>"
            f"<a class='gid' href='{_record_url(di, coll, r['id'])}'>{e(r['id'])}</a></div>"
        )
    grid = (f"<div class='gal-grid'>{''.join(cards)}</div>" if cards else
            "<div class='empty'><div class='empty-ico'>🖼</div><p>No records.</p></div>")
    head = (f"<div class='page-head'><h1>{e(coll)} "
            f"<span class='count'>· gallery · {total:,} records</span></h1></div>")
    body = (head + _coll_nav(di, coll, "gallery")
            + _class_chips(base, classes, class_filter) + grid)
    return render_page(f"gallery · {coll}", _crumb(store, di, coll, "gallery") + body,
                       active=(di, coll))


@app.route("/c/<int:di>/<coll>/stats")
def stats(di: int, coll: str) -> str:
    if di < 0 or di >= len(STORES):
        abort(404)
    store = STORES[di]
    collection = store.collection(coll)
    total = collection.count()
    rows = _load_rows(collection, None, with_emb=True)
    shown = len(rows)
    classes = _class_counts(collection)

    confs, sights = [], []
    n_pos = n_frame = n_emb = 0
    emb_models: set[str] = set()
    emb_dims: set[str] = set()
    for r in rows:
        m = r["meta"]
        try:
            confs.append(float(m.get("position_conf", m.get("confidence"))))
        except (TypeError, ValueError):
            pass
        try:
            sights.append(float(m.get("sightings")))
        except (TypeError, ValueError):
            pass
        if _position(m) is not None:
            n_pos += 1
        if m.get("frame_ref"):
            n_frame += 1
        if r["emb"] is not None and len(r["emb"]) > 0:
            n_emb += 1
        if m.get("embedding_model"):
            emb_models.add(str(m.get("embedding_model")))
        if m.get("embedding_dim") is not None:
            emb_dims.add(str(m.get("embedding_dim")))

    checks = []

    def chk(ok: bool, label: str) -> None:
        checks.append(f"<li class='{'ok' if ok else 'bad'}'>"
                      f"{'✓' if ok else '⚠'} {e(label)}</li>")

    chk(n_pos == shown, f"{n_pos}/{shown} have a 3D position")
    chk(n_emb == shown, f"{n_emb}/{shown} have an embedding")
    chk(n_frame > 0, f"{n_frame}/{shown} have a frame image")
    chk(len(emb_dims) <= 1, f"embedding dim(s): {', '.join(sorted(emb_dims)) or '—'}")
    chk(len(emb_models) <= 1, f"embedding model(s): {', '.join(sorted(emb_models)) or '—'}")

    def card(title: str, inner: str) -> str:
        return f"<div class='card statcard'><h3 class='stat-h'>{e(title)}</h3>{inner}</div>"

    note = (f"<div class='warn'>Computed over the first {shown:,} of {total:,} "
            f"records.</div>" if total > shown else "")
    grid = ("<div class='statgrid'>"
            + card("Records by class", _bars(classes, hue_by_label=True))
            + card("Confidence", _histogram(confs, lo=0.0, hi=1.0))
            + card("Sightings", _histogram(sights))
            + card("Consistency", f"<ul class='checks'>{''.join(checks)}</ul>")
            + "</div>")
    head = (f"<div class='page-head'><h1>{e(coll)} "
            f"<span class='count'>· stats · {total:,} records</span></h1></div>")
    body = head + _coll_nav(di, coll, "stats") + note + grid
    return render_page(f"stats · {coll}", _crumb(store, di, coll, "stats") + body,
                       active=(di, coll))


@app.route("/c/<int:di>/<coll>/changes")
def changes(di: int, coll: str) -> str:
    if di < 0 or di >= len(STORES):
        abort(404)
    store = STORES[di]
    collection = store.collection(coll)
    base = f"/c/{di}/{quote(coll)}/changes"
    since = request.args.get("since", 300, type=int)
    rows = _load_rows(collection, None)
    now = time.time()
    cutoff = now - since
    have_first = any("first_seen_ts" in r["meta"] for r in rows)

    appeared, refreshed, gone = [], [], []
    for r in rows:
        m = r["meta"]
        try:
            last = float(m.get("last_seen_ts"))
        except (TypeError, ValueError):
            continue
        try:
            first = float(m.get("first_seen_ts"))
        except (TypeError, ValueError):
            first = None
        if have_first and first is not None and first > cutoff:
            appeared.append(r)
        elif last > cutoff:
            refreshed.append(r)
        else:
            gone.append(r)
    for lst in (appeared, refreshed, gone):
        lst.sort(key=_sort_key("last_seen_ts"), reverse=True)

    def feed(items: list[dict]) -> str:
        if not items:
            return "<p class='muted'>none</p>"
        out = []
        for r in items[:100]:
            m = r["meta"]
            furl = _frame_url(m)
            th = f"<img class='thumb' data-zoom src='{furl}'>" if furl else ""
            badge = _class_badge(m["class_name"]) if m.get("class_name") else ""
            out.append(
                f"<a class='feedrow' href='{_record_url(di, coll, r['id'])}'>{th}{badge}"
                f"<span class='muted small'>{_ts_html(m.get('last_seen_ts'))}</span></a>"
            )
        return "".join(out)

    windows = [(60, "1m"), (300, "5m"), (900, "15m"), (3600, "1h"),
               (21600, "6h"), (86400, "24h")]
    picker = "".join(
        f"<a class='mctl{' on' if since == s else ''}' href='{_url(base, since=s)}'>{l}</a>"
        for s, l in windows
    )
    if have_first:
        cols = (f"<div><div class='section-h'>🟢 appeared ({len(appeared)})</div>{feed(appeared)}</div>"
                f"<div><div class='section-h'>🔄 refreshed ({len(refreshed)})</div>{feed(refreshed)}</div>"
                f"<div><div class='section-h'>⚪ disappeared ({len(gone)})</div>{feed(gone)}</div>")
        hint = ""
    else:
        cols = (f"<div><div class='section-h'>🟢 recently seen ({len(refreshed)})</div>{feed(refreshed)}</div>"
                f"<div><div class='section-h'>⚪ stale ({len(gone)})</div>{feed(gone)}</div>")
        hint = ("<div class='muted small' style='margin-bottom:8px'>This collection "
                "has no <code>first_seen_ts</code>, so appeared/refreshed can't be "
                "split — showing recently-seen vs stale.</div>")
    head = (f"<div class='page-head'><h1>{e(coll)} "
            f"<span class='count'>· changes</span></h1></div>")
    body = (head + _coll_nav(di, coll, "changes")
            + f"<div class='map-ctl'><span class='muted'>window:</span>{picker}</div>"
            + hint + f"<div class='changes-grid'>{cols}</div>")
    return render_page(f"changes · {coll}", _crumb(store, di, coll, "changes") + body,
                       active=(di, coll))


@app.route("/frame")
def frame() -> Response:
    ref = request.args.get("ref", "")
    if not ref:
        abort(404)
    p = Path(ref)
    p = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    if not any(str(p).startswith(str(root)) for root in FRAME_ROOTS):
        abort(403)
    if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        abort(404)
    return send_file(p)


def _meta_value(key: str, val: Any) -> str:
    if _is_ts_key(key):
        rt = _rel_time(val)
        if rt:
            return f"{e(rt[0])} <span class='muted'>· {e(rt[1])}</span>"
    if key in ("confidence", "position_conf"):
        try:
            f = float(val)
            return _bar(f, f"{f:.4f}")
        except (TypeError, ValueError):
            pass
    n = _fmt_num(val)
    return e(n) if n is not None else e(val)


def _sparkline(vals: list[float], width: float = 480.0, height: float = 60.0) -> str:
    n = len(vals)
    if n == 0:
        return ""
    step = max(1, n // 160)
    s = vals[::step]
    lo, hi = min(s), max(s)
    rng = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(s):
        x = (i / (len(s) - 1) * width) if len(s) > 1 else 0.0
        y = height - (v - lo) / rng * height
        pts.append(f"{x:.1f},{y:.1f}")
    mid = height - (0 - lo) / rng * height if lo <= 0 <= hi else None
    midline = (f"<line class='mid' x1='0' y1='{mid:.1f}' x2='{width:.0f}' y2='{mid:.1f}'/>"
               if mid is not None else "")
    return (f"<svg class='spark' viewBox='0 0 {width:.0f} {height:.0f}' "
            f"preserveAspectRatio='none'>{midline}"
            f"<path d='M{' L'.join(pts)}'/></svg>")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only ChromaDB web viewer")
    ap.add_argument(
        "--dirs",
        default=os.getenv("CHROMA_VIEWER_DIRS", "chroma_db,chroma_db_scene"),
        help="comma-separated Chroma persist directories",
    )
    ap.add_argument(
        "--port", type=int, default=int(os.getenv("CHROMA_VIEWER_PORT", "8500"))
    )
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    _build_stores(dirs)
    _build_frame_roots(dirs, os.getenv("SCENE_FRAMES_DIR", "frames"))

    print(f"[chroma-viewer] dirs={dirs}")
    for s in STORES:
        status = s.error or f"{len(s.collections())} collection(s)"
        print(f"[chroma-viewer]   {s.directory}: {status}")
    print(f"[chroma-viewer] open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
