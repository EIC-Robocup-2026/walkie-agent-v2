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
TABLE_KEYS = [
    "class_name",
    "confidence",
    "position_conf",
    "sightings",
    "last_seen_ts",
    "embedding_model",
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
                  cur_sort: str, cur_dir: str) -> str:
    if not rows:
        return ("<div class='empty'><div class='empty-ico'>∅</div>"
                "<p>No records to show.</p>"
                "<p class='muted'>This collection is empty, or nothing matched "
                "your filter. New rows appear here as the robot writes them.</p></div>")
    cols = _columns(rows)
    out = ["<div class='card tablewrap'><table><thead><tr>"]
    out += [_header(base, c, cur_sort, cur_dir) for c in cols]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        out += [_cell(di, coll, c, r) for c in cols]
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _render_map(di: int, coll: str, rows: list[dict]) -> str:
    pts = []
    for r in rows:
        pos = _position(r["meta"])
        if pos:
            pts.append((pos[0], pos[1], str(r["meta"].get("class_name", "?")), str(r["id"])))
    if len(pts) < 1:
        return ""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    minx, maxx = min(minx, 0.0), max(maxx, 0.0)  # keep origin in view
    miny, maxy = min(miny, 0.0), max(maxy, 0.0)
    if maxx - minx < 1e-6:
        minx, maxx = minx - 1, maxx + 1
    if maxy - miny < 1e-6:
        miny, maxy = miny - 1, maxy + 1
    W, H, pad = 680.0, 420.0, 28.0
    px = lambda x: pad + (x - minx) / (maxx - minx) * (W - 2 * pad)
    py = lambda y: H - (pad + (y - miny) / (maxy - miny) * (H - 2 * pad))  # flip y up

    svg = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' class='map' "
           f"preserveAspectRatio='xMidYMid meet'>"]
    svg.append(f"<rect x='0' y='0' width='{W:.0f}' height='{H:.0f}' class='map-bg'/>")
    # axes through origin
    if minx <= 0 <= maxx:
        svg.append(f"<line x1='{px(0):.1f}' y1='0' x2='{px(0):.1f}' y2='{H:.0f}' class='map-axis'/>")
    if miny <= 0 <= maxy:
        svg.append(f"<line x1='0' y1='{py(0):.1f}' x2='{W:.0f}' y2='{py(0):.1f}' class='map-axis'/>")
    # origin (robot start)
    if minx <= 0 <= maxx and miny <= 0 <= maxy:
        svg.append(f"<circle cx='{px(0):.1f}' cy='{py(0):.1f}' r='5' class='map-origin'/>"
                   f"<text x='{px(0) + 8:.1f}' y='{py(0) - 8:.1f}' class='map-olabel'>origin</text>")
    for x, y, cls, rid in pts[:1000]:
        svg.append(
            f"<a href='{_record_url(di, coll, rid)}'>"
            f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' r='6' "
            f"style='fill:hsl({_hue(cls)} 70% 55%)' class='map-pt'>"
            f"<title>{e(cls)} — ({x:.2f}, {y:.2f})\n{e(rid)}</title></circle></a>"
        )
    svg.append("</svg>")
    classes = sorted({p[2] for p in pts})
    legend = "".join(
        f"<span class='leg'><i style='background:hsl({_hue(c)} 70% 55%)'></i>{e(c)}</span>"
        for c in classes
    )
    note = (f"x: {minx:.2f}…{maxx:.2f} · y: {miny:.2f}…{maxy:.2f} (m, map frame)")
    return (
        "<details class='card map-card' open><summary>🗺 Position map "
        f"<span class='muted'>· {len(pts)} located</span></summary>"
        f"<div class='map-body'>{''.join(svg)}<div class='legend'>{legend}</div>"
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
# Page shell
# --------------------------------------------------------------------------- #

BASE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} · Chroma Viewer</title>
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
  body{font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
       background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
  a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}
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
  .c-id a{font-family:ui-monospace,monospace;font-size:12px}
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
     font-weight:600;user-select:none} .map-body{padding:0 14px 14px}
  svg.map{width:100%;height:auto;border-radius:9px;border:1px solid var(--border);display:block}
  .map-bg{fill:var(--bg)} .map-axis{stroke:var(--border);stroke-width:1;stroke-dasharray:4 4}
  .map-pt{cursor:pointer;stroke:var(--card);stroke-width:1.5;transition:r .1s}
  .map-pt:hover{r:9} .map-origin{fill:none;stroke:var(--accent);stroke-width:2}
  .map-olabel{fill:var(--muted);font-size:11px}
  .legend{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 2px}
  .leg{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
  .leg i{width:10px;height:10px;border-radius:50%}

  dl.meta{display:grid;grid-template-columns:max-content 1fr;gap:7px 20px;margin:0;padding:14px 16px}
  dl.meta dt{color:var(--muted);font-family:ui-monospace,monospace;font-size:12.5px}
  dl.meta dd{margin:0;word-break:break-word}
  .detail-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px}
  .detail-head .idtxt{font-family:ui-monospace,monospace;font-size:13px;
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
  document.addEventListener('click',function(ev){
    var z=ev.target.closest&&ev.target.closest('[data-zoom]');
    if(z){ev.preventDefault();lbi.src=z.getAttribute('src');lb.hidden=false;return;}
    var c=ev.target.closest&&ev.target.closest('[data-copy]');
    if(c){navigator.clipboard&&navigator.clipboard.writeText(c.getAttribute('data-copy'));
      c.textContent='✓';setTimeout(function(){c.textContent='⧉';},900);return;}
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
      if(r.ok){var t=await r.text();if(m){m.innerHTML=t;m.scrollTop=top;}}}
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
    body = (
        head + _toolbar(di, coll, q="", mode="substring")
        + _class_chips(base, classes, class_filter) + cap_note
        + _render_map(di, coll, rows_all)
        + _render_table(di, coll, page_rows, base, sort, direction)
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
    body = (
        head + _toolbar(di, coll, q=q, mode=mode)
        + _class_chips(base, classes, class_filter) + warn_html
        + _render_map(di, coll, rows)
        + _render_table(di, coll, rows, base, sort, direction)
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

    crumb = (f"<div class='crumb'><a href='/'>overview</a> / "
             f"<code>{e(store.directory)}</code> / "
             f"<a href='{_browse_path(di, coll)}'>{e(coll)}</a> / <b>record</b></div>")
    return render_page(str(r["id"]), crumb + "".join(parts), active=(di, coll))


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
