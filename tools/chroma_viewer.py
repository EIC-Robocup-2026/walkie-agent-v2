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
a ``frame_ref`` pointing at a readable image.

It is strictly **read-only** — no add/update/delete routes exist — so it is
safe to run against a directory the robot is actively writing to (SQLite
handles the concurrent reads).

Run it::

    uv run python -m tools.chroma_viewer
    uv run python -m tools.chroma_viewer --dirs chroma_db,chroma_db_scene --port 8500

then open http://localhost:8500.

Environment overrides (CLI flags win):
    CHROMA_VIEWER_DIRS   comma-separated dirs   (default: chroma_db,chroma_db_scene)
    CHROMA_VIEWER_PORT   port                   (default: 8500)
    SCENE_FRAMES_DIR     extra image root       (default: frames)
    WALKIE_AI_BASE_URL   for CLIP semantic search on scene_entries
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from flask import Flask, Response, abort, render_template_string, request, send_file

# Pick up .env (CHROMA_VIEWER_*, SCENE_FRAMES_DIR, WALKIE_AI_BASE_URL) at import
# time, before the module-level getenv reads below, mirroring main.py.
load_dotenv()

PAGE_SIZE = 50
SEARCH_SCAN_CAP = 5000  # max rows scanned for a substring search
# Initial auto-refresh interval (seconds) used until the browser overrides it
# via the header dropdown (persisted per-browser in localStorage). 0 = off.
DEFAULT_REFRESH_SEC = os.getenv("CHROMA_VIEWER_REFRESH_SEC", "5")

# Metadata keys promoted to table columns, in display order, when present.
# Everything else is still shown on the per-record detail page.
PRIORITY_KEYS = [
    "class_name",
    "confidence",
    "position_conf",
    "sightings",
    "caption",
    "last_seen_ts",
    "first_seen_ts",
    "embedding_model",
]

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
    return {
        "dim": len(vals),
        "norm": norm,
        "min": min(vals),
        "max": max(vals),
        "head": vals[:8],
    }


def _columns_for(rows: list[dict]) -> list[str]:
    present = set()
    for r in rows:
        present.update(r["meta"].keys())
    return [k for k in PRIORITY_KEYS if k in present]


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def _substring_search(coll, q: str) -> list[dict[str, Any]]:
    q = q.lower()
    res = coll.get(
        include=["metadatas", "documents"],
        limit=SEARCH_SCAN_CAP,
    )
    rows = _rows(res)
    hits = []
    for r in rows:
        hay = (r["doc"] + " " + " ".join(str(v) for v in r["meta"].values())).lower()
        if q in hay:
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
        out.append(
            {"id": rid, "meta": dict(meta or {}), "doc": doc or "",
             "emb": None, "distance": dist}
        )
    return out


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

BASE = """
<!doctype html><html><head><meta charset="utf-8">
<title>{{ title }} · Chroma Viewer</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.5 system-ui, sans-serif; margin: 0; }
  header { background:#1f2933; color:#fff; padding:10px 16px; display:flex;
           gap:16px; align-items:center; flex-wrap:wrap; }
  header a { color:#9ecbff; text-decoration:none; }
  header .crumb { color:#cbd5e1; }
  main { padding:16px; }
  table { border-collapse:collapse; width:100%; }
  th, td { border-bottom:1px solid #8884; padding:6px 8px; text-align:left;
           vertical-align:top; font-size:13px; }
  th { background:#8881; position:sticky; top:0; }
  td.id a { font-family:ui-monospace,monospace; }
  .doc { max-width:340px; }
  .thumb { height:54px; border-radius:4px; }
  .chip { display:inline-block; background:#8882; border-radius:10px;
          padding:1px 8px; margin:2px 4px 2px 0; font-size:12px; }
  .muted { color:#888; }
  .err { color:#c0392b; }
  .warn { background:#f39c1222; border:1px solid #f39c12; padding:8px 12px;
          border-radius:6px; margin-bottom:12px; }
  form.search { margin-bottom:14px; display:flex; gap:8px; flex-wrap:wrap; }
  input[type=text]{ padding:6px 10px; min-width:260px; }
  button, select { padding:6px 10px; }
  .pager a { padding:4px 10px; border:1px solid #8884; border-radius:4px;
             text-decoration:none; margin-right:6px; }
  dl.meta { display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; }
  dl.meta dt { color:#888; font-family:ui-monospace,monospace; }
  dl.meta dd { margin:0; word-break:break-word; }
  img.full { max-width:480px; border-radius:6px; }
  code { font-family:ui-monospace,monospace; }
  .rt { margin-left:auto; display:flex; align-items:center; gap:6px;
        font-size:13px; color:#cbd5e1; }
  .rt #rt-dot { width:8px; height:8px; border-radius:50%; background:#475569; }
  .rt.live #rt-dot { background:#22c55e; box-shadow:0 0 6px #22c55e; }
  .rt select { padding:2px 6px; }
</style></head><body>
<header>
  <a href="/"><b>Chroma Viewer</b></a>
  <span class="crumb">{{ crumb|safe }}</span>
  <span class="rt" id="rt"><span id="rt-dot"></span>auto-refresh:
    <select id="rt-sel">
      <option value="0">off</option>
      <option value="2">2s</option>
      <option value="5">5s</option>
      <option value="10">10s</option>
      <option value="30">30s</option>
    </select>
  </span>
</header>
<main>{{ body|safe }}</main>
<script>
(function () {
  var KEY = 'chromaViewerRefresh';
  var sel = document.getElementById('rt-sel');
  var box = document.getElementById('rt');
  var timer = null;
  var cur = localStorage.getItem(KEY);
  if (cur === null) cur = '{{ default_refresh }}';
  sel.value = cur;

  function arm() {
    if (timer) { clearTimeout(timer); timer = null; }
    var s = parseInt(localStorage.getItem(KEY) || sel.value || '0', 10);
    box.classList.toggle('live', s > 0);
    if (s > 0) timer = setTimeout(tick, s * 1000);
  }
  function tick() {
    // Don't reload while the user is mid-type in the search box.
    var ae = document.activeElement;
    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'SELECT')) { arm(); return; }
    location.reload();
  }
  sel.addEventListener('change', function () {
    localStorage.setItem(KEY, sel.value);
    arm();
  });
  arm();
})();
</script>
</body></html>
"""


def page(title: str, crumb: str, body: str) -> str:
    return render_template_string(
        BASE, title=title, crumb=crumb, body=body,
        default_refresh=DEFAULT_REFRESH_SEC,
    )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _row_cells(r: dict, columns: list[str]) -> str:
    cells = []
    for k in columns:
        cells.append("<td>" + _fmt(r["meta"].get(k, "")) + "</td>")
    return "".join(cells)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.route("/")
def home() -> str:
    parts = ["<h2>Stores</h2>"]
    for di, store in enumerate(STORES):
        parts.append(f"<h3><code>{store.directory}</code> "
                     f"<span class='muted'>{store.path}</span></h3>")
        if store.error:
            parts.append(f"<p class='err'>{store.error}</p>")
            continue
        colls = store.collections()
        if not colls:
            parts.append("<p class='muted'>no collections</p>")
            continue
        parts.append("<table><tr><th>collection</th><th>records</th></tr>")
        for name, cnt in colls:
            cnt_txt = "?" if cnt < 0 else f"{cnt:,}"
            parts.append(
                f"<tr><td><a href='/c/{di}/{quote(name)}'>{name}</a></td>"
                f"<td>{cnt_txt}</td></tr>"
            )
        parts.append("</table>")
    return page("Stores", "", "".join(parts))


@app.route("/c/<int:di>/<coll>")
def browse(di: int, coll: str) -> str:
    if di < 0 or di >= len(STORES):
        abort(404)
    store = STORES[di]
    collection = store.collection(coll)
    total = collection.count()
    page_num = max(1, request.args.get("page", 1, type=int))
    offset = (page_num - 1) * PAGE_SIZE
    res = collection.get(
        include=["metadatas", "documents"], limit=PAGE_SIZE, offset=offset
    )
    rows = _rows(res)
    body = _render_table(di, coll, rows, total=total, page_num=page_num)
    crumb = f"→ <code>{store.directory}</code> / <b>{coll}</b>"
    return page(coll, crumb, body)


@app.route("/c/<int:di>/<coll>/search")
def search(di: int, coll: str) -> str:
    if di < 0 or di >= len(STORES):
        abort(404)
    store = STORES[di]
    collection = store.collection(coll)
    q = (request.args.get("q") or "").strip()
    mode = request.args.get("mode", "substring")
    warn = ""
    rows: list[dict] = []
    if q:
        if mode == "semantic":
            sample = _rows(collection.get(include=["metadatas"], limit=5))
            try:
                rows = _semantic_search(collection, q, sample)
            except Exception as e:  # noqa: BLE001
                warn = (f"Semantic search unavailable ({e}); "
                        "showing substring matches instead.")
                rows = _substring_search(collection, q)
        else:
            rows = _substring_search(collection, q)
    body = _render_table(
        di, coll, rows, total=len(rows), page_num=1, q=q, mode=mode, warn=warn
    )
    crumb = f"→ <code>{store.directory}</code> / <b>{coll}</b> / search"
    return page(f"search · {coll}", crumb, body)


@app.route("/c/<int:di>/<coll>/r/<path:rid>")
def record(di: int, coll: str, rid: str) -> str:
    if di < 0 or di >= len(STORES):
        abort(404)
    store = STORES[di]
    collection = store.collection(coll)
    res = collection.get(
        ids=[rid], include=["metadatas", "documents", "embeddings"]
    )
    rows = _rows(res, with_emb=True)
    if not rows:
        abort(404)
    r = rows[0]
    parts = [f"<h2 class='id'><code>{r['id']}</code></h2>"]
    if r["doc"]:
        parts.append(f"<p><b>document:</b> {r['doc']}</p>")

    frame = _frame_url(r["meta"])
    if frame:
        parts.append(f"<p><a href='{frame}'><img class='full' src='{frame}'></a></p>")

    parts.append("<h3>metadata</h3><dl class='meta'>")
    for k in sorted(r["meta"].keys()):
        parts.append(f"<dt>{k}</dt><dd>{_fmt(r['meta'][k])}</dd>")
    parts.append("</dl>")

    stats = _emb_stats(r["emb"])
    if stats:
        head = ", ".join(f"{v:.4f}" for v in stats["head"])
        parts.append(
            "<h3>embedding</h3><dl class='meta'>"
            f"<dt>dim</dt><dd>{stats['dim']}</dd>"
            f"<dt>L2 norm</dt><dd>{stats['norm']:.4f}</dd>"
            f"<dt>min / max</dt><dd>{stats['min']:.4f} / {stats['max']:.4f}</dd>"
            f"<dt>head[0:8]</dt><dd><code>{head}</code></dd>"
            "</dl>"
        )
    else:
        parts.append("<p class='muted'>no embedding stored</p>")

    crumb = (f"→ <code>{store.directory}</code> / "
             f"<a href='/c/{di}/{quote(coll)}'>{coll}</a> / record")
    return page(r["id"], crumb, "".join(parts))


@app.route("/frame")
def frame() -> Response:
    ref = request.args.get("ref", "")
    if not ref:
        abort(404)
    p = Path(ref)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if not any(str(p).startswith(str(root)) for root in FRAME_ROOTS):
        abort(403)
    if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        abort(404)
    return send_file(p)


def _render_table(
    di: int,
    coll: str,
    rows: list[dict],
    *,
    total: int,
    page_num: int,
    q: str = "",
    mode: str = "substring",
    warn: str = "",
) -> str:
    columns = _columns_for(rows)
    sel_sub = "selected" if mode != "semantic" else ""
    sel_sem = "selected" if mode == "semantic" else ""
    parts = [
        f"<form class='search' action='/c/{di}/{quote(coll)}/search' method='get'>"
        f"<input type='text' name='q' placeholder='search…' value='{q}'>"
        f"<select name='mode'>"
        f"<option value='substring' {sel_sub}>substring</option>"
        f"<option value='semantic' {sel_sem}>semantic</option>"
        f"</select>"
        f"<button type='submit'>Search</button>"
        f"<a class='chip' href='/c/{di}/{quote(coll)}'>browse all</a>"
        f"</form>"
    ]
    if warn:
        parts.append(f"<div class='warn'>{warn}</div>")
    has_dist = any("distance" in r for r in rows)
    has_frame = any(_frame_url(r["meta"]) for r in rows)
    parts.append(f"<p class='muted'>{len(rows)} shown · {total:,} total</p>")
    parts.append("<table><tr><th>id</th>")
    if has_frame:
        parts.append("<th>frame</th>")
    parts.append("<th>document</th>")
    for k in columns:
        parts.append(f"<th>{k}</th>")
    if has_dist:
        parts.append("<th>distance</th>")
    parts.append("</tr>")

    for r in rows:
        rid_url = quote(str(r["id"]))
        parts.append(
            f"<tr><td class='id'><a href='/c/{di}/{quote(coll)}/r/{rid_url}'>"
            f"{r['id']}</a></td>"
        )
        if has_frame:
            furl = _frame_url(r["meta"])
            parts.append(
                f"<td><img class='thumb' src='{furl}'></td>" if furl else "<td></td>"
            )
        parts.append(f"<td class='doc'>{r['doc']}</td>")
        parts.append(_row_cells(r, columns))
        if has_dist:
            d = r.get("distance")
            parts.append(f"<td>{d:.4f}</td>" if d is not None else "<td></td>")
        parts.append("</tr>")
    parts.append("</table>")

    # Pager only on plain browse (search returns a single ranked page).
    if not q:
        last = max(1, math.ceil(total / PAGE_SIZE))
        nav = ["<p class='pager'>"]
        if page_num > 1:
            nav.append(f"<a href='/c/{di}/{quote(coll)}?page={page_num-1}'>← prev</a>")
        nav.append(f"<span class='muted'>page {page_num} / {last}</span> ")
        if page_num < last:
            nav.append(f"<a href='/c/{di}/{quote(coll)}?page={page_num+1}'>next →</a>")
        nav.append("</p>")
        parts.append("".join(nav))
    return "".join(parts)


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
