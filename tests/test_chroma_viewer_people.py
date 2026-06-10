"""Viewer-side tests for the people table: notes/attributes columns and the
None-metadata guard.

Embedding-only collections (``people_appearance``, ``scene_captions``) store
records without metadata — ChromaDB returns ``None`` for those entries, not
``{}``. The class-breakdown Counters must skip them instead of crashing the
page (regression: the home page 500'd the moment a ``people_appearance``
collection existed).
"""

import os
from types import SimpleNamespace as NS

# tools.chroma_viewer is an entrypoint: importing it runs load_dotenv() +
# load_config(), which setdefault()s every config.toml leaf into os.environ.
# That would leak SCENE_* values into the rest of the pytest session and
# break tests that assert the code defaults (tests/perception). Both loaders
# only ADD keys (never overwrite), so dropping the added keys restores the
# pre-import environment exactly.
_pre_import_env = set(os.environ)
from tools.chroma_viewer import _cell, _class_counts, _columns, _rows  # noqa: E402

for _k in set(os.environ) - _pre_import_env:
    del os.environ[_k]


def _row(meta):
    return {"id": "x", "meta": meta, "doc": "", "emb": None}


def test_people_columns_include_attributes_and_notes():
    rows = [_row({"name": "Alice", "drink": "cola", "attributes": "blue shirt", "notes": "from Bangkok"})]
    cols = _columns(rows)
    assert "attributes" in cols and "notes" in cols


def test_notes_cell_renders_one_bullet_per_fact():
    r = _row({"notes": "from Bangkok\nlikes football"})
    html = _cell(0, "people", "notes", r)
    assert "· from Bangkok" in html and "· likes football" in html
    assert "<br>" in html


def test_empty_notes_cell_is_blank():
    assert _cell(0, "people", "notes", _row({"notes": ""})) == "<td></td>"
    assert _cell(0, "people", "notes", _row({})) == "<td></td>"


def test_class_counts_skips_none_metadata():
    fake = NS(get=lambda **kw: {"metadatas": [None, {"class_name": "cup"}, None, {}]})
    assert _class_counts(fake) == [("cup", 1)]


def test_rows_tolerates_none_metadata():
    res = {"ids": ["a", "b"], "metadatas": [None, {"name": "Bob"}], "documents": ["", "Bob"]}
    rows = _rows(res)
    assert rows[0]["meta"] == {}
    assert rows[1]["meta"]["name"] == "Bob"
