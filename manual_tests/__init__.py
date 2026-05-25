"""Manual demo / smoke scripts — NOT the pytest suite.

These exercise the walkie-ai-server clients against a *live* server and a
local webcam (or robot). They are interactive (windows, loops, real I/O), so
they live outside ``tests/`` (pyproject's ``testpaths``) and pytest never
collects them. Run one explicitly from the repo root, e.g.::

    uv run python -m manual_tests.test_object_detection

The automated, offline unit tests live in ``tests/`` and run with ``pytest``.
"""
