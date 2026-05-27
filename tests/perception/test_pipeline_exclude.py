"""process_frame's exclude_classes filter (keeps moving classes like people
out of long-term memory — they can't be position-deduped and just duplicate)."""

from __future__ import annotations

from perception.mocks import (
    FakeCaptioner,
    FakeDetectedObject,
    FakeEmbedder,
    FakePositionLifter,
    make_tiny_image,
)
from perception.pipeline import process_frame


def _detector(classes):
    from perception.mocks import FakeDetector

    objs = [
        FakeDetectedObject(class_name=c, class_id=i, confidence=0.9, bbox=(0, 0, 10, 10))
        for i, c in enumerate(classes)
    ]
    return FakeDetector([objs])


def _run(classes, exclude):
    return process_frame(
        make_tiny_image(0),
        detector=_detector(classes),
        lifter=FakePositionLifter(default=[1.0, 1.0, 0.5]),
        captioner=FakeCaptioner("a scene"),
        embedder=FakeEmbedder(),
        exclude_classes=exclude,
    )[0]


def test_excluded_class_is_dropped():
    out = _run(["person", "chair", "person"], exclude=["person"])
    assert [d.class_name for d in out] == ["chair"]


def test_exclude_is_case_insensitive():
    out = _run(["Person", "Chair"], exclude=["PERSON"])
    assert [d.class_name for d in out] == ["Chair"]


def test_no_exclude_keeps_everything():
    out = _run(["person", "chair"], exclude=None)
    assert sorted(d.class_name for d in out) == ["chair", "person"]


def test_all_excluded_returns_empty():
    out = _run(["person", "person"], exclude=["person"])
    assert out == []
