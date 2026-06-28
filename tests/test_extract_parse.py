"""Tolerant LLM-output parsing for ctx.extract (tasks/base.py::_parse_to_schema).

Local LLM backends (LLM_USE_LOCAL, e.g. a vLLM Qwen) frequently answer order
extraction with a bare Python-style array — ``['coke']`` — instead of the schema
object ``{"items": ["coke"]}``. The strict json_schema path then raises:

    Invalid JSON: expected value at line 1 column 2 ... input_value="['coke']"

_parse_to_schema recovers these (bare array -> sole list field; single-quoted
Python literals; code fences) so a heard order still parses instead of being lost.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from tasks.base import _parse_to_schema, _schema_list_field


class Order(BaseModel):
    items: list[str] = Field(default_factory=list)


class Multi(BaseModel):
    name: str = ""
    tags: list[str] = Field(default_factory=list)


def test_bare_single_quoted_array_wraps_into_sole_list_field():
    """The exact failing case: model returned ['coke']."""
    out = _parse_to_schema("['coke']", Order)
    assert isinstance(out, Order) and out.items == ["coke"]


def test_bare_json_array_two_items():
    out = _parse_to_schema('["coke", "fries"]', Order)
    assert out.items == ["coke", "fries"]


def test_proper_json_object_still_works():
    out = _parse_to_schema('{"items": ["coke", "water"]}', Order)
    assert out.items == ["coke", "water"]


def test_single_quoted_python_dict_literal():
    out = _parse_to_schema("{'items': ['coke']}", Order)
    assert out.items == ["coke"]


def test_object_wins_over_prose_and_code_fence():
    out = _parse_to_schema("Sure!\n```json\n{\"items\": [\"tea\"]}\n```", Order)
    assert out.items == ["tea"]


def test_array_embedded_in_prose():
    out = _parse_to_schema("The order is ['coke', 'rice'] I think", Order)
    assert out.items == ["coke", "rice"]


def test_unparseable_returns_none():
    assert _parse_to_schema("uh, I'm not sure", Order) is None


def test_bare_array_not_wrapped_when_multiple_list_fields_ambiguous():
    """A bare array is only wrapped when there's exactly ONE list field."""
    # Multi has a single list field (tags) -> bare array wraps into it.
    assert _schema_list_field(Multi) == "tags"
    out = _parse_to_schema("['a', 'b']", Multi)
    assert out is not None and out.tags == ["a", "b"]
