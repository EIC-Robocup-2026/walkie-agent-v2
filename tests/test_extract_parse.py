"""Tolerant LLM-output parsing for ctx.extract (tasks/base.py::_parse_to_schema).

Local LLM backends (LLM_USE_LOCAL, e.g. a vLLM Qwen) frequently answer order
extraction with a bare Python-style array — ``['coke']`` — instead of the schema
object ``{"items": ["coke"]}``. The strict json_schema path then raises:

    Invalid JSON: expected value at line 1 column 2 ... input_value="['coke']"

_parse_to_schema recovers these (bare array -> sole list field; single-quoted
Python literals; code fences) so a heard order still parses instead of being lost.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

from tasks.base import (
    _parse_to_schema,
    _schema_example,
    _schema_list_field,
    _schema_prompt,
)


class Order(BaseModel):
    items: list[str] = Field(default_factory=list)


class Multi(BaseModel):
    name: str = ""
    tags: list[str] = Field(default_factory=list)


class Step(BaseModel):
    primitive: Literal["navigate", "pick", "say"] = Field(description="the action")
    object: Optional[str] = Field(None, description="object referenced")
    raw: str = Field(description="source clause")


class Plan(BaseModel):
    steps: list[Step] = Field(default_factory=list, description="ordered steps")


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


# --- _schema_prompt: show an EXAMPLE instance, never the raw JSON Schema ----
# The on-robot bug: dumping schema.model_json_schema() made the local model parrot
# the schema back (answer buried in properties.items.items), validated as empty.

def test_schema_prompt_shows_example_instance_not_raw_schema():
    p = _schema_prompt("Parse the order.", Order)
    assert '{"items": ["..."]}' in p          # a concrete instance of the shape
    assert "properties" not in p and "model_json_schema" not in p  # not the schema dump
    assert "Parse the order." in p


def test_schema_example_optional_is_null_literal_is_first_choice():
    ex = {n: _schema_example(f.annotation) for n, f in Step.model_fields.items()}
    assert ex["primitive"] == "navigate"   # Literal -> first choice
    assert ex["object"] is None            # Optional[str] -> null
    assert ex["raw"] == "..."              # required str -> placeholder


def test_schema_prompt_lists_literal_choices_and_nests_models():
    p = _schema_prompt("Plan it.", Plan)
    # nested list-of-model is expanded and the Literal's choices are spelled out
    assert "navigate | pick | say" in p
    assert '"primitive"' in p and '"raw"' in p
    # the example object is valid JSON of the right outer shape
    example_str = p.split("EXACTLY this shape:\n", 1)[1].split("\n", 1)[0]
    example = json.loads(example_str)
    assert example["steps"][0]["primitive"] == "navigate"
    assert example["steps"][0]["object"] is None


def test_schema_echo_round_trips_to_empty_without_the_fix():
    """Documents the failure mode: when a model echoes the JSON Schema, the strict
    validator reads an empty object (the symptom the example-prompt fix removes)."""
    echoed = json.dumps(Order.model_json_schema())
    out = _parse_to_schema(echoed, Order)
    assert out is not None and out.items == []  # answer was lost -> empty order
