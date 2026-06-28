"""Shared harness for the local-LLM ability tests (``test_llm.py`` and
``test_llm_hard.py``).

NOT a test module (no ``test_`` prefix) so pytest won't collect it. Holds the
endpoint/model resolution, the skip gate, the ``ChatOpenAI`` builder, a raw chat
helper (langchain drops vLLM's ``reasoning`` field), and the generic
structured-extraction helper used by both suites.

Configured via env (mirrors main.build_model):
    LOCAL_BASE_URL  (default http://localhost:8000/v1)
    LOCAL_MODEL     (default qwen3.5-9b)
    MODEL_API_KEY   (default EMPTY)
    LLM_TEST_MAX_TOKENS (default 2048 — reasoning models need headroom)
"""
from __future__ import annotations

import json
import os
import urllib.request

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

load_dotenv()

BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("MODEL_API_KEY", "EMPTY")
MAX_TOKENS = int(os.getenv("LLM_TEST_MAX_TOKENS", "2048"))


def _served_models(base_url: str) -> list[str] | None:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        return None


_SERVED = _served_models(BASE_URL)
_WANT = os.getenv("LOCAL_MODEL", "qwen3.5-9b")
if _SERVED and _WANT in _SERVED:
    MODEL = _WANT
elif _SERVED and len(_SERVED) == 1:
    MODEL = _SERVED[0]
else:
    MODEL = _WANT

# Each test module sets `pytestmark = SKIP`.
SKIP = pytest.mark.skipif(
    not _SERVED or MODEL not in _SERVED,
    reason=f"local LLM not reachable at {BASE_URL} serving a known model "
    f"(wanted {_WANT!r}, server offers {_SERVED}); skipped offline",
)

# Reasoning models put a <think> trace in the output that corrupts guided JSON;
# the agent suppresses it for extraction (eval_llm_compare.build_chat) — mirror it.
_NO_THINK = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def _model(**kw):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=BASE_URL, api_key=API_KEY, model=MODEL,
        temperature=0, max_tokens=MAX_TOKENS, timeout=180, **kw,
    )


def _raw_chat(content: str, *, enable_thinking: bool = False, max_tokens: int = 1024) -> dict:
    """Raw /chat/completions returning the message dict — used where langchain
    drops fields (vLLM emits the reasoning trace under ``reasoning``) and to
    toggle the chat-template thinking flag."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]


def _tool_calls(msg: AIMessage) -> list[dict]:
    return list(getattr(msg, "tool_calls", []) or [])


def _parse_obj(content, schema, required_keys):
    """Pull the first JSON object out of model text and validate it against
    schema, rejecting a schema echo (asked-for keys absent)."""
    import re

    match = re.search(r"\{.*\}", str(content), re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if any(k not in data for k in required_keys):
            return None
        return schema.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


def _extract(schema, prompt, shape_hint, required_keys, ok):
    """How the agent extracts a typed object (tasks/base.extract): native
    with_structured_output → full-schema JSON prompt → simple concrete JSON
    SHAPE (small models echo the nested schema but handle a plain shape).
    Returns (obj | None, which_path)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        out = _model(**_NO_THINK).with_structured_output(schema).invoke(prompt)
        if isinstance(out, schema) and ok(out):
            return out, "native"
    except Exception:  # noqa: BLE001
        pass

    attempts = [
        ("json-schema", "Respond ONLY with a JSON object matching this schema:\n"
         + json.dumps(schema.model_json_schema())),
        ("json-shape", f"Return ONLY a JSON object of the form {shape_hint} "
         "extracted from the text."),
    ]
    for tag, sys_prompt in attempts:
        try:
            reply = _model(**_NO_THINK).invoke(
                [SystemMessage(content=sys_prompt), HumanMessage(content=prompt)]
            )
            obj = _parse_obj(reply.content, schema, required_keys)
            if obj is not None and ok(obj):
                return obj, tag
        except Exception:  # noqa: BLE001
            continue
    return None, "none"


# --- shared schemas ---------------------------------------------------------
class Step(BaseModel):
    """One step of a robot command."""
    action: str = Field(description="the verb, e.g. 'go', 'pick', 'bring'")
    target: str = Field(description="the object or location the action applies to")


class Plan(BaseModel):
    """An ordered plan parsed from a natural-language robot command."""
    steps: list[Step] = Field(description="ordered steps to execute the command")


Plan.model_rebuild()  # resolve the forward ref to Step regardless of load order
