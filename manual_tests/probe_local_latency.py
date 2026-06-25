"""Diagnose WHY the local model is slow on GPSR parsing — one cheap measurement.

Per command, GPSR parse is a single structured-output LLM call (tasks/GPSR/parse.py).
23s/cmd is therefore one generation. This splits the two possible causes so we
fix the right thing:

  * big completion_tokens + a <think> block  -> reasoning-bound -> CLIENT fix here
    (enable_thinking=False + max_tokens cap), measurable without touching the box.
  * small output but low tok/s               -> throughput-bound -> fix on the
    serving box (FP8/AWQ quant, spec decoding, right-size the model).

Run (box 10.0.0.202 must be up):
    uv run python -m manual_tests.probe_local_latency
    uv run python -m manual_tests.probe_local_latency --thinking off   # A/B the toggle
"""

from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv

# A few real corpus lines incl. two of the 84%-run's misses (empty required fields),
# so we see whether thinking is what was buying those groundings.
PROBES = [
    "go to the kitchen, find the cola, and bring it to me",
    "tell me the name of the person at the bed",
    "what's the biggest object on the desk",
    "move forward one meter",
]


def main() -> None:
    load_dotenv()
    from walkie_config import load_config
    load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--thinking", choices=["on", "off"], default="on",
                    help="off -> pass chat_template_kwargs.enable_thinking=False")
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()

    import requests
    from langchain_openai import ChatOpenAI

    base = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    model = os.getenv("LOCAL_MODEL", "qwen3.5-9b")
    key = os.getenv("MODEL_API_KEY") or "none"
    print(f"probe: {model} @ {base}  thinking={args.thinking}  max_tokens={args.max_tokens}")

    # 1) RAW call — exposes usage + whether content carries a <think> block,
    #    which the structured path may hide behind guided decoding.
    extra: dict = {}
    if args.thinking == "off":
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    body = {
        "model": model, "temperature": 0,
        "messages": [
            {"role": "system", "content": "Extract a JSON plan from the command."},
            {"role": "user", "content": PROBES[0]},
        ],
        **({"max_tokens": args.max_tokens} if args.max_tokens else {}),
        **extra,
    }
    t0 = time.time()
    r = requests.post(f"{base.rstrip('/')}/chat/completions",
                      json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
    dt = time.time() - t0
    r.raise_for_status()
    d = r.json()
    content = d["choices"][0]["message"]["content"] or ""
    usage = d.get("usage", {})
    ct = usage.get("completion_tokens")
    print("\n--- RAW chat/completions ---")
    print(f"  latency        {dt:.1f}s")
    print(f"  usage          {usage}")
    print(f"  has <think>    {'<think>' in content}")
    if ct:
        print(f"  tok/s          {ct/dt:.1f}  (completion {ct} tok)")
    print(f"  content[:300]  {content[:300]!r}")
    print("\n  => big tokens + <think>  => reasoning-bound  => fix client-side here")
    print("     small out + low tok/s => throughput-bound => fix on the serving box")

    # 2) REAL structured path — the latency GPSR actually pays per command.
    kw = {}
    if args.max_tokens:
        kw["max_tokens"] = args.max_tokens
    if extra:
        kw["extra_body"] = extra
    chat = ChatOpenAI(base_url=base, api_key=key, model=model, temperature=0, **kw)
    from tasks.GPSR.parse import parse_command
    from tasks.GPSR.world import load_world
    world = load_world(include_absent=True)
    print("\n--- REAL parse_command path (structured) ---")
    for cmd in PROBES:
        t0 = time.time()
        plan = parse_command(chat, cmd, world)
        print(f"  {time.time()-t0:5.1f}s  complete={plan.is_complete}  {cmd!r}")


if __name__ == "__main__":
    main()
