"""Compare LLM backends on the work Walkie actually asks of a model.

The question this answers: *can the locally-served model (vLLM Qwen on
``LOCAL_BASE_URL``) drive the agent stack as well as the OpenRouter provider
baseline (claude-sonnet-4.5)?* It runs the SAME battery against every selected
model and prints a side-by-side scorecard, so a regression is a number, not a
vibe.

Three dimensions, cheapest-first:

1. **Capability probe** — does the backend even support the two mechanisms the
   agents depend on: ``with_structured_output`` (the GPSR parser) and
   ``bind_tools`` tool-calling (every agent)? A model that fails here scores 0
   downstream for *plumbing* reasons — this separates that from capability.
2. **GPSR parser coverage** — the quantitative spine. Feeds the GPSR command
   corpus (``tests/gpsr_command_corpus.txt``) through the real
   ``tasks.GPSR.parse.parse_command`` and measures the fraction that ground to a
   COMPLETE typed plan with a clean spoken render. This mirrors the Phase-0
   acceptance gate (``tests/test_gpsr_coverage.py``) but across models.
3. **Agent delegation smoke** — runs a handful of representative instructions
   through the full four-agent stub stack (all hardware tools short-circuited,
   LLM is the only real dependency) and checks the orchestrator routes each to
   the expected sub-agent / tool. Breadth, not depth.

Everything runs offline-against-the-LLM: no robot, no walkie-ai-server. The
local side needs ``localhost:8000`` reachable; the provider side needs
``OPENROUTER_API_KEY``. Hold temperature (0) and corpus fixed across models so a
score gap is the model.

    # both defaults (local qwen vs claude-sonnet-4.5), full corpus:
    uv run python -m manual_tests.eval_llm_compare

    # quick smoke (5 commands, skip agent stack):
    uv run python -m manual_tests.eval_llm_compare --limit 5 --no-agent

    # pick models / write a markdown report:
    uv run python -m manual_tests.eval_llm_compare --models local,sonnet --out llm_eval.md
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# --- model registry ---------------------------------------------------------
#
# Each entry is a self-contained ChatOpenAI spec. `local` points at the on-box
# vLLM (the candidate); `sonnet`/`gemini` are OpenRouter baselines. Add more by
# copying a row. `max_tokens=None` lets the server use its default budget — the
# local reasoning model needs room for its <think> trace before the answer, so
# we never cap it tight.

@dataclass
class ModelSpec:
    label: str
    base_url: str
    model: str
    api_key_env: str  # env var holding the key
    max_tokens: int | None = None


MODELS: dict[str, ModelSpec] = {
    "local": ModelSpec(
        label="local-qwen3.5-9b",
        base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1"),
        model=os.getenv("LOCAL_MODEL", "qwen3.5-9b"),
        api_key_env="MODEL_API_KEY",
    ),
    "sonnet": ModelSpec(
        label="claude-sonnet-4.5",
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model="anthropic/claude-sonnet-4.5",
        api_key_env="OPENROUTER_API_KEY",
    ),
    "gemini": ModelSpec(
        label="gemini-3-flash",
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model="google/gemini-3-flash-preview:nitro",
        api_key_env="OPENROUTER_API_KEY",
    ),
}


def build_chat(spec: ModelSpec, no_think: bool = False, max_tokens: int | None = None):
    from langchain_openai import ChatOpenAI

    kw = {}
    cap = max_tokens or spec.max_tokens
    if cap:
        kw["max_tokens"] = cap
    if no_think:
        # vLLM/Qwen: suppress the <think> trace for this structured-extraction task.
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return ChatOpenAI(
        base_url=spec.base_url,
        api_key=os.getenv(spec.api_key_env) or "none",
        model=spec.model,
        temperature=0,
        **kw,
    )


# --- 1. capability probe ----------------------------------------------------

@dataclass
class CapabilityResult:
    structured_output: bool = False
    tool_calling: bool = False
    detail: str = ""


def probe_capability(chat) -> CapabilityResult:
    """Confirm the backend supports structured output + tool-calling. A failure
    here means downstream zeros are plumbing, not the model."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    res = CapabilityResult()

    class _Person(BaseModel):
        name: str = Field(description="the person's name")
        age: int = Field(description="age in years")

    try:
        out = chat.with_structured_output(_Person, method="json_schema").invoke(
            [SystemMessage(content="Extract the person."),
             HumanMessage(content="Alice is 30 years old.")]
        )
        res.structured_output = out.name.lower() == "alice" and out.age == 30
    except Exception as exc:  # noqa: BLE001
        res.detail += f"structured_output: {type(exc).__name__}: {exc}; "

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    try:
        out = chat.bind_tools([get_weather]).invoke(
            [HumanMessage(content="Use the tool to get the weather in Bangkok.")]
        )
        calls = getattr(out, "tool_calls", []) or []
        res.tool_calling = any(c["name"] == "get_weather" for c in calls)
    except Exception as exc:  # noqa: BLE001
        res.detail += f"tool_calling: {type(exc).__name__}: {exc}; "

    return res


# --- 2. GPSR parser coverage ------------------------------------------------

def load_corpus(limit: int | None = None) -> list[str]:
    """The GPSR command corpus, stripped of comments and "<category>: " prefixes
    (same loader shape as tests/test_gpsr_coverage.py)."""
    path = Path(__file__).resolve().parents[1] / "tests" / "gpsr_command_corpus.txt"
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head = line.split(":", 1)[0]
        if ":" in line and head.replace("_", "").isalpha() and " " not in head:
            line = line.split(":", 1)[1].strip()
        out.append(line)
    return out[:limit] if limit else out


@dataclass
class CoverageResult:
    total: int = 0
    complete: int = 0
    render_defects: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def pct(self) -> float:
        return self.complete / self.total if self.total else 0.0


def run_coverage(chat, corpus: list[str]) -> CoverageResult:
    from tasks.GPSR.parse import parse_command
    from tasks.GPSR.plan import render_plan_speech
    from walkie_world.map.vocab import load_world

    world = load_world(include_absent=True)
    res = CoverageResult(total=len(corpus))
    t0 = time.time()
    for cmd in corpus:
        plan = parse_command(chat, cmd, world)
        if plan.is_complete:
            res.complete += 1
            speech = render_plan_speech(plan)
            if not speech or "could not work out a plan" in speech or "_" in speech:
                res.render_defects.append(f"{cmd!r} -> {speech!r}")
        else:
            gaps = [u for s in plan.steps for u in s.unresolved] or ["no steps"]
            steps = [s.primitive.value for s in plan.steps]
            res.failures.append(f"{cmd!r} steps={steps} gaps={gaps}")
    res.seconds = time.time() - t0
    return res


# --- 3. agent delegation smoke ----------------------------------------------

# Each case: an instruction + the tool the main orchestrator SHOULD reach for.
# Scored loosely — we only check the expected tool appears among the main
# agent's tool calls (routing correctness), not the sub-agent's internal steps.
AGENT_CASES: list[tuple[str, str]] = [
    ("Move forward one meter.", "delegate_to_actuator"),
    ("Go to the kitchen.", "delegate_to_actuator"),
    ("What do you see in front of you right now?", "delegate_to_vision"),
    ("Where did you last see the cola?", "delegate_to_database"),
    ("Say hello to everyone.", "speak"),
]


@dataclass
class AgentResult:
    total: int = 0
    correct: int = 0
    rows: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def pct(self) -> float:
        return self.correct / self.total if self.total else 0.0


def build_stub_stack(chat):
    """The full four-agent stack with hardware tools short-circuited."""
    import types

    os.environ["WALKIE_STUB_TOOLS"] = "1"
    os.environ["WALKIE_GRAPHS_ENABLED"] = "0"
    os.environ["DISABLE_LISTENING"] = "1"

    from agents.actuator_agent import create_actuator_agent
    from agents.core.robot_context import RobotContext
    from agents.database_agent import create_database_agent
    from agents.vision_agent import create_vision_agent
    from agents.walkie_agent import create_walkie_main_agent

    try:
        RobotContext.init(perception_path=os.getenv("PERCEPTION_PATH", "perception.json"))
    except Exception:  # noqa: BLE001 — already initialised across models
        pass
    RobotContext.get().stage = "ready"

    walkie = types.SimpleNamespace(
        nav=None, status=None, arm=None, tools=None,
        camera=None, speaker=None, microphone=None, close=lambda: None,
    )
    walkieAI = types.SimpleNamespace()
    actuator = create_actuator_agent(chat, walkieAI, walkie)
    vision = create_vision_agent(chat, walkieAI, walkie)
    database = create_database_agent(chat, walkieAI, walkie, graphs=None)
    return create_walkie_main_agent(chat, walkieAI, walkie, actuator, vision, database)


def run_agent_smoke(chat) -> AgentResult:
    from langchain_core.messages import HumanMessage

    agent = build_stub_stack(chat)
    res = AgentResult(total=len(AGENT_CASES))
    t0 = time.time()
    for i, (instruction, expected) in enumerate(AGENT_CASES):
        called: list[str] = []
        try:
            out = agent.invoke(
                {"messages": [HumanMessage(content=instruction)]},
                config={"configurable": {"thread_id": f"eval-{i}"}},
            )
            for m in out["messages"]:
                for c in getattr(m, "tool_calls", []) or []:
                    called.append(c["name"])
        except Exception as exc:  # noqa: BLE001
            res.rows.append(f"✗ {instruction!r} -> ERROR {type(exc).__name__}: {exc}")
            continue
        ok = expected in called
        res.correct += int(ok)
        mark = "✓" if ok else "✗"
        res.rows.append(f"{mark} {instruction!r} expect={expected} got={called or '[]'}")
    res.seconds = time.time() - t0
    return res


# --- orchestration + reporting ----------------------------------------------

@dataclass
class ModelReport:
    spec: ModelSpec
    cap: CapabilityResult
    coverage: CoverageResult | None = None
    agent: AgentResult | None = None


def evaluate(spec: ModelSpec, corpus: list[str], do_agent: bool,
             no_think: bool = False, max_tokens: int | None = None) -> ModelReport:
    flags = (" [thinking off]" if no_think else "") + (f" [max_tokens={max_tokens}]" if max_tokens else "")
    print(f"\n{'='*72}\n  {spec.label}{flags}  ({spec.model} @ {spec.base_url})\n{'='*72}")
    chat = build_chat(spec, no_think=no_think, max_tokens=max_tokens)

    print("[1/3] capability probe ...", flush=True)
    cap = probe_capability(chat)
    print(f"      structured_output={cap.structured_output}  tool_calling={cap.tool_calling}"
          + (f"  ({cap.detail})" if cap.detail else ""))

    rep = ModelReport(spec=spec, cap=cap)

    print(f"[2/3] GPSR coverage over {len(corpus)} commands ...", flush=True)
    rep.coverage = run_coverage(chat, corpus)
    cv = rep.coverage
    print(f"      coverage {cv.complete}/{cv.total} = {cv.pct:.0%}  "
          f"({cv.seconds:.0f}s, {cv.seconds/max(cv.total,1):.1f}s/cmd, "
          f"{len(cv.render_defects)} render defects)")

    if do_agent:
        print(f"[3/3] agent delegation smoke ({len(AGENT_CASES)} cases) ...", flush=True)
        rep.agent = run_agent_smoke(chat)
        print(f"      routing {rep.agent.correct}/{rep.agent.total} = {rep.agent.pct:.0%} "
              f"({rep.agent.seconds:.0f}s)")
    else:
        print("[3/3] agent smoke skipped (--no-agent)")
    return rep


def render_report(reports: list[ModelReport], do_agent: bool) -> str:
    lines: list[str] = []
    w = lines.append
    w("# LLM backend comparison\n")
    w("Same corpus, temperature=0, run offline against each backend.\n")

    # scorecard
    w("## Scorecard\n")
    header = "| metric | " + " | ".join(r.spec.label for r in reports) + " |"
    w(header)
    w("|" + "---|" * (len(reports) + 1))
    def row(name, fn):
        w(f"| {name} | " + " | ".join(fn(r) for r in reports) + " |")
    row("structured output", lambda r: "✅" if r.cap.structured_output else "❌")
    row("tool calling", lambda r: "✅" if r.cap.tool_calling else "❌")
    row("GPSR coverage", lambda r: f"**{r.coverage.pct:.0%}** ({r.coverage.complete}/{r.coverage.total})" if r.coverage else "—")
    row("render defects", lambda r: str(len(r.coverage.render_defects)) if r.coverage else "—")
    row("parse latency/cmd", lambda r: f"{r.coverage.seconds/max(r.coverage.total,1):.1f}s" if r.coverage else "—")
    if do_agent:
        row("agent routing", lambda r: f"{r.agent.pct:.0%} ({r.agent.correct}/{r.agent.total})" if r.agent else "—")
    w("")

    # per-model detail
    for r in reports:
        w(f"## {r.spec.label} — detail\n")
        if r.coverage and r.coverage.failures:
            w("**GPSR parse misses:**\n")
            for f in r.coverage.failures:
                w(f"- {f}")
            w("")
        if r.coverage and r.coverage.render_defects:
            w("**Render defects:**\n")
            for d in r.coverage.render_defects:
                w(f"- {d}")
            w("")
        if r.agent:
            w("**Agent routing:**\n")
            for row_ in r.agent.rows:
                w(f"- {row_}")
            w("")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    from walkie_config import load_config
    load_config()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="local,sonnet",
                    help=f"comma-separated keys from {sorted(MODELS)} (default local,sonnet)")
    ap.add_argument("--limit", type=int, default=None, help="cap corpus to first N commands")
    ap.add_argument("--no-agent", action="store_true", help="skip the agent delegation smoke")
    ap.add_argument("--no-think", action="store_true",
                    help="suppress the <think> trace (Qwen enable_thinking=False) for all models")
    ap.add_argument("--max-tokens", type=int, default=None, help="cap completion tokens for all models")
    ap.add_argument("--out", default=None, help="write a markdown report to this path")
    args = ap.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    bad = [k for k in keys if k not in MODELS]
    if bad:
        ap.error(f"unknown model keys {bad}; choose from {sorted(MODELS)}")

    corpus = load_corpus(args.limit)
    print(f"corpus: {len(corpus)} commands | models: {keys} | agent smoke: {not args.no_agent}")

    reports = [evaluate(MODELS[k], corpus, do_agent=not args.no_agent,
                        no_think=args.no_think, max_tokens=args.max_tokens) for k in keys]

    md = render_report(reports, do_agent=not args.no_agent)
    print("\n" + "=" * 72)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"\n[report written to {args.out}]")


if __name__ == "__main__":
    main()
