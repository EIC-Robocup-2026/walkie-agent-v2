"""The web UI — one page, a card per challenge.

Each card: Run/Stop, a status pill, the DISABLE_LISTENING toggle, a ``*_SLICE``
selector (Restaurant / Pick&Place), an extra-env field, a live log with a stdin
box (type GPSR commands here in DISABLE_LISTENING mode), and the last-run
scorecard. State is driven by polling the shared ``manager`` on a per-client
``ui.timer`` — NiceGUI cancels those automatically when the tab closes, so there's
no subscribe/unsubscribe bookkeeping and no cross-client context juggling.
"""

from __future__ import annotations

from nicegui import ui

from .process_manager import RunState, manager
from .registry import (
    DATA_DIR,
    REPO_ROOT,
    Challenge,
    discover,
    python_launcher,
    read_scorecard,
)

_PILL: dict[RunState, tuple[str, str]] = {
    RunState.IDLE: ("idle", "grey"),
    RunState.RUNNING: ("running", "green"),
    RunState.EXITED: ("exited 0", "blue"),
    RunState.FAILED: ("failed", "red"),
    RunState.STOPPED: ("stopped", "orange"),
}

_SCORE_COLS = [
    {"name": "label", "label": "Line", "field": "label", "align": "left"},
    {"name": "units", "label": "×", "field": "units", "align": "right"},
    {"name": "points", "label": "pts", "field": "points", "align": "right"},
]


def _parse_extra(text: str) -> dict[str, str]:
    """``"A=1, B=2"`` → ``{"A": "1", "B": "2"}`` (ignores malformed pieces)."""
    out: dict[str, str] = {}
    for piece in (text or "").split(","):
        piece = piece.strip()
        if "=" in piece:
            k, v = piece.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out


def _build_card(ch: Challenge) -> None:
    with ui.card().classes("w-full"):
        # --- header ---
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            with ui.column().classes("gap-0"):
                ui.label(ch.title).classes("text-lg font-bold")
                meta = f"§{ch.rulebook} · " if ch.rulebook else ""
                ui.label(f"{meta}{ch.module}").classes("text-xs text-grey font-mono")
            pill = ui.badge("idle").props("color=grey")
        ui.label(ch.description).classes("text-sm text-grey-7")

        # --- controls ---
        with ui.row().classes("items-center gap-3 w-full"):
            run_btn = ui.button("Run", icon="play_arrow")
            stop_btn = ui.button("Stop", icon="stop").props("color=red outline")
            stop_btn.set_enabled(False)
            listen = ui.switch("typed input (DISABLE_LISTENING)", value=True)
            slice_sel = (
                ui.select(list(ch.slice.choices), value=ch.slice.default,
                          label=ch.slice.env).classes("w-44").props("dense outlined")
                if ch.slice else None
            )
        extra = ui.input("extra env — KEY=VALUE, comma-separated") \
            .classes("w-full").props("dense")

        # --- live log + stdin ---
        with ui.expansion("Logs", icon="terminal", value=True).classes("w-full"):
            log = ui.log(max_lines=4000).classes(
                "w-full h-64 bg-black text-green-3 text-xs"
            ).style("font-family: monospace")
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                stdin_in = ui.input(
                    placeholder="send a line to stdin (e.g. a GPSR command)…"
                ).classes("grow").props("dense")

                async def _send() -> None:
                    if stdin_in.value:
                        ok = await manager.send_stdin(ch.name, stdin_in.value)
                        if ok:
                            stdin_in.value = ""
                        else:
                            ui.notify("not running / stdin closed", type="warning")

                stdin_in.on("keydown.enter", _send)
                ui.button("Send", on_click=_send).props("flat dense")

        # --- scorecard ---
        with ui.expansion("Scorecard — last completed run", icon="scoreboard") \
                .classes("w-full"):
            @ui.refreshable
            def score_panel() -> None:
                data = read_scorecard(ch)
                if not data:
                    ui.label("No scorecard yet — tasks write it when a run ends or "
                             "is stopped gracefully.").classes("text-xs text-grey")
                    return
                claimed = data.get("claimed", 0)
                ceiling = data.get("non_arm_ceiling", "—")
                total = data.get("rulebook_total", "—")
                pen = data.get("penalties", 0)
                ui.label(
                    f"claimed {claimed}  ·  non-arm ceiling {ceiling}  ·  "
                    f"rulebook {total}" + (f"  ·  penalties {pen}" if pen else "")
                ).classes("text-sm font-bold")
                rows = [r for r in data.get("breakdown", []) if r.get("units")]
                if rows:
                    ui.table(columns=_SCORE_COLS, rows=rows, row_key="key") \
                        .classes("w-full").props("dense flat")
                else:
                    ui.label("ran, but no lines scored yet").classes("text-xs text-grey")

            score_panel()

    # --- handlers ---
    async def _run() -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        overrides = ch.env_overrides(
            disable_listening=listen.value,
            slice_value=(slice_sel.value if slice_sel else None),
            extra=_parse_extra(extra.value),
        )
        log.clear()
        await manager.start(ch.argv(), name=ch.name, cwd=str(REPO_ROOT),
                            env_overrides=overrides)

    async def _stop() -> None:
        await manager.stop(ch.name)

    run_btn.on_click(_run)
    stop_btn.on_click(_stop)

    # --- live polling (auto-cancelled on tab close) ---
    cursor = {"seen": 0, "state": None}

    def _tick() -> None:
        rt = manager.runtime(ch.name)
        for line in list(rt.output):
            if line.seq > cursor["seen"]:
                log.push(line.render())
                cursor["seen"] = line.seq
        if rt.state != cursor["state"]:
            cursor["state"] = rt.state
            label, color = _PILL[rt.state]
            pill.set_text(label)
            pill.props(f"color={color}")
            running = rt.state == RunState.RUNNING
            run_btn.set_enabled(not running)
            stop_btn.set_enabled(running)

    ui.timer(0.3, _tick)
    ui.timer(2.0, score_panel.refresh)


def create_page() -> None:
    @ui.page("/")
    def index() -> None:
        ui.add_head_html("<style>.nicegui-log{white-space:pre-wrap}</style>")
        with ui.column().classes("w-full max-w-4xl mx-auto gap-4 p-4"):
            ui.label("🤖 Walkie Task Commander").classes("text-2xl font-bold")
            launcher = " ".join(python_launcher())
            ui.markdown(
                f"Launch & monitor RoboCup@Home challenges from "
                f"`{REPO_ROOT}`.\n\n"
                f"- **Stop** sends SIGINT first (graceful — the task runs its "
                f"cleanup and writes its scorecard), then escalates.\n"
                f"- **Scorecard** is the *last completed* run, not a live tally "
                f"(tasks only persist it at the end).\n"
                f"- **DISABLE_LISTENING** reads typed prompts from stdin → use the "
                f"log's input box; turn it off to use the robot mic.\n"
                f"- launcher: `{launcher}`  ·  scorecards → `{DATA_DIR.name}/`"
            ).classes("text-sm text-grey-7")

            challenges = discover()
            if not challenges:
                ui.label(f"No tasks/*/run.py found under {REPO_ROOT}") \
                    .classes("text-red")
                return
            for ch in challenges:
                _build_card(ch)
