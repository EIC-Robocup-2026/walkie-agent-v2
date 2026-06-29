"""Subprocess lifecycle for challenge runs — start, stream, stop, clean up.

A single :data:`manager` (``ProcessManager``) is shared across every browser tab.
It owns one :class:`ChallengeRuntime` per challenge: the subprocess handle, a
bounded output buffer (each line tagged with a monotonic ``seq`` so a tab can
replay-then-tail without losing its place), and the run state.

Decoupling by design: a challenge is launched as a plain OS subprocess
(``<repo>/.venv/bin/python -m tasks.<NAME>.run``), NOT imported. The web app never
touches the agent code, so ``main`` can churn underneath it. See ``registry.py``
for the launcher resolution (direct venv python → lock-safe; ``uv run --no-sync``
fallback) and why we avoid bare ``uv run`` (it re-resolves and dirties uv.lock).

Stop sends SIGINT first (not SIGTERM): SIGINT raises KeyboardInterrupt in the
task's main thread, so its ``finally:`` runs — releasing nav/arm/camera over zenoh
and writing the final scorecard, exactly like a manual Ctrl+C on ``./run.sh``.
Only if it ignores SIGINT do we escalate to SIGTERM then SIGKILL. The process is
its own session leader (``start_new_session=True``), so each signal goes to the
whole tree (python + its threads/children) via ``killpg``.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

MAX_BUFFER_LINES = 5000

# Stop escalation: give the task this long at each level before going harder.
SIGINT_GRACE_SEC = 10.0   # run finally: release robot + write scorecard
SIGTERM_GRACE_SEC = 4.0


class RunState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    EXITED = "exited"      # clean exit (returncode 0)
    FAILED = "failed"      # non-zero exit
    STOPPED = "stopped"    # we signalled it


@dataclass
class OutputLine:
    seq: int
    text: str
    stream: str  # "stdout" | "stderr" | "system"

    def render(self) -> str:
        if self.stream == "system":
            return f"» {self.text}"
        if self.stream == "stderr":
            return f"⚠ {self.text}"
        return self.text


@dataclass
class ChallengeRuntime:
    name: str
    state: RunState = RunState.IDLE
    process: asyncio.subprocess.Process | None = None
    exit_code: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_cmd: str = ""
    _seq: int = 0
    output: deque[OutputLine] = field(
        default_factory=lambda: deque(maxlen=MAX_BUFFER_LINES)
    )

    def append(self, text: str, stream: str) -> None:
        self._seq += 1
        self.output.append(OutputLine(self._seq, text.rstrip("\n"), stream))


class ProcessManager:
    def __init__(self) -> None:
        self._runtimes: dict[str, ChallengeRuntime] = {}

    def runtime(self, name: str) -> ChallengeRuntime:
        rt = self._runtimes.get(name)
        if rt is None:
            rt = ChallengeRuntime(name=name)
            self._runtimes[name] = rt
        return rt

    def is_running(self, name: str) -> bool:
        rt = self._runtimes.get(name)
        return bool(rt and rt.state == RunState.RUNNING)

    def running_names(self) -> list[str]:
        return [n for n, rt in self._runtimes.items() if rt.state == RunState.RUNNING]

    # ------- lifecycle -------

    async def start(self, argv: list[str], *, name: str, cwd: str,
                    env_overrides: dict[str, str]) -> None:
        """Launch ``argv`` for challenge *name*. No-op if already running."""
        rt = self.runtime(name)
        if rt.state == RunState.RUNNING:
            return

        rt.exit_code = None
        rt.finished_at = None
        rt.started_at = datetime.utcnow()
        rt.output.clear()
        rt.last_cmd = " ".join(argv)

        env = {**os.environ, **env_overrides}
        rt.append(f"$ {rt.last_cmd}", "system")
        rt.append(f"(cwd={cwd})", "system")
        for k, v in env_overrides.items():
            rt.append(f"  {k}={v}", "system")

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,   # own process group → killpg signals the tree
            )
        except (OSError, ValueError) as exc:
            rt.append(f"failed to launch: {exc}", "system")
            rt.state = RunState.FAILED
            return

        rt.process = proc
        rt.state = RunState.RUNNING
        asyncio.create_task(self._supervise(rt, proc))

    async def _supervise(self, rt: ChallengeRuntime,
                         proc: asyncio.subprocess.Process) -> None:
        await asyncio.gather(
            self._pump(proc.stdout, rt, "stdout"),
            self._pump(proc.stderr, rt, "stderr"),
        )
        code = await proc.wait()
        rt.exit_code = code
        rt.finished_at = datetime.utcnow()
        if rt.state != RunState.STOPPED:
            rt.state = RunState.EXITED if code == 0 else RunState.FAILED
        rt.append(f"[exited with code {code}]", "system")
        rt.process = None

    @staticmethod
    async def _pump(stream: asyncio.StreamReader | None,
                    rt: ChallengeRuntime, kind: str) -> None:
        if stream is None:
            return
        while True:
            raw = await stream.readline()
            if not raw:
                break
            rt.append(raw.decode(errors="replace"), kind)

    async def send_stdin(self, name: str, text: str) -> bool:
        """Write ``text`` + newline to the running task's stdin (typed-prompt mode)."""
        rt = self._runtimes.get(name)
        if not rt or not rt.process or rt.process.stdin is None:
            return False
        try:
            rt.process.stdin.write((text + "\n").encode())
            await rt.process.stdin.drain()
            rt.append(f"< {text}", "system")
            return True
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            return False

    async def stop(self, name: str) -> None:
        """Graceful-first stop: SIGINT → SIGTERM → SIGKILL on the process group."""
        rt = self._runtimes.get(name)
        if not rt or not rt.process or rt.state != RunState.RUNNING:
            return
        proc = rt.process
        rt.state = RunState.STOPPED
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return

        for sig, grace, note in (
            (signal.SIGINT, SIGINT_GRACE_SEC, "SIGINT (graceful — running cleanup)"),
            (signal.SIGTERM, SIGTERM_GRACE_SEC, "SIGTERM"),
            (signal.SIGKILL, 0.0, "SIGKILL"),
        ):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            rt.append(f"sent {note}", "system")
            if grace <= 0:
                return
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace)
                return
            except asyncio.TimeoutError:
                continue

    async def shutdown(self) -> None:
        """Stop every running challenge — called on web-app shutdown."""
        await asyncio.gather(
            *(self.stop(n) for n in self.running_names()),
            return_exceptions=True,
        )


manager = ProcessManager()
