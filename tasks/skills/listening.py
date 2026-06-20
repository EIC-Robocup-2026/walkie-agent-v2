"""Background, prompt-free voice-command listener.

The generic two-thread mic loop split out of the Receptionist task: it keeps
the microphone re-arming while STT and any per-utterance work run off the hot
path, so the robot never goes deaf during a nav step or a model round-trip.
What to *do* with each transcript (classify it, decide whether to stop) is the
caller's business and lives in the *on_transcript* callback.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable

from tasks.base import TaskContext


class CommandListener:
    """Background mic loop that turns each utterance into a callback, never going deaf.

    While the robot drives (or otherwise keeps busy) it must still hear a spoken
    command. Doing record -> transcribe -> handle serially on the caller's thread
    would leave the microphone dark during every busy step AND during every STT +
    handler round-trip — precisely when the user is most likely to speak. So this
    splits the work across two daemon threads:

    * a *recorder* that loops ``record_until_silence`` and re-arms IMMEDIATELY,
      pushing each captured clip onto a queue without waiting for STT or the
      handler (so the mic is re-listening within milliseconds of a phrase ending);
    * a *worker* that drains the queue, transcribes each clip and hands the text
      to *on_transcript*.

    *on_transcript* is called with each (non-empty) transcript; return a truthy
    value to set :attr:`triggered`, the generic stop signal a follow loop polls
    (see :func:`tasks.skills.navigation.follow_person`). Only this listener may
    touch the microphone while it runs — one ``sd.InputStream`` can be open at a
    time, so the caller must not call ``ctx.listen`` concurrently. Best-effort:
    every mic / STT / handler failure is swallowed so a glitch never crashes the
    step. Use as a context manager::

        with CommandListener(ctx, on_transcript=handle) as listener:
            while ...:
                ... do work ...
                if listener.triggered.is_set():
                    break
    """

    def __init__(
        self,
        ctx: TaskContext,
        on_transcript: Callable[[str], bool | None],
        *,
        record_timeout: float = 30.0,
    ) -> None:
        self.ctx = ctx
        self.on_transcript = on_transcript
        self.triggered = threading.Event()  # set once a handler returns truthy
        self.last_text = ""                  # most recent transcript (debug/inspection)
        self.record_timeout = record_timeout
        self._stop = threading.Event()
        self._queue: "queue.Queue[bytes | None]" = queue.Queue()
        self._threads: list[threading.Thread] = []

    def start(self) -> "CommandListener":
        if self._threads:
            return self
        self._stop.clear()
        self.triggered.clear()
        self._threads = [
            threading.Thread(target=self._record_loop, daemon=True),
            threading.Thread(target=self._work_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)  # wake the worker so it can see the stop flag
        for t in self._threads:
            # The recorder may be mid-capture (blocked up to record_timeout);
            # give it that long plus a margin to wind down its InputStream.
            t.join(timeout=max(2.0, self.record_timeout + 1.0))
        self._threads = []

    def __enter__(self) -> "CommandListener":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _record_loop(self) -> None:
        # Dev/CI path (DISABLE_LISTENING): no mic — read typed lines and feed
        # them straight to the handler (no latency to hide, so no queue).
        if self.ctx.disable_listening:
            while not self._stop.is_set():
                try:
                    line = input("[listen] > ")
                except EOFError:
                    return
                self._handle_text(line)
            return
        while not self._stop.is_set():
            try:
                audio = self.ctx.walkie.microphone.record_until_silence(
                    timeout=self.record_timeout
                )
                
            except Exception as exc:
                print(f"[skills] CommandListener record failed ({exc})")
                self._stop.wait(0.2)
                continue
            if audio and not self._stop.is_set():
                self._queue.put(audio)  # hand off; re-arm the mic right away

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            audio = self._queue.get()
            if audio is None:  # stop sentinel
                return
            try:
                text = self.ctx.walkieAI.stt.transcribe(audio)
            except Exception as exc:
                print(f"[skills] CommandListener STT failed ({exc})")
                continue
            self._handle_text(text)

    def _handle_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        print(f"[heard] {text}")
        self.last_text = text
        try:
            if self.on_transcript(text):
                self.triggered.set()
        except Exception as exc:
            print(f"[skills] CommandListener handler failed ({exc})")
