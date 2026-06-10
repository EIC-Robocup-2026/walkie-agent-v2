from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Stage = Literal["explore", "ready"]


@dataclass
class SpeechEntry:
    agent: str
    text: str
    ts: float


class RobotContext:
    """Process-wide singleton holding cross-agent state.

    - speech log: what each agent has said recently (so all agents can see it)
    - stage: 'explore' or 'ready' — used by perception middleware to no-op early
    - perception_path: where the walkie_graphs perception loop writes the latest snapshot
    """

    _instance: "RobotContext | None" = None
    _lock = threading.Lock()

    def __init__(self, perception_path: Path, max_speech: int = 20) -> None:
        self.perception_path = Path(perception_path)
        self.speech_log: deque[SpeechEntry] = deque(maxlen=max_speech)
        self.stage: Stage = "explore"
        self._speech_lock = threading.Lock()

    @classmethod
    def init(cls, perception_path: str | Path = "perception.json") -> "RobotContext":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(perception_path=Path(perception_path))
            return cls._instance

    @classmethod
    def get(cls) -> "RobotContext":
        if cls._instance is None:
            raise RuntimeError("RobotContext not initialized. Call RobotContext.init() first.")
        return cls._instance

    def add_speech(self, agent: str, text: str) -> None:
        with self._speech_lock:
            self.speech_log.append(SpeechEntry(agent=agent, text=text, ts=time.time()))

    def recent_speech_text(self, max_age_sec: float = 60.0) -> str:
        now = time.time()
        with self._speech_lock:
            entries = [e for e in self.speech_log if now - e.ts <= max_age_sec]
        if not entries:
            return ""
        lines = []
        for e in entries:
            ago = max(0, int(now - e.ts))
            lines.append(f"[{e.agent} {ago}s ago]: {e.text!r}")
        return "\n".join(lines)

    def perception_snapshot(self) -> dict | None:
        if not self.perception_path.exists():
            return None
        try:
            with self.perception_path.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
