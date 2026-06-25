"""Score estimation + live scoring for the competition tasks.

One :class:`ScoreSheet` per challenge encodes its rulebook scoresheet **as data** —
the single source of truth for both:

* the **live runtime tally** (:class:`ScoreTracker`: award events during a run ->
  running estimate + atomic JSON snapshot), and
* the **planning worksheet** (:func:`estimate`: per-line capture % -> low/exp/high
  points, the same shape as ``docs/SCORING.md``).

**What the runtime tally means — read this.** :meth:`ScoreTracker.award` fires when
the *robot believes* it did the scoring action (recognized an object, indicated a
placement, ...). A hallucinated detection still increments it. So the tally is an
**optimistic ceiling of *attempted / claimed* points, NOT referee-awarded score** —
the same honesty discipline as "offline control flow verified, not end-to-end".
Read it alongside the capture % for a realistic number.

No hardware imports — safe to import on a GPU-less box (mirrors ``tasks/base.py``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum


class LineKind(str, Enum):
    POSITIVE = "positive"   # points earned for doing the action
    PENALTY = "penalty"     # points lost (points < 0)
    BONUS = "bonus"         # one-time / conditional reward (points > 0, max_units 1)


@dataclass(frozen=True)
class ScoreLine:
    """One line of a rulebook scoresheet.

    ``points`` is PER UNIT (negative for penalties); ``max_units`` caps how many
    times it can score (e.g. 12 objects). ``arm=True`` marks a line that needs the
    robot manipulator — gated today, so excluded from the non-arm budget.
    """

    key: str               # stable id used by award() / capture maps
    label: str             # human label from the scoresheet
    points: int            # points per unit (negative for penalties)
    max_units: int = 1     # cap on scoring units
    arm: bool = False      # needs the robot arm (gated)
    kind: LineKind = LineKind.POSITIVE
    note: str = ""

    @property
    def max_points(self) -> int:
        """Most this line can contribute (points x max_units)."""
        return self.points * self.max_units


@dataclass
class ScoreSheet:
    """A challenge's scoresheet as data. ``rulebook_total`` is the official
    'total scorable' from the PDF — the sum of *positive* lines, excluding
    penalties and special bonuses (verify with :meth:`positive_total`)."""

    challenge: str
    rulebook_total: int
    lines: list[ScoreLine]

    def __post_init__(self) -> None:
        keys = [ln.key for ln in self.lines]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(f"{self.challenge}: duplicate score keys {dupes}")

    def line(self, key: str) -> ScoreLine:
        for ln in self.lines:
            if ln.key == key:
                return ln
        raise KeyError(f"{self.challenge}: no score line {key!r}")

    def positives(self) -> list[ScoreLine]:
        return [ln for ln in self.lines if ln.points > 0]

    def penalties(self) -> list[ScoreLine]:
        return [ln for ln in self.lines if ln.points < 0]

    def non_arm_lines(self) -> list[ScoreLine]:
        """Positive lines achievable WITHOUT the arm (the budget scorable today)."""
        return [ln for ln in self.lines if ln.points > 0 and not ln.arm]

    def arm_lines(self) -> list[ScoreLine]:
        """Positive lines gated on the arm skill."""
        return [ln for ln in self.lines if ln.points > 0 and ln.arm]

    def positive_total(self) -> int:
        """Sum of every positive line at full units — should equal rulebook_total."""
        return sum(ln.max_points for ln in self.positives())

    def non_arm_ceiling(self) -> int:
        """Most we can score with the arm gated off (perfect non-arm run)."""
        return sum(ln.max_points for ln in self.non_arm_lines())


@dataclass
class ScoreTracker:
    """Live tally of *attempted / claimed* points during a run (see module docstring).

    Award scoring events as the task believes it completes them; read :meth:`earned`
    / :meth:`summary` for a running claimed total, or :meth:`write` a JSON snapshot
    the team can watch during a practice run.
    """

    sheet: ScoreSheet
    path: str | None = None
    units: dict[str, int] = field(default_factory=dict)

    def award(self, key: str, n: int = 1) -> int:
        """Record *n* units of line *key* (clamped to its max_units).

        Returns the points added this call (>=0 for positives, <=0 for penalties).
        An unknown key raises — a typo should fail loudly, not silently score 0.
        """
        line = self.sheet.line(key)
        have = self.units.get(key, 0)
        added = max(0, min(n, line.max_units - have))
        self.units[key] = have + added
        return added * line.points

    def penalize(self, key: str, n: int = 1) -> int:
        """Record a penalty line (alias of :meth:`award` for readability)."""
        return self.award(key, n)

    def units_of(self, key: str) -> int:
        return self.units.get(key, 0)

    def earned(self) -> int:
        """Net claimed points (positives + penalties)."""
        return sum(self.sheet.line(k).points * n for k, n in self.units.items())

    def earned_positive(self) -> int:
        return sum(self.sheet.line(k).points * n for k, n in self.units.items()
                   if self.sheet.line(k).points > 0)

    def penalties(self) -> int:
        return sum(self.sheet.line(k).points * n for k, n in self.units.items()
                   if self.sheet.line(k).points < 0)

    def breakdown(self) -> list[dict]:
        """One row per line that scored (in sheet order)."""
        rows = []
        for line in self.sheet.lines:
            n = self.units.get(line.key, 0)
            if n:
                rows.append({"key": line.key, "label": line.label, "units": n,
                             "points": n * line.points, "arm": line.arm})
        return rows

    def summary(self) -> str:
        out = [f"[scorecard] {self.sheet.challenge} — ATTEMPTED/claimed points "
               "(NOT referee-awarded; a wrong detection still counts here)"]
        for r in self.breakdown():
            tag = " (arm)" if r["arm"] else ""
            out.append(f"  {r['units']:>2} x {r['label']}{tag}: {r['points']:+d}")
        out.append(f"  = {self.earned():+d} claimed  (positives {self.earned_positive():+d}, "
                   f"penalties {self.penalties():+d}; non-arm ceiling "
                   f"{self.sheet.non_arm_ceiling()}, rulebook total {self.sheet.rulebook_total})")
        return "\n".join(out)

    def snapshot(self) -> dict:
        return {
            "challenge": self.sheet.challenge,
            "disclaimer": ("attempted/claimed points — NOT referee-awarded; "
                           "a wrong detection still increments"),
            "claimed": self.earned(),
            "claimed_positive": self.earned_positive(),
            "penalties": self.penalties(),
            "non_arm_ceiling": self.sheet.non_arm_ceiling(),
            "rulebook_total": self.sheet.rulebook_total,
            "breakdown": self.breakdown(),
        }

    def write(self, path: str | None = None) -> None:
        """Atomically write the JSON snapshot (tmp + os.replace, like perception.json)."""
        target = path or self.path
        if not target:
            return
        tmp = f"{target}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.snapshot(), f, indent=2)
        os.replace(tmp, target)


@dataclass(frozen=True)
class Capture:
    """Per-line capture %: expected fraction of a line's points we actually earn,
    under partial scoring + current validation. low = bad luck / unvalidated,
    exp = realistic, high = clean run (matches docs/SCORING.md columns)."""

    low: float
    exp: float
    high: float


def estimate(sheet: ScoreSheet, captures: dict[str, Capture], *,
             include_arm: bool = False) -> dict:
    """Expected points per line given capture % — reproduces the SCORING.md tables.

    Multiplies each positive line's max_points by its low/exp/high capture fraction.
    A line with no capture entry contributes 0 (unmodelled). ``include_arm=False``
    (default) gives the non-arm estimate — the budget scorable with the arm gated.
    """
    rows: list[dict] = []
    total = {"low": 0.0, "exp": 0.0, "high": 0.0}
    for line in sheet.positives():
        if line.arm and not include_arm:
            continue
        cap = captures.get(line.key) or Capture(0.0, 0.0, 0.0)
        row = {"key": line.key, "label": line.label, "max_points": line.max_points,
               "low": line.max_points * cap.low,
               "exp": line.max_points * cap.exp,
               "high": line.max_points * cap.high}
        for k in total:
            total[k] += row[k]
        rows.append(row)
    return {"challenge": sheet.challenge, "include_arm": include_arm,
            "rows": rows, "total": total}
