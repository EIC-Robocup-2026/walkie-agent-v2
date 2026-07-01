"""Inspection task steps: a linear deterministic scaffold.

The referee inspects the robot end-to-end (see the module docstring in skills.py):
notice the entry door open and drive through → stop for a person who steps in front
→ reach the inspection points → declare external devices → prove it is loud enough →
move to the exit on a signal. Every step is best-effort: a failure logs and the run
continues to the next item (Task.run never raises), so one flaky read can't abort the
inspection.

All heavy / robot imports live inside :mod:`tasks.Inspection.skills` (lazily), so
``build_inspection_task`` and this module import fine on a GPU-less box.
"""

from __future__ import annotations

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import skills


class EnterThroughDoor(SubTask):
    """Wait for the entry door, notice it open, and drive inside."""

    def run(self, ctx: TaskContext) -> StepResult:
        skills.enter_through_door(ctx)
        return StepResult.DONE


class SafetyStop(SubTask):
    """Stop and hold while the referee stands in front (safety demonstration)."""

    def run(self, ctx: TaskContext) -> StepResult:
        skills.demonstrate_safety_stop(ctx)
        return StepResult.DONE


class VisitInspectionPoints(SubTask):
    """Drive to each configured inspection point in order."""

    def run(self, ctx: TaskContext) -> StepResult:
        skills.visit_inspection_points(ctx)
        return StepResult.DONE


class DeclareDevices(SubTask):
    """Tell the referee about the external devices in use."""

    def run(self, ctx: TaskContext) -> StepResult:
        skills.declare_devices(ctx)
        return StepResult.DONE


class LoudnessTest(SubTask):
    """Speak a clear test phrase (and optionally confirm audibility)."""

    def run(self, ctx: TaskContext) -> StepResult:
        skills.loudness_test(ctx)
        return StepResult.DONE


class WaitForExitSignal(SubTask):
    """Answer the referee's questions until they signal "go to the exit"."""

    def run(self, ctx: TaskContext) -> StepResult:
        skills.wait_for_exit_signal(ctx)
        return StepResult.DONE


class GoToExit(SubTask):
    """Drive to the exit; the referee then presses the stop button."""

    def run(self, ctx: TaskContext) -> StepResult:
        skills.go_to_exit(ctx)
        return StepResult.DONE


def build_inspection_task(ctx: TaskContext) -> Task:
    """Assemble the Inspection task. Pure: no hardware touched at build time."""
    return Task(
        "Inspection",
        [
            EnterThroughDoor(),
            # SafetyStop(),
            VisitInspectionPoints(),
            # DeclareDevices(),
            # LoudnessTest(),
            WaitForExitSignal(),
            GoToExit(),
        ],
        ctx,
    )
