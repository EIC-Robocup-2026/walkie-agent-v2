#!/usr/bin/env bash
# Shared launcher invoked by each tasks/<NAME>/run.sh.
#
# Exports WALKIE_TASK_DIR / WALKIE_TASK for the named task, then hands off to the
# repo-root run.sh so all of its subcommands (start / fresh / reset / viewer /
# doctor) and the stale-viewer-port cleanup work unchanged, now task-aware.
#
# Usage (normally you call the per-task wrapper, not this directly):
#   tasks/_run.sh <task_dir> [run.sh subcommand]   # default subcommand: start
set -euo pipefail

TASK_DIR="${1:?usage: _run.sh <task_dir> [run.sh subcommand]}"
shift
TASK_DIR="$(cd "$TASK_DIR" && pwd)"
REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

export WALKIE_TASK_DIR="$TASK_DIR"
export WALKIE_TASK="$(basename "$TASK_DIR")"

echo "[task] $WALKIE_TASK -> $TASK_DIR"
exec "$REPO_ROOT/run.sh" "${@:-start}"
