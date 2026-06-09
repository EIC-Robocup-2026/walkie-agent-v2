#!/usr/bin/env bash
# Launch Walkie configured for this task. Thin wrapper over tasks/_run.sh.
#   ./run.sh            # start
#   ./run.sh fresh      # wipe scene+object DBs, then start
#   ./run.sh reset      # wipe DBs only
#   ./run.sh viewer     # standalone Chroma viewer
# Any run.sh subcommand from the repo root works here, now task-aware.
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec "$TASK_DIR/../_run.sh" "$TASK_DIR" "$@"
