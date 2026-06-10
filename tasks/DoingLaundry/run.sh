#!/usr/bin/env bash
# Launch Walkie for the Doing Laundry Challenge (retrieve + fold clothes) — rulebook §5.4
#   ./run.sh            # start    ./run.sh fresh   # wipe DBs then start
# Any repo-root run.sh subcommand works here, now task-aware.
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec "$TASK_DIR/../_run.sh" "$TASK_DIR" "$@"
