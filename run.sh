#!/usr/bin/env bash
# Unified launcher for walkie-agent-v2.
#
# 'start' brings the robot up ready to take commands immediately — there is no
# explore stage and nothing to press Enter for. The walkie_graphs scene memory
# builds and updates itself in the background while you talk to the robot, so it
# can see, remember, and act from the first second.
#
# Usage:
#   ./run.sh                # start the agent
#   ./run.sh start          # same as above
#   ./run.sh reset          # wipe the walkie_graphs store (chroma + pcds + captures + bg), no prompt
#   ./run.sh fresh          # reset, then start the agent
#   ./run.sh task <NAME>    # run a single task (e.g. ./run.sh task HRI)
#   ./run.sh tasks          # list available tasks
#   ./run.sh help           # this message
#
# Each task lives under tasks/<NAME>/run.py and is launched as a module
# (uv run python -m tasks.<NAME>.run) so its relative imports resolve.
#
# The walkie_graphs store is the only long-term memory backend; 'reset' wipes it
# via services/walkie_graphs/tools/reset.py. Run reset with the robot stopped —
# ChromaDB's persistent client is single-process.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

cmd="${1:-start}"

reset_store() {
    uv run python -m services.walkie_graphs.tools.reset -y
}

list_tasks() {
    for d in tasks/*/run.py; do
        [ -e "$d" ] || continue
        basename "$(dirname "$d")"
    done
}

case "$cmd" in
    start)
        exec uv run python main.py
        ;;

    reset)
        reset_store
        ;;

    fresh)
        reset_store
        exec uv run python main.py
        ;;

    task)
        name="${2:-}"
        if [ -z "$name" ]; then
            echo "usage: ./run.sh task <NAME>" >&2
            echo "available tasks:" >&2
            list_tasks | sed 's/^/  /' >&2
            exit 1
        fi
        if [ ! -e "tasks/$name/run.py" ]; then
            echo "unknown task: $name" >&2
            echo "available tasks:" >&2
            list_tasks | sed 's/^/  /' >&2
            exit 1
        fi
        exec uv run python -m "tasks.$name.run"
        ;;

    tasks)
        list_tasks
        ;;

    help|-h|--help)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        ;;

    *)
        echo "unknown subcommand: $cmd" >&2
        echo "try: ./run.sh help" >&2
        exit 1
        ;;
esac
