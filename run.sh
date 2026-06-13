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
#   ./run.sh help           # this message
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

    help|-h|--help)
        sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
        ;;

    *)
        echo "unknown subcommand: $cmd" >&2
        echo "try: ./run.sh help" >&2
        exit 1
        ;;
esac
