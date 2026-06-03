#!/usr/bin/env bash
# Unified launcher for walkie-agent-v2.
#
# 'start' brings the robot up ready to take commands immediately — there is no
# explore stage and nothing to press Enter for. The CLIP scene memory builds and
# updates itself in the background while you talk to the robot, so it can see,
# remember, and act from the first second.
#
# Usage:
#   ./run.sh                # start the agent (viewer auto-starts in-process)
#   ./run.sh start          # same as above
#   ./run.sh reset          # wipe both vector DBs (object + scene), no prompt
#   ./run.sh reset-scene    # wipe only CLIP scene memory (chroma_db_scene)
#   ./run.sh reset-object   # wipe only the legacy object DB (chroma_db)
#   ./run.sh fresh          # reset both, then start the agent
#   ./run.sh viewer         # standalone viewer (snapshot copy — safe while agent runs)
#   ./run.sh doctor         # diagnose a corrupt/desynced store (read-only)
#   ./run.sh help           # this message
#
# The agent already auto-starts the in-process Chroma viewer (CHROMA_VIEWER_*
# in config.toml). 'viewer' here is the standalone fallback that opens a
# snapshot copy — useful when the agent isn't running, or for browsing without
# touching the live SQLite files.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

cmd="${1:-start}"

# Free the in-process DB viewer port before launching. A previous run whose
# shutdown hung (interrupt during a blocking robot/threading call) can leave
# main.py alive holding the port, so the next start dies with
# "Port 8500 is in use by another program". Kill whatever still holds it so the
# launch is reliable. Honours CHROMA_VIEWER_PORT; default 8500.
free_viewer_port() {
    local port="${CHROMA_VIEWER_PORT:-8500}"
    local pids
    pids="$(lsof -ti:"$port" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "[run] port $port held by stale process(es): $pids — terminating"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
        sleep 1
        pids="$(lsof -ti:"$port" 2>/dev/null || true)"
        if [ -n "$pids" ]; then
            # shellcheck disable=SC2086
            kill -9 $pids 2>/dev/null || true
            sleep 1
        fi
    fi
}

case "$cmd" in
    start)
        free_viewer_port
        exec uv run python main.py
        ;;

    reset)
        uv run python -m tools.reset_db --all -y
        ;;

    reset-scene)
        uv run python -m tools.reset_db --scene -y
        ;;

    reset-object)
        uv run python -m tools.reset_db --object -y
        ;;

    fresh)
        uv run python -m tools.reset_db --all -y
        free_viewer_port
        exec uv run python main.py
        ;;

    viewer)
        # Snapshot mode is the default — safe to run alongside main.py.
        # Override with CHROMA_VIEWER_LIVE=1 only if the agent is stopped.
        exec uv run python -m tools.chroma_viewer
        ;;

    doctor)
        exec uv run python -m tools.db_doctor --scene
        ;;

    help|-h|--help)
        sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
        ;;

    *)
        echo "unknown subcommand: $cmd" >&2
        echo "try: ./run.sh help" >&2
        exit 1
        ;;
esac
