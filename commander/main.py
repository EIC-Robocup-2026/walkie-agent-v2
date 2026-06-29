"""Walkie Task Commander — LAN web UI to run RoboCup@Home challenges.

    cd commander && uv run python main.py

Binds 0.0.0.0:8083 (walkie-commander uses 8082) so it's reachable from any device
on the LAN. Launch it from a shell where you'd normally run ``./run.sh`` — the
challenge subprocesses inherit this process's environment (on the robot: a
ROS-sourced shell).
"""

from __future__ import annotations

import socket

from nicegui import app, ui

from walkie_runner.pages import create_page
from walkie_runner.process_manager import manager
from walkie_runner.registry import REPO_ROOT

PORT = 8083


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _banner() -> None:
    bar = "═" * 58
    print(f"\n  ╔{bar}╗")
    print("  ║  Walkie Task Commander".ljust(61) + "║")
    print(f"  ╠{bar}╣")
    print(f"  ║  Local:    http://localhost:{PORT}".ljust(61) + "║")
    print(f"  ║  Network:  http://{_lan_ip()}:{PORT}".ljust(61) + "║")
    print(f"  ║  Repo:     {REPO_ROOT}"[:61].ljust(61) + "║")
    print(f"  ╚{bar}╝\n")


async def _on_shutdown() -> None:
    print("Shutting down — stopping running challenges…")
    await manager.shutdown()


create_page()
app.on_shutdown(_on_shutdown)

if __name__ in {"__main__", "__mp_main__"}:
    _banner()
    ui.run(
        title="Walkie Task Commander",
        dark=True,
        reload=False,
        show=False,
        host="0.0.0.0",
        port=PORT,
        favicon="🤖",
    )
