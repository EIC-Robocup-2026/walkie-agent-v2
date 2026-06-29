WALKIE_EXPLORE_VIZ=rerun WALKIE_EXPLORE_RERUN_SERVE=1 WALKIE_EXPLORE_RERUN_WEB_PORT=8008 uv run python main.py

uv run python -m services.realtime_explore.tools.reset -y

sudo apt install pavucontrol