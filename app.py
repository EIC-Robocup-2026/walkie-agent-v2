"""Flask application entrypoint."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Deferred import: create_app registers blueprints which instantiate
# and load all model singletons.
# ---------------------------------------------------------------------------
from api import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
