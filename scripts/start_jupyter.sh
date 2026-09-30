#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
exec python -m jupyterlab --no-browser --ip=127.0.0.1 --port=8890 --allow-root
