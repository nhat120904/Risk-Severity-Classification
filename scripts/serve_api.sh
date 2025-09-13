#!/usr/bin/env bash
set -euo pipefail

# Run FastAPI server for RightShip Risk Classifier

cd "$(dirname "$0")/.."

export PYTHONPATH=.

exec uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload


