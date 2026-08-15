#!/bin/sh
set -eu

python scripts/download_bge_model.py

exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    "$@"
