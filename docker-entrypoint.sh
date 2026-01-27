#!/bin/sh
set -e

# 1. API SERVER BLOCK
if [ "$1" = 'api' ]; then
    # Remove the first argument ("api") from the list
    shift

    echo "🚀 Starting FastAPI Server..."
    # Pass all remaining arguments ($@) to uvicorn
    # If you pass ["api", "--reload"], this executes:
    # uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    exec uv run uvicorn seer.api.main:app --host 0.0.0.0 --port 8000 "$@"
fi

# 2. WORKER BLOCK
if [ "$1" = 'worker' ]; then
    shift

    echo "Starting  Worker..."
    exec uv run taskiq worker seer.worker.broker:broker
fi

# Fallback for other commands
exec "$@"
