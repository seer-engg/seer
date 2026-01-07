#!/bin/bash
set -e

echo "Running database migrations..."
uv run alembic upgrade head
echo "Migrations completed successfully"
