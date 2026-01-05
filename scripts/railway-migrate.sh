#!/bin/bash
set -e

echo "Running database migrations..."
uv run aerich upgrade
echo "Migrations completed successfully"
