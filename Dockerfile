# Seer Backend Server Dockerfile
# Based on official LangGraph API image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install PostgreSQL client libraries required for psycopg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    postgresql-client \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
# NOTE: In development, docker-compose.yml mounts ./:/app as a volume,
# which overrides this COPY. The COPY is still needed for:
# 1. Production builds (no volume mount)
# 2. Initial dependency installation (uv sync runs during build)
# The volume mount in docker-compose.yml allows instant code updates without rebuilds.
COPY . /app

# Install project and dependencies from pyproject.toml
# This installs all dependencies listed in [project] dependencies section
# Dependencies are installed in the image, so they're available even with volume mounts
RUN uv sync

# IMPORTANT: remove the base image entrypoint that starts the API server
ENTRYPOINT []
# Expose the default API port
EXPOSE 8000

# Default command runs FastAPI server
# Can be overridden in docker-compose.yml
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]