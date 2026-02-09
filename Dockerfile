# Seer Backend Server Dockerfile
# Based on official LangGraph API image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install PostgreSQL client libraries required for psycopg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    postgresql-client \
    git \
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


# Set environment variable to specify project version for setuptools_scm
# override in CI/CD for actual release builds
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SEER=0.1.4
# Generic fallback for build contexts without .git metadata
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.4

# Install project and dependencies from pyproject.toml
# This installs all dependencies listed in [project] dependencies section
# Dependencies are installed in the image, so they're available even with volume mounts
RUN uv sync

# Install Playwright browser for browser automation node
# 1. Install system dependencies (fonts, graphics libraries) required by Chromium
# 2. Install Chromium browser binary (~280MB)
# Note: Only Chromium is installed to minimize image size; add firefox/webkit if needed
RUN uv run playwright install-deps chromium && \
    uv run playwright install chromium

# Expose the default API port
EXPOSE 8000

COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

CMD ["api"]
