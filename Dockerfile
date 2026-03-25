# Seer Backend Server Dockerfile
# Multi-stage build: builder installs deps, runtime copies only what's needed

# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build-only system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set version for setuptools_scm (no .git in build context)
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SEER=0.1.4
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.4

# Layer cache: install deps before copying source
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy source and reinstall (picks up the project itself)
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime-only system deps (no git, no curl)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    postgresql-client \
    # WeasyPrint system dependencies (HTML → PDF conversion)
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz0b \
    libfontconfig1 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    shared-mime-info && \
    rm -rf /var/lib/apt/lists/*

# Copy uv and virtual environment from builder
COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /root/.local/bin/uvx /root/.local/bin/uvx
ENV PATH="/root/.local/bin:$PATH"

# Copy app with installed deps
COPY --from=builder /app /app

# Set version env vars for runtime
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SEER=0.1.4
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.4

# Install Playwright browser for browser automation node
RUN --mount=type=cache,target=/root/.cache/ms-playwright \
    uv run playwright install-deps chromium && \
    uv run playwright install chromium

EXPOSE 8000

COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

CMD ["api"]
