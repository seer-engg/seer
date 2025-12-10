# LangGraph Standalone Server Dockerfile for Railway
# Based on official LangGraph API image
FROM langchain/langgraph-api:3.13

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install project and dependencies from pyproject.toml
# This installs all dependencies listed in [project] dependencies section
RUN pip install --no-cache-dir -e .

# Expose the default LangGraph API port
EXPOSE 8000

# The base image already sets the entrypoint to run LangGraph API server
# It will automatically detect langgraph.json files and serve the graphs
# Environment variables needed:
# - DATABASE_URI: Postgres connection string for checkpoints (e.g., postgresql://user:pass@host:port/db)
# - REDIS_URI: Optional Redis connection string for async tasks
# - LANGFUSE_SECRET_KEY: Optional for tracing (Langfuse self-hosted)

