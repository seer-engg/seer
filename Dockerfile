# LangGraph Standalone Server Dockerfile for Railway
# Based on official LangGraph API image
FROM langchain/langgraph-api:3.11

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install project in editable mode (if pyproject.toml exists)
RUN if [ -f pyproject.toml ]; then pip install --no-cache-dir -e .; fi

# Expose the default LangGraph API port
EXPOSE 8000

# The base image already sets the entrypoint to run LangGraph API server
# It will automatically detect langgraph.json files and serve the graphs
# Environment variables needed:
# - DATABASE_URI: Postgres connection string for checkpoints (e.g., postgresql://user:pass@host:port/db)
# - REDIS_URI: Optional Redis connection string for async tasks
# - LANGSMITH_API_KEY: Optional for tracing

