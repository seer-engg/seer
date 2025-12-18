# LangGraph Standalone Server Dockerfile for Railway
# Based on official LangGraph API image
FROM langchain/langgraph-api:3.13

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install project and dependencies from pyproject.toml
# This installs all dependencies listed in [project] dependencies section
RUN uv sync

# IMPORTANT: remove the base image entrypoint that starts the API server
ENTRYPOINT []
# Expose the default LangGraph API port
EXPOSE 8000

CMD ["uv", "run", "langgraph", "dev", "--port", "8000", "--host", "0.0.0.0", "--config", "langgraph.json", "--no-browser", "--no-reload"]