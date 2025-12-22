# Seer Backend Server Dockerfile
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
# Expose the default API port
EXPOSE 8000

# Default command runs FastAPI server
# Can be overridden in docker-compose.yml
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]