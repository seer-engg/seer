# Docker Setup for Seer Backend

This guide explains how to run the Seer backend locally using Docker Compose, and connect it to the closed-source frontend.

## Quick Start

1. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start all services:**
   ```bash
   seer dev
   ```
   
   Or manually:
   ```bash
   docker-compose up -d
   ```

3. **The `seer dev` command will automatically:**
   - Start Postgres, MLflow, and the backend server
   - Wait for services to be ready
   - Open your browser to: `http://localhost:5173/workflows?backend=http://localhost:8000`

## Command Options

```bash
# Use default URLs (frontend: localhost:5173, backend: localhost:8000)
seer dev

# Custom frontend URL
seer dev --frontend-url http://localhost:3000

# Custom backend URL
seer dev --backend-url http://localhost:9000

# Don't open browser automatically
seer dev --no-browser
```

## Services

- **Postgres** (port 5432): Database for workflows, users, and OAuth connections
- **MLflow** (port 5000): ML experiment tracking
- **Backend API** (port 8000): FastAPI server with workflow endpoints

## Environment Variables

Required in `.env`:

```bash
# Database (auto-configured in docker-compose)
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=seer
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/seer

# Backend API
PORT=8000

# MLflow (auto-configured)
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=langgraph-local

# Other required vars (see .env.example)
OPENAI_API_KEY=your_key
COMPOSIO_API_KEY=your_key
# ... etc
```

## Connecting Frontend to Self-Hosted Backend

The frontend (closed-source) can connect to your self-hosted backend using a URL parameter:

```
http://localhost:5173/workflows?backend=http://localhost:8000
```

The frontend will:
1. Read the `backend` query parameter
2. Use it as the API base URL instead of `VITE_BACKEND_API_URL`
3. Allow you to create and execute workflows using your local backend

## Database Migrations

After starting Postgres for the first time, run migrations:

```bash
# Inside the backend container or locally
docker-compose exec langgraph-server uv run aerich upgrade
```

Or if running locally (not in Docker):
```bash
uv run aerich upgrade
```

## Troubleshooting

### Backend won't start
- Check logs: `docker-compose logs langgraph-server`
- Verify Postgres is healthy: `docker-compose ps`
- Check DATABASE_URL is correct

### Frontend can't connect
- Verify backend is running: `curl http://localhost:8000/health`
- Check CORS settings in backend
- Ensure backend URL in query parameter matches your backend

### Database connection errors
- Wait for Postgres to be healthy before starting backend
- Check Postgres logs: `docker-compose logs postgres`
- Verify DATABASE_URL format: `postgresql://user:pass@host:port/db`

## Development Workflow

1. Start services: `./scripts/start-dev.sh`
2. Frontend opens automatically with backend parameter
3. Create workflows in the visual editor
4. Execute workflows against your local backend
5. View execution history and logs

## Stopping Services

```bash
docker-compose down
```

To remove volumes (clears database):
```bash
docker-compose down -v
```

