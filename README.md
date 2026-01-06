## Seer

Seer is a **workflow builder with fine-grained control** for creating and executing automated workflows with integrated tools and services. Build complex automation workflows with visual editing, AI-assisted development, and seamless integrations (Google Workspace, GitHub, and more).

### Core Architecture Principle

**If workflows and agents are fundamentally different at the UI layer, they should be different at the API layer.**

This principle guides our API design: workflows (deterministic, node-based execution) and agents (dynamic, message-based conversations) have distinct mental models, data structures, and user needs. Rather than forcing unification through pattern matching or transformation layers, we maintain separate APIs and components that align with their fundamental differences. This reduces complexity, improves maintainability, and ensures each system can evolve independently.

### Quick Start

```bash
git clone <repo> && cd seer
docker compose up
```

That's it! Starts Docker services (Postgres, Redis, backend, worker), streams logs, and waits for readiness.

### Deploy to Railway

Deploy Seer to Railway with one click:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/[TEMPLATE-ID])

**What gets deployed:** FastAPI backend, background worker, PostgreSQL, and Redis

**Setup:** Click button, enter `OPENAI_API_KEY`, wait 5-7 minutes. Estimated cost: $15-30/month.

For detailed deployment instructions, see [Railway Deployment Guide](./docs/deployment/RAILWAY.md).

### Using the Workflow Editor

After running `docker compose up`, the workflow editor is available at:
- **Frontend**: http://localhost:5173/workflows?backend=http://localhost:8000
- **Backend API**: http://localhost:8000

### Configuration

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional integrations (add as needed)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
TAVILY_API_KEY=...
```

Docker automatically configures `DATABASE_URL` and `REDIS_URL`.

For complete configuration options, see [Configuration Reference](./docs/advanced/CONFIGURATION.md).

### Usage

**Start development environment:**
```bash
docker compose up
```

**View logs:**
```bash
docker compose logs -f
```

**Stop services:**
```bash
docker compose down
```

### Development Workflow

**Steps:**
1. Run: `docker compose up`
2. Code changes hot-reload via volume mounts (uvicorn --reload)
3. Access workflow builder at: http://localhost:5173/workflows?backend=http://localhost:8000
4. View logs in the terminal or run: `docker compose logs -f`
5. Stop: `docker compose down`

**Services started:**
- **Backend API** (port 8000): FastAPI server with workflow execution engine
- **Postgres** (port 5432): Workflow and user data persistence
- **Redis** (port 6379): Taskiq message broker
- **Taskiq Worker**: run `uv run taskiq worker worker.broker:broker` (or use Docker) to process triggers/polling/workflow runs

### Integrations & API Keys

**Core Requirements:**
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` - Required for workflow execution and AI assistance

**Optional Integrations:**
- **Google Workspace** - `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (Gmail, Drive, Sheets)
- **GitHub** - `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` (Repos, Issues, PRs)
- **Supabase** - `SUPABASE_CLIENT_ID`, `SUPABASE_CLIENT_SECRET` ([Setup Guide](./docs/integrations/SUPABASE.md))
- **Web Search** - `TAVILY_API_KEY`

For complete configuration options, see [Configuration Reference](./docs/advanced/CONFIGURATION.md).

### Key Features

**🛠️ Visual Workflow Builder**
- Drag-and-drop interface for creating automation workflows
- Node-based editor with custom blocks and integrations
- Real-time workflow validation and execution

**🤖 AI-Assisted Development**
- Chat interface for workflow design and debugging
- AI suggestions for workflow improvements
- Intelligent error handling and recovery

**🔗 Rich Integrations**
- **Google Workspace**: Gmail, Drive, Sheets with OAuth
- **GitHub**: Repository management, issues, PRs
- **Web Tools**: Search, content fetching, APIs
- **Databases**: PostgreSQL with approval-based write controls

**⚡ Advanced Execution Engine**
- Streaming execution with real-time updates
- Interrupt handling for human-in-the-loop workflows
- Persistent state management with PostgreSQL

**🔒 Enterprise-Ready**
- Self-hosted or cloud deployment options
- OAuth-based authentication (Clerk integration)
- Role-based access control
- Audit trails and execution history

### Documentation

- [Quick Start](#quick-start) - Get running in 60 seconds
- [Railway Deployment](./docs/deployment/RAILWAY.md) - Production deployment guide
- [Worker Setup](./worker/README.md) - Background task worker configuration
- [Integrations](./docs/integrations/SUPABASE.md) - Google, GitHub, Supabase setup
- [Advanced Features](./docs/advanced/) - Triggers, proposals, and more
- [Configuration Reference](./docs/advanced/CONFIGURATION.md) - Complete configuration options
- [Complete Documentation](./docs/) - Full documentation index
