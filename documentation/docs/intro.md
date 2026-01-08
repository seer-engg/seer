---
sidebar_position: 1
slug: /
---

# Getting Started with Seer

Seer is a **workflow builder with fine-grained control** for creating and executing automated workflows with integrated tools and services. Build complex automation workflows with visual editing, AI-assisted development, and seamless integrations (Google Workspace, GitHub, and more).

## Core Architecture Principle

**If workflows and agents are fundamentally different at the UI layer, they should be different at the API layer.**

This principle guides our API design: workflows (deterministic, node-based execution) and agents (dynamic, message-based conversations) have distinct mental models, data structures, and user needs. Rather than forcing unification through pattern matching or transformation layers, we maintain separate APIs and components that align with their fundamental differences. This reduces complexity, improves maintainability, and ensures each system can evolve independently.

## Quick Start

Get Seer running in 60 seconds:

```bash
git clone <repo> && cd seer
docker compose up
```

That's it! This starts all Docker services (Postgres, Redis, backend, worker), streams logs, and waits for readiness.

## Using the Workflow Editor

After running `docker compose up`, the workflow editor is available at:
- **Frontend**: http://localhost:5173/workflows?backend=http://localhost:8000
- **Backend API**: http://localhost:8000

## Configuration

Create a `.env` file in your project root:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional integrations (add as needed)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
TAVILY_API_KEY=...
```

Docker automatically configures `DATABASE_URL` and `REDIS_URL`.

For complete configuration options, see [Configuration Reference](./advanced/CONFIGURATION.md).

## Key Features

### 🛠️ Visual Workflow Builder
- Drag-and-drop interface for creating automation workflows
- Node-based editor with custom blocks and integrations
- Real-time workflow validation and execution

### 🤖 AI-Assisted Development
- Chat interface for workflow design and debugging
- AI suggestions for workflow improvements
- Intelligent error handling and recovery

### 🔗 Rich Integrations
- **Google Workspace**: Gmail, Drive, Sheets with OAuth
- **GitHub**: Repository management, issues, PRs
- **Web Tools**: Search, content fetching, APIs
- **Databases**: PostgreSQL with approval-based write controls

### ⚡ Advanced Execution Engine
- Streaming execution with real-time updates
- Interrupt handling for human-in-the-loop workflows
- Persistent state management with PostgreSQL

### 🔒 Enterprise-Ready
- Self-hosted or cloud deployment options
- OAuth-based authentication (Clerk integration)
- Role-based access control
- Audit trails and execution history

## Development Workflow

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
- **Taskiq Worker**: Run `uv run taskiq worker worker.broker:broker` to process triggers/polling/workflow runs

## Next Steps

- [Deploy to Railway](./deployment/RAILWAY.md) - Production deployment guide
- [Supabase Integration](./integrations/SUPABASE.md) - Multi-credential setup
- [Workflow Triggers](./advanced/TRIGGERS.md) - Event-driven execution
- [Configuration Reference](./advanced/CONFIGURATION.md) - Complete options
