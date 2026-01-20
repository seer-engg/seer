## Seer

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/seer-engg/seer/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/seer-engg/seer?style=social)](https://github.com/seer-engg/seer/stargazers)
[![Documentation](https://img.shields.io/badge/docs-docs.getseer.dev-blue)](https://docs.getseer.dev)
[![Discord](https://img.shields.io/badge/discord-join-7289DA?logo=discord&logoColor=white)](https://discord.gg/NuYsDdhJ)
[![Twitter Follow](https://img.shields.io/twitter/follow/get_seer?style=social)](https://x.com/get_seer)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-get--seer-0077B5?logo=linkedin)](https://www.linkedin.com/company/get-seer)

Seer is a **open-source workflow builder with built-in oversight and cost controls** for creating and executing automated workflows with integrated tools and services.

## Quick Start (Docker)

1) Clone and start the stack (Postgres, Redis, API, worker):
```bash
git clone https://github.com/seer-engg/seer
cd seer
docker compose up
```

2) Access the app:
- Frontend: http://localhost:5173/workflows?backend=http://localhost:8000
- Backend API: http://localhost:8000

## Local Development (without full Docker)

- Prereqs: Python 3.12+, [uv](https://github.com/astral-sh/uv) installed (`pip install uv`), Postgres + Redis running (use `docker compose up postgres redis`).
- Install deps: `uv sync`
- Run API: `uv run uvicorn seer.api.main:app --reload --port 8000`
- Run worker: `uv run taskiq worker seer.worker.broker:broker`
- Run tests: `uv run pytest`

## Project Layout (backend)

- `src/seer/api/` – FastAPI routers, middleware, API models (workflows, tools, integrations, triggers, agents).
- `src/seer/services/` – business logic used by API/worker (workflow execution, triggers, integrations).
- `src/seer/core/` – workflow compiler/runtime, schema models, global compiler singleton.
- `src/seer/tools/` – tool registry, executor, credential resolver, provider implementations; `src/seer/tool_hub/` for tool index/search.
- `src/seer/worker/` – Taskiq worker, background tasks, trigger polling.
- `src/seer/agents/` – agent orchestration (LangGraph-based workflow agent).
- `src/seer/database/` – Tortoise ORM models/config; migrations live in `/migrations`.
- `src/seer/analytics/`, `src/seer/observability/`, `src/seer/utilities/` – shared instrumentation and helpers.
- `documentation/` – docs site assets; `scripts/` – maintenance helpers; `tests/` – automated tests.

## Configuration

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

Helpful commands:
- Start everything: `docker compose up`
- Follow logs: `docker compose logs -f`
- Stop services: `docker compose down`
- Run API locally: `uv run uvicorn seer.api.main:app --reload --port 8000`
- Run worker locally: `uv run taskiq worker seer.worker.broker:broker`
- Tests: `uv run pytest`

### Key Features

**🛠️ Visual Workflow Builder**
- Drag-and-drop interface for creating automation workflows
- Visual editor with custom blocks and integrations
- Real-time workflow validation and execution

**💰 Cost Governance**
- Built-in spend caps and token limits per workflow
- Budget controls to prevent runaway AI expenses
- Transparent cost tracking and reporting
- Per-workflow and account-level spending limits

**👁️ Execution Transparency**
- Complete run history for every workflow execution
- Detailed tracing with timing and outputs
- Human oversight checkpoints for critical decisions
- Audit trails and execution logs

**🤖 AI-Assisted Automation**
- Chat interface for workflow design and debugging
- AI suggestions for workflow improvements
- Intelligent error handling and recovery

**🔗 Rich Integrations**
- **Google Workspace**: Gmail, Drive, Sheets with minimal permissions
- **GitHub**: Repository management, issues, PRs
- **Web Tools**: Search, content fetching, APIs
- **Databases**: PostgreSQL with approval-based write controls

**🔒 Self-Hostable**
- Self-hosted or cloud deployment options
- Minimal permissions approach (read-only first)
- OAuth-based authentication (Clerk integration)
- Role-based access control

### Documentation

📚 **[Complete Documentation](https://docs.getseer.dev)** - Full docs site with guides, API reference, and examples

- [Quick Start](#quick-start-docker) - Get running in 60 seconds
- [Architecture](https://docs.getseer.dev) - Backend overview and concepts
- [Worker Setup](src/seer/worker/README.md) - Background task worker configuration
- [Integrations](https://docs.getseer.dev/integrations/SUPABASE) - Google, GitHub, Supabase setup
- [Advanced Features](https://docs.getseer.dev/advanced/TRIGGERS) - Triggers, proposals, and more
- [Configuration Reference](https://docs.getseer.dev/advanced/CONFIGURATION) - Complete configuration options

### License

Seer is open source under the MIT license. Enterprise features (if any exist)
reside in the `ee/` directory and are licensed separately. See LICENSE for details.
