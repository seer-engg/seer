## Seer

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/seer-engg/seer/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/seer-engg/seer?style=social)](https://github.com/seer-engg/seer/stargazers)
[![DeepWiki](https://img.shields.io/badge/docs-DeepWiki-blue)](https://deepwiki.com/seer-engg/seer)
[![Discord](https://img.shields.io/badge/discord-join-7289DA?logo=discord&logoColor=white)](https://discord.gg/NuYsDdhJ)
[![Twitter Follow](https://img.shields.io/twitter/follow/get_seer?style=social)](https://x.com/get_seer)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-get--seer-0077B5?logo=linkedin)](https://www.linkedin.com/company/get-seer)

**Open-source workflow builder with cost controls and human oversight.** Build AI automations with tools like Gmail, GitHub, and Supabase - with built-in spend caps and approval gates.

## Quick Start (Docker)

1) Clone and start the stack (Postgres, Valkey, API, worker):
```bash
git clone https://github.com/seer-engg/seer
cd seer
docker compose up
```

2) Run database migrations:
```bash
docker compose exec api uv run aerich upgrade
```

3) Access the app:
- **Cloud Frontend (Default):** Visit http://localhost:8000 → redirects to https://app.getseer.dev
- **Local Frontend:** Set `FRONTEND_URL=http://localhost:5173` in `.env` file
- API Docs: http://localhost:8000/docs
- Backend Health: http://localhost:8000/health

Your browser will automatically open and connect the cloud frontend to your local backend. Sign in with Clerk to start using Seer.

## Local Development (without full Docker)

- Prereqs: Python 3.12+, [uv](https://github.com/astral-sh/uv) installed (`pip install uv`), Postgres + Valkey running (use `docker compose up postgres valkey`).
- Install deps: `uv sync`
- Run API: `uv run uvicorn seer.api.main:app --reload --port 8000`
- Run worker: `uv run taskiq worker seer.worker.broker:broker`
- Run tests: `uv run pytest`

## Project Layout (backend)

- `src/seer/api/` – FastAPI routers, middleware, API models (workflows, tools, integrations, triggers, agents).
- `src/seer/services/` – business logic used by API/worker (workflow execution, triggers, integrations).
- `src/seer/core/` – workflow compiler/runtime, schema models, global compiler singleton.
- `src/seer/tools/` – tool registry, executor, credential resolver, provider implementations.
- `src/seer/worker/` – Taskiq worker, background tasks, trigger polling.
- `src/seer/agents/` – agent orchestration (LangGraph-based workflow agent).
- `src/seer/database/` – Tortoise ORM models/config; migrations live in `/migrations`.
- `src/seer/analytics/`, `src/seer/observability/`, `src/seer/utilities/` – shared instrumentation and helpers.
- `scripts/` – maintenance helpers; `tests/` – automated tests.

## Configuration

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional integrations (add as needed)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
BRAVE_SEARCH_API_KEY=...
```

Docker automatically configures `DATABASE_URL` and `REDIS_URL`.

**Commands:**
```bash
docker compose up                # Start all services
docker compose logs -f           # Follow logs
uv run pytest                    # Run tests
```

## Migrations

> Run migrations manually after pulling updates.

```bash
# Docker
docker compose exec api uv run aerich upgrade

# Local
uv run aerich upgrade
```

### Why Seer?

**💰 Cost Controls** - Per-workflow spend caps and token limits prevent runaway AI expenses

**👁️ Human Oversight** - Approval gates for critical operations; complete audit trails

**🔗 Powerful Integrations** - Gmail, GitHub, Supabase, PostgreSQL with minimal permissions

**🤖 AI-Native** - Chat interface for workflow design; intelligent error handling

**🔒 Self-Hostable** - Deploy anywhere; full control over your data

### Documentation

📚 **[Complete Documentation](https://deepwiki.com/seer-engg/seer)** - Comprehensive guides, architecture, and examples on DeepWiki

- [Quick Start](#quick-start-docker) - Get running in 60 seconds
- [Worker Setup](src/seer/worker/README.md) - Background task worker configuration

### License

MIT License - 100% open source. See [LICENSE](LICENSE).
