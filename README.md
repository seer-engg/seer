## Seer

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](https://github.com/seer-engg/seer/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/seer-engg/seer?style=social)](https://github.com/seer-engg/seer/stargazers)
[![DeepWiki](https://img.shields.io/badge/docs-DeepWiki-blue)](https://deepwiki.com/seer-engg/seer)
[![Discord](https://img.shields.io/badge/discord-join-7289DA?logo=discord&logoColor=white)](https://discord.gg/NuYsDdhJ)
[![Twitter Follow](https://img.shields.io/twitter/follow/get_seer?style=social)](https://x.com/get_seer)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-get--seer-0077B5?logo=linkedin)](https://www.linkedin.com/company/get-seer)

**Open-source workflow builder with cost controls and human oversight.** Build AI automations with tools like Gmail, GitHub, and Supabase - with built-in spend caps and approval gates.

## Quick Start

```bash
git clone https://github.com/seer-engg/seer
cd seer
cp .env.example .env
# Add your OPENAI_API_KEY to .env
docker compose up
```

That's it. Open http://localhost:5173 and start building workflows.

No accounts, no sign-ups — the app runs in local mode by default with no login required.

## What's Running

`docker compose up` starts everything:

| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | [localhost:5173](http://localhost:5173) | React app (Vite dev server) |
| **API** | [localhost:8000](http://localhost:8000) | FastAPI backend |
| **Worker** | — | Background task worker (Taskiq) |
| **Postgres** | 5432 | Database (pgvector) |
| **Valkey** | 6379 | Redis-compatible cache |
| **Browserless** | 3000 | Headless Chrome for browser tools |

API docs are at http://localhost:8000/docs.

## Configuration

Create `.env` from the example:

```bash
cp .env.example .env
```

**Required:**
```bash
OPENAI_API_KEY=sk-...       # Or any LLM provider key
```

**Optional integrations (add as needed):**
```bash
ANTHROPIC_API_KEY=sk-ant-... # Claude models
OPENROUTER_API_KEY=sk-or-... # Kimi, other models
GOOGLE_CLIENT_ID=...         # Google integrations (Gmail, Sheets)
GOOGLE_CLIENT_SECRET=...
BRAVE_SEARCH_API_KEY=...     # Web search tool
```

Docker automatically configures `DATABASE_URL` and `REDIS_URL`.

### Authentication

Auth mode is controlled by a single env var:

| `AUTH_PROVIDER` | Mode | Use case |
|-----------------|------|----------|
| `local` (default) | No login, single user | Self-hosted, personal use |
| `clerk` | Clerk auth with login | Multi-user, cloud deployment |

For `clerk` mode, also set `CLERK_JWKS_URL`, `CLERK_ISSUER`, `CLERK_SECRET_KEY`, and `VITE_CLERK_PUBLISHABLE_KEY`. See `.env.example` for details.

## Local Development (without Docker)

If you prefer running services directly:

```bash
# Start only infra
docker compose up postgres valkey browserless -d

# Backend (requires Python 3.12+, uv)
uv sync
uv run aerich upgrade
uv run uvicorn seer.api.main:app --reload --port 8000

# Worker
uv run taskiq worker seer.worker.broker:broker

# Frontend (requires Node 22+)
cd seer-frontend
npm install
npm run dev
```

## Project Layout

```
seer/
├── src/seer/              # Python backend
│   ├── api/               # FastAPI routers, middleware
│   ├── services/          # Business logic (workflows, triggers, integrations)
│   ├── core/              # Workflow compiler/runtime, schema models
│   ├── tools/             # Tool registry, executor, provider implementations
│   ├── agents/            # LangGraph-based agent orchestration
│   ├── worker/            # Taskiq background tasks, trigger polling
│   ├── database/          # Tortoise ORM models/config
│   └── auth/              # Auth provider abstraction (Clerk / local)
├── seer-frontend/         # React + TypeScript + Vite frontend
│   └── src/
├── migrations/            # Database migrations
├── tests/                 # Backend tests
├── scripts/               # Maintenance scripts
└── docker-compose.yml     # Full stack setup
```

## Migrations

Run migrations after pulling updates:

```bash
# Docker
docker compose exec api uv run aerich upgrade

# Local
uv run aerich upgrade
```

## Commands

```bash
docker compose up              # Start all services
docker compose up -d           # Start in background
docker compose logs -f api     # Follow API logs
docker compose down            # Stop all services
docker compose down -v         # Stop and remove data
uv run pytest                  # Run backend tests
```

## Why Seer?

- **Cost Controls** - Per-workflow spend caps and token limits prevent runaway AI expenses
- **Human Oversight** - Approval gates for critical operations; complete audit trails
- **Powerful Integrations** - Gmail, GitHub, Supabase, PostgreSQL with minimal permissions
- **AI-Native** - Chat interface for workflow design; intelligent error handling
- **Self-Hostable** - `git clone` + `docker compose up`. No accounts needed

## Documentation

[Complete Documentation](https://deepwiki.com/seer-engg/seer) - Comprehensive guides, architecture, and examples on DeepWiki

## License

AGPL-3.0 License - See [LICENSE](LICENSE).
